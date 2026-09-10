"""The LangGraph agent loop.

    START -> agent -> [tool calls?] -> tools -> observe -> agent -> ... -> END
                          |
                          +-- plain text --> END

Three nodes, one conditional edge, and a back-edge:

`agent`   (plan)     Sends the conversation to Claude and gets back either a
                     final piece of text or one or more tool calls. It never
                     touches Todoist. Which *model* it uses depends on the mode
                     the observe step set: the tool-bound one for acting, a
                     tool-less one when the verdict was "ask the user" -- so a
                     clarifying question cannot turn into a guessed tool call.
                     Its system prompt carries the habits memory found relevant
                     to this message, so the plan can account for them.

`tools`   (act)      Runs every requested tool and appends one ToolMessage per
                     call. It decides nothing; it records what happened --
                     success, or which *kind* of failure -- in the message's
                     `artifact`, a side channel the model never sees. The
                     artifact also carries the tool's event (the task as
                     Todoist returned it), for memory.

`observe` (replan)   Reads those artifacts and turns each into a verdict using a
                     fixed table (`classify`): done, retry, ask, or fail. It
                     applies the retry budget, pauses before retrying a rate
                     limit, appends a one-line verdict to the tool result so the
                     model knows what to do, and sets the next turn's mode.

`_route`             After `agent`: tool calls -> `tools`; plain text -> END;
                     and a hard stop at MAX_TOOL_LOOPS.

The split is deliberate. The *policy* -- retry, or ask, or give up -- is code:
in one place, unit-tested, explainable. Claude only decides *how*: what the
corrected call is, how to phrase the question, how to report the failure.

Memory sits *around* the graph, in `run()`, not inside it. Before a run, the
last few messages of the chat become the conversation so far and the relevant
habits become a note in the prompt; after it, the exchange is logged and the
tool events are learned from. The graph itself is pure: given the same
messages it does the same thing, which is what keeps it testable without a
database and keeps every read and write of memory in one place.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import ValidationError

import config
import todoist
from memory import Memory
from tools import TOOLS, NeedsClarification, UnexpectedResult

_TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}

# Claude requires the first message to be the user's. A conversation window can
# begin with a check-in Nexus sent on its own; this stands in for the clock.
CHECK_IN_MARKER = "[scheduled check-in]"

SYSTEM_PROMPT = """You are Nexus, the user's personal task assistant. You manage \
their Todoist.

You are talking over Telegram, so keep replies to one or two short sentences of \
plain text. No markdown headings and no bullet points, except when you are \
listing tasks.

Right now it is {now}.

You have five tools: create_task, list_tasks, complete_task, update_task, \
delete_task.

Rules:
- Use a tool whenever the user asks about or wants to change their tasks. Never \
invent a task name, id, or due date -- take them from what the tools return.
- If the user is vague about which task or what the task is ("remind me about \
the thing", "move that one"), ask one short clarifying question instead of \
guessing. Never \
create a task whose content is a placeholder.
- The messages before the latest one are the recent conversation, including \
check-ins you sent on a schedule (a "{marker}" line stands for the clock that \
sent one). "It" and "that one" usually refer to something in them.
- Pass due dates in the user's own words ("tomorrow at 3pm", "next friday"). \
Todoist parses them; do not convert them to a date yourself.
- Priorities use Todoist's p-scale: p1 is the most urgent, p4 is normal.
- A tool result may end with a line starting "Observer:". That is the loop's \
verdict on the call -- follow it. It says either to retry with a specific \
correction, or to ask the user, or that nothing more can be done.
- Never tell the user something worked when the tool said it did not. After a \
tool succeeds, confirm what actually happened using the values it returned, \
e.g. "Added 'Submit iTrade PR review', due Tue 9 Sep 3:00pm".{memory}{mode_note}"""

MODE_NOTES = {
    "act": "",
    "retry": (
        "\n\nThis turn: the observer asked for a retry. Follow its repair note exactly "
        "-- call the tool it names with the corrected input. If you cannot correct "
        "it, ask the user instead."
    ),
    "respond": (
        "\n\nThis turn you cannot use tools. Your last tool call did not succeed and "
        "the observer has decided a retry will not help. Either ask the user ONE "
        "short question they can answer in a single reply, or tell them plainly "
        "what failed. Do not claim anything was done."
    ),
}


class AgentState(MessagesState):
    """MessagesState gives us `messages` with append-on-update semantics.

    `steps`       how many times we have been through the agent node this run;
                  `_route` uses it as the loop's safety valve.
    `retries`     how many retries the observe step has granted this run.
    `mode`        what the *next* agent turn is for -- "act" (tools bound),
                  "retry" (tools bound, follow the repair note) or "respond"
                  (no tools).
    `memory_note` the habits relevant to this message, already rendered for
                  the system prompt; empty when memory has nothing to say.
    """

    steps: int
    retries: int
    mode: str
    memory_note: str


# --- The replan table ---------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    action: Literal["done", "retry", "ask", "fail"]
    note: str  # appended to the tool result, for the model
    backoff: bool = False  # pause before the retry: a rate limit or an outage


def classify(artifact: dict, retries_used: int) -> Verdict:
    """Decide what a tool outcome means for the loop. Pure, so it is trivially
    testable and can be read as a table.

    The questions, in order:
      1. Did it work?                                   -> done
      2. Is this something only the user can resolve?   -> ask
      3. Could a corrected (or repeated) call fix it,
         and is there retry budget left?                -> retry, else ask/fail
      4. Otherwise                                      -> fail, honestly

    "ask" and "fail" both end the tool loop for this message; they differ only
    in what the model is told to say.
    """
    outcome = artifact.get("outcome", "crash")
    budget_left = retries_used < config.MAX_RETRIES

    if outcome == "ok":
        return Verdict("done", "")

    if outcome == "clarify":
        return Verdict("ask", "only the user can resolve this; ask them, do not retry.")

    if outcome == "unexpected":
        # A half-success: something happened, but not what was asked. The tool
        # said how to repair it -- and the repair is usually not a plain redo.
        repair = artifact.get("repair") or "correct the input and try once more."
        if budget_left:
            return Verdict("retry", f"RETRY once. {repair}")
        return Verdict("ask", "the retry did not fix it either; ask the user how they want it.")

    if outcome == "bad_args":
        if budget_left:
            return Verdict("retry", "RETRY once with valid arguments -- see the error above.")
        return Verdict("ask", "the arguments were invalid twice; ask the user for the detail.")

    if outcome == "api_error":
        status = artifact.get("status_code")
        if status in (401, 403):
            return Verdict(
                "fail",
                "Todoist refused our credentials; no retry can help. Tell the user to "
                "check API_TOKEN_TODOIST.",
            )
        if status == 404:
            return Verdict("ask", "that task no longer exists; tell the user and ask what to do.")
        if status == 400:
            if budget_left:
                return Verdict(
                    "retry",
                    "RETRY once with corrected input -- Todoist rejected something in the "
                    "request; the error above says what.",
                )
            return Verdict("ask", "Todoist rejected the corrected input too; ask the user.")
        # 429, 5xx, or no status at all (never reached Todoist): transient.
        if budget_left:
            return Verdict(
                "retry", "RETRY once with the same input; this looks temporary.", backoff=True
            )
        return Verdict(
            "fail",
            "still failing after a retry; tell the user Todoist is not responding and to "
            "try again later.",
        )

    return Verdict("fail", "an internal error; tell the user plainly and do not retry.")


# --- Nodes --------------------------------------------------------------------


def _system_prompt(mode: str = "act", memory_note: str = "") -> str:
    # Rebuilt per call so the model always knows today's date -- it needs that to
    # sanity-check and echo back due dates like "tomorrow at 3pm".
    now = datetime.now(config.TIMEZONE)
    return SYSTEM_PROMPT.format(
        now=now.strftime("%A %d %B %Y, %H:%M %Z"),
        marker=CHECK_IN_MARKER,
        memory=memory_note,
        mode_note=MODE_NOTES.get(mode, ""),
    )


def _agent_node(state: AgentState, acting, responding) -> dict:
    """Plan: hand the conversation to Claude and let it decide what to do.

    In "respond" mode the model has no tools bound at all. That is what turns
    "ask the user" from a hope into a guarantee: the model *cannot* answer a
    clarifying-question turn with another guessed tool call.
    """
    mode = state.get("mode", "act")
    model = responding if mode == "respond" else acting
    # The system prompt is prepended rather than stored in state, so it is not
    # duplicated every time we come back round the loop.
    prompt = SystemMessage(_system_prompt(mode, state.get("memory_note", "")))
    reply = model.invoke([prompt] + state["messages"])
    return {"messages": [reply], "steps": state.get("steps", 0) + 1, "mode": "act"}


def _tool_node(state: AgentState) -> dict:
    """Act: run the requested tools and record, truthfully, what happened.

    Nothing in here raises and nothing in here decides. Each ToolMessage carries
    the human-readable result for the model *and* an `artifact` -- a structured
    record of the outcome kind, plus the tool's event, that the model never
    sees. The observe node turns the outcome into a verdict; `run()` hands the
    event to memory.
    """
    observations = []

    for call in state["messages"][-1].tool_calls:
        tool = _TOOLS_BY_NAME.get(call["name"])
        status = "error"
        if tool is None:
            # If the model hallucinates a tool, say so rather than crash the run.
            content, artifact = f"There is no tool called '{call['name']}'.", {"outcome": "crash"}
        else:
            try:
                # Invoked with the whole tool call, a tool answers with a
                # ToolMessage: the text for the model, and the event as artifact.
                result = tool.invoke(call)
                content, status = result.content, "success"
                artifact = {"outcome": "ok", "event": result.artifact}
            except NeedsClarification as exc:
                content, artifact = str(exc), {"outcome": "clarify"}
            except UnexpectedResult as exc:
                content = str(exc)
                artifact = {"outcome": "unexpected", "repair": exc.repair, "event": exc.event}
            except ValidationError as exc:
                content, artifact = f"Invalid arguments: {exc}", {"outcome": "bad_args"}
            except todoist.TodoistError as exc:
                content = str(exc)
                artifact = {"outcome": "api_error", "status_code": exc.status_code}
            except Exception as exc:
                content, artifact = f"{type(exc).__name__}: {exc}", {"outcome": "crash"}

        observations.append(
            ToolMessage(
                content=content,
                tool_call_id=call["id"],
                name=call["name"],
                status=status,
                artifact=artifact,
            )
        )

    return {"messages": observations}


def _latest_observations(messages) -> list[ToolMessage]:
    """The ToolMessages the tools node just appended: the contiguous tail."""
    tail = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        tail.append(message)
    tail.reverse()
    return tail


def _observe_node(state: AgentState) -> dict:
    """Replan: judge each observation, spend the retry budget, set the mode.

    The verdict goes two places. Its note is appended to the tool result (same
    message id, so it replaces the original in state) -- that is what the model
    reads. Its action sets `mode`, which is what the *graph* acts on: "respond"
    for ask/fail turns off the tools for the next agent turn.
    """
    retries = state.get("retries", 0)
    rewritten, actions = [], []

    for observation in _latest_observations(state["messages"]):
        verdict = classify(observation.artifact or {}, retries)
        actions.append(verdict.action)

        if verdict.action == "retry":
            retries += 1
            if verdict.backoff:
                time.sleep(config.RETRY_BACKOFF_SECONDS)

        if verdict.note:
            rewritten.append(
                ToolMessage(
                    content=f"{observation.content}\n\nObserver: {verdict.note}",
                    tool_call_id=observation.tool_call_id,
                    name=observation.name,
                    id=observation.id,  # same id -> replaces the original
                    status=observation.status,
                    artifact=observation.artifact,
                )
            )

    if any(action in ("ask", "fail") for action in actions):
        mode = "respond"
    elif "retry" in actions:
        mode = "retry"
    else:
        mode = "act"

    return {"messages": rewritten, "retries": retries, "mode": mode}


def _route(state: AgentState) -> Literal["tools", "__end__"]:
    """The conditional edge after `agent`: act again, or stop and answer?"""
    last = state["messages"][-1]

    if not getattr(last, "tool_calls", None):
        return END  # Claude replied in plain text -- that is the answer.

    if state.get("steps", 0) >= config.MAX_TOOL_LOOPS:
        return END  # Safety valve; run() turns this into an honest "I got stuck".

    return "tools"


# --- Wiring -------------------------------------------------------------------


def _build_llm() -> ChatAnthropic:
    """The raw Claude client; the graph binds tools onto it as needed."""
    headers = None
    if config.ANTHROPIC_WORKSPACE_ID:
        # An org-level personal access token has to say which workspace it is
        # acting in. A key created inside a workspace already knows.
        headers = {"anthropic-workspace-id": config.ANTHROPIC_WORKSPACE_ID}
    return ChatAnthropic(
        model=config.MODEL,
        api_key=config.ANTHROPIC_API_KEY,
        max_tokens=config.MAX_TOKENS,
        default_headers=headers,
    )


def _build_model():
    """Claude with the five tools bound, so it can emit tool calls."""
    return _build_llm().bind_tools(TOOLS)


def check_model() -> None:
    """One call to the free token-count endpoint, so a bad key -- or a personal
    access token that was never told its workspace -- fails at startup instead
    of on the first message. Raises whatever the SDK raises."""
    _build_llm().get_num_tokens_from_messages([HumanMessage("ping")])


def build_graph(model=None):
    """Wire the nodes together and compile.

    `model` is the *raw* chat model (the tests inject a scripted one). The graph
    binds the tools itself because it needs both flavours: with tools for
    acting, without for a turn whose only job is to talk to the user.
    """
    llm = model if model is not None else _build_llm()
    acting = llm.bind_tools(TOOLS)
    responding = llm

    graph = StateGraph(AgentState)
    graph.add_node("agent", lambda state: _agent_node(state, acting, responding))
    graph.add_node("tools", _tool_node)
    graph.add_node("observe", _observe_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route, {"tools": "tools", END: END})
    graph.add_edge("tools", "observe")  # act, then judge
    graph.add_edge("observe", "agent")  # then think again

    return graph.compile()


# --- One message, end to end --------------------------------------------------


def _history(rows: list[tuple[str, str]]) -> list[BaseMessage]:
    """The recent log as messages, with a user turn in front if the window
    happens to open on a check-in Nexus sent by itself."""
    messages: list[BaseMessage] = [
        HumanMessage(text) if role == "user" else AIMessage(text) for role, text in rows
    ]
    if messages and isinstance(messages[0], AIMessage):
        messages.insert(0, HumanMessage(CHECK_IN_MARKER))
    return messages


def _remember(memory: Memory, chat_id: int, new_messages: list[BaseMessage]) -> None:
    """Log what the run did and learn from it: one row per tool call, and the
    event behind each successful (or half-successful) one."""
    args_by_call = {}
    for message in new_messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                args_by_call[call["id"]] = call["args"]
        elif isinstance(message, ToolMessage):
            artifact = message.artifact or {}
            memory.log(
                chat_id,
                "tool",
                message.content,
                kind=message.name,
                detail={
                    "args": args_by_call.get(message.tool_call_id, {}),
                    "outcome": artifact.get("outcome"),
                },
            )
            memory.learn(artifact.get("event"))


def run(graph, user_text: str, memory: Memory, chat_id: int = 0) -> str:
    """Run one Telegram message through the graph and return the reply text."""
    rows = memory.recent(chat_id)
    history = _history(rows)
    note = memory.note([user_text, *(text for role, text in rows if role == "user")])
    memory.log(chat_id, "user", user_text)

    final = graph.invoke(
        {
            "messages": [*history, HumanMessage(user_text)],
            "steps": 0,
            "retries": 0,
            "mode": "act",
            "memory_note": note,
        }
    )
    _remember(memory, chat_id, final["messages"][len(history) + 1 :])

    reply = final["messages"][-1].text.strip()
    if not reply:
        # We left the loop still wanting to call tools, i.e. we hit MAX_TOOL_LOOPS.
        reply = (
            "I got stuck on that one - I kept trying tools without getting to an "
            "answer. Could you rephrase it?"
        )

    memory.log(chat_id, "assistant", reply, kind="reply")
    return reply

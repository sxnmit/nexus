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

`run()` also keeps the record: one `Run` per message with the outcome, what
the model calls cost, how long it all took, and a trace of what the model saw
and did (`_trace`). The nodes contribute two facts the messages would not
otherwise carry -- how long each model and tool call took, and the observer's
verdict on each tool call -- in the messages' `response_metadata`, which the
model never sees either. Every run is tagged with `PROMPT_VERSION` and the
build, so a change to the prompt or the code can be seen in what followed.
"""

import hashlib
import json
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
from memory import Memory, Run, run_id
from tools import TOOLS, NeedsClarification, UnexpectedResult

_TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}

# What run() says when the loop ended without a reply.
STUCK_REPLY = (
    "I got stuck on that one - I kept trying tools without getting to an answer. "
    "Could you rephrase it?"
)

# Claude requires the first message to be the user's. A conversation window can
# begin with a check-in Nexus sent on its own; this stands in for the clock.
CHECK_IN_MARKER = "[scheduled check-in]"

SYSTEM_PROMPT = """You are Nexus, the user's personal task assistant. You manage \
their Todoist.

You are talking over Telegram, so keep replies to one or two short sentences of \
plain text. No markdown headings and no bullet points, except when you are \
listing tasks.

Right now it is {now}.

You have six tools: create_task, list_tasks, complete_task, update_task, \
delete_task, set_reminder.

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
- "Remind me to X tomorrow at 3pm" is a task: create_task with that due date. \
set_reminder is for an extra notification on a task that already exists -- \
"remind me 30 minutes before the PR review" (before="30 minutes") or "ping me \
about the PR review at 9am" (at="9am"). Todoist sends the notification, not you. \
A "before" reminder needs a task with a due time.
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


def _prompt_version() -> str:
    """A short hash of everything besides the model that shapes its behaviour:
    the prompt template, the mode notes, and the tools' names, descriptions
    and argument schemas. Recorded on every run, so "did Tuesday's prompt
    change help?" is a question the record can answer."""
    material = json.dumps(
        [SYSTEM_PROMPT, MODE_NOTES, [[tool.name, tool.description, tool.args] for tool in TOOLS]],
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:12]


PROMPT_VERSION = _prompt_version()


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
    started = time.perf_counter()
    reply = model.invoke([prompt] + state["messages"])
    # How long the call took, kept with the call. A copy, so the model's own
    # object (a scripted reply, in tests) is left alone.
    reply = reply.model_copy(
        update={"response_metadata": {**reply.response_metadata, "latency_ms": _ms(started)}}
    )
    return {"messages": [reply], "steps": state.get("steps", 0) + 1, "mode": "act"}


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


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
        started = time.perf_counter()
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
                response_metadata={"latency_ms": _ms(started)},
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

    The verdict goes three places. Its note is appended to the tool result (same
    message id, so it replaces the original in state) -- that is what the model
    reads. Its action sets `mode`, which is what the *graph* acts on: "respond"
    for ask/fail turns off the tools for the next agent turn. And the action is
    kept on the message, for the record; a "done" needs no rewrite, so the
    trace reads a clean outcome as done.
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
                    response_metadata={**observation.response_metadata, "verdict": verdict.action},
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


def build_llm(model: str | None = None, max_tokens: int | None = None) -> ChatAnthropic:
    """A raw Claude client: the agent's model by default, or another for
    another job (the judge grades with one). The graph binds tools onto it as
    needed."""
    headers = None
    if config.ANTHROPIC_WORKSPACE_ID:
        # An org-level personal access token has to say which workspace it is
        # acting in. A key created inside a workspace already knows.
        headers = {"anthropic-workspace-id": config.ANTHROPIC_WORKSPACE_ID}
    return ChatAnthropic(
        model=model or config.MODEL,
        api_key=config.ANTHROPIC_API_KEY,
        max_tokens=max_tokens or config.MAX_TOKENS,
        default_headers=headers,
    )


def _build_model():
    """Claude with the six tools bound, so it can emit tool calls."""
    return build_llm().bind_tools(TOOLS)


def check_model() -> None:
    """One call to the free token-count endpoint, so a bad key -- or a personal
    access token that was never told its workspace -- fails at startup instead
    of on the first message. Raises whatever the SDK raises."""
    build_llm().get_num_tokens_from_messages([HumanMessage("ping")])


def build_graph(model=None):
    """Wire the nodes together and compile.

    `model` is the *raw* chat model (the tests inject a scripted one). The graph
    binds the tools itself because it needs both flavours: with tools for
    acting, without for a turn whose only job is to talk to the user.
    """
    llm = model if model is not None else build_llm()
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


def _remember(memory: Memory, chat_id: int, new_messages: list[BaseMessage], run: str) -> None:
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
                run_id=run,
            )
            memory.learn(artifact.get("event"))


# --- The record ---------------------------------------------------------------


def _verdict(observation: ToolMessage) -> str:
    """What the observer decided about a tool call. A clean outcome is never
    rewritten, so it carries no verdict and simply means done."""
    outcome = (observation.artifact or {}).get("outcome")
    return observation.response_metadata.get("verdict", "done" if outcome == "ok" else "fail")


def _usage(message: AIMessage) -> dict:
    """Token counts as the API reported them, or zeros for a model that gave none."""
    usage = message.usage_metadata or {}
    details = usage.get("input_token_details") or {}
    return {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "cache_read": details.get("cache_read", 0),
    }


def _trace(
    rows: list[tuple[str, str]], note: str, user_text: str, new: list[BaseMessage], reply: str
) -> dict:
    """What an evaluator needs to judge a run without re-running it: what the
    model saw (the replayed conversation, the habits note, the message), every
    step it took -- each model call with its text, tool calls, tokens and time;
    each tool result with its outcome and the observer's verdict -- and what
    went back."""
    steps = []
    for message in new:
        if isinstance(message, AIMessage):
            steps.append(
                {
                    "role": "assistant",
                    "text": message.text,
                    "tool_calls": [
                        {"id": call["id"], "name": call["name"], "args": call["args"]}
                        for call in message.tool_calls
                    ],
                    "usage": _usage(message),
                    "latency_ms": message.response_metadata.get("latency_ms"),
                    "stop_reason": message.response_metadata.get("stop_reason"),
                }
            )
        elif isinstance(message, ToolMessage):
            artifact = message.artifact or {}
            steps.append(
                {
                    "role": "tool",
                    "name": message.name,
                    "call_id": message.tool_call_id,
                    "text": message.content,
                    "outcome": artifact.get("outcome"),
                    "status_code": artifact.get("status_code"),
                    "verdict": _verdict(message),
                    "latency_ms": message.response_metadata.get("latency_ms"),
                }
            )
    return {
        "history": [{"role": role, "text": text} for role, text in rows],
        "memory_note": note,
        "user": user_text,
        "steps": steps,
        "reply": reply,
    }


def _outcome(new: list[BaseMessage], stuck: bool) -> str:
    """How a message run ended, from the observer's verdicts (see memory.Run)."""
    if stuck:
        return "stuck"
    verdicts = [_verdict(message) for message in new if isinstance(message, ToolMessage)]
    if "fail" in verdicts:
        return "failed"
    if "ask" in verdicts:
        return "asked"
    return "ok"


def run(graph, user_text: str, memory: Memory, chat_id: int = 0) -> str:
    """Run one Telegram message through the graph and return the reply text.

    Around the graph: replay the recent conversation and look up the habits
    first; afterwards log the exchange, learn from the tool events, and record
    the run. A crash inside the graph is recorded too, then re-raised, so the
    caller still answers honestly and the record still shows it."""
    this = run_id()
    started, clock = memory.now(), time.perf_counter()
    rows = memory.recent(chat_id)
    history = _history(rows)
    note = memory.note([user_text, *(text for role, text in rows if role == "user")])
    memory.log(chat_id, "user", user_text, run_id=this)

    def record(reply: str, outcome: str, new: list[BaseMessage], final: dict, error: str = ""):
        calls = [message for message in new if isinstance(message, AIMessage)]
        usage = [_usage(message) for message in calls]
        served = next((c.response_metadata.get("model") for c in calls), None)
        memory.record(
            Run(
                id=this,
                ts=started,
                chat_id=chat_id,
                kind="message",
                trigger=user_text,
                reply=reply,
                outcome=outcome,
                latency_ms=_ms(clock),
                steps=final.get("steps", 0),
                retries=final.get("retries", 0),
                calls=len(calls),
                tool_calls=sum(isinstance(message, ToolMessage) for message in new),
                input_tokens=sum(u["input"] for u in usage),
                output_tokens=sum(u["output"] for u in usage),
                cache_read_tokens=sum(u["cache_read"] for u in usage),
                model=served or config.MODEL,
                prompt=PROMPT_VERSION,
                build=config.COMMIT,
                error=error,
                trace=_trace(rows, note, user_text, new, reply),
            )
        )

    try:
        final = graph.invoke(
            {
                "messages": [*history, HumanMessage(user_text)],
                "steps": 0,
                "retries": 0,
                "mode": "act",
                "memory_note": note,
            }
        )
    except Exception as exc:
        record("", "crashed", [], {}, error=f"{type(exc).__name__}: {exc}")
        raise

    new = final["messages"][len(history) + 1 :]
    _remember(memory, chat_id, new, this)

    reply = final["messages"][-1].text.strip()
    stuck = not reply  # we left the loop still wanting tools: MAX_TOOL_LOOPS
    if stuck:
        reply = STUCK_REPLY

    memory.log(chat_id, "assistant", reply, kind="reply", run_id=this)
    record(reply, _outcome(new, stuck), new, final)
    return reply

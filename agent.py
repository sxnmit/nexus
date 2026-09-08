"""The LangGraph agent loop.

    START -> agent -> [has tool calls?] -> tools -> agent -> ... -> END
                            |
                            +-- no tool calls --> END

Two nodes and one conditional edge is the whole thing:

`agent`  (plan)   Sends the conversation so far to Claude with the three tools
                  bound. Claude replies with either a final piece of text or one
                  or more tool calls. This node never touches Todoist -- it only
                  decides.

`tools`  (act +   Runs every tool Claude asked for and appends one ToolMessage
          observe) per call. Crucially it does *not* raise: a tool that blows up
                  comes back as a ToolMessage with status="error" whose content
                  is the error text. That is the "observe" half of the loop.

`_route` (the     Looks at the last message. Tool calls -> go to `tools`. Plain
          loop)   text -> we have an answer, go to END. It also stops the loop
                  after MAX_TOOL_LOOPS trips so a confused model cannot spin.

The edge `tools -> agent` is what makes this an agent rather than a single
function call: after every observation Claude gets to look again and decide what
to do next. That is the only mechanism behind "retry with corrected input or ask
a clarifying question" -- an error observation is just another thing to react to,
so a failed tool call and a successful one flow through exactly the same path.

There is no memory here on purpose. Every Telegram message starts a fresh
conversation, so `state["messages"]` only ever holds one exchange plus whatever
tool traffic it needed.
"""

from datetime import datetime
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

import config
from tools import TOOLS

_TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}

SYSTEM_PROMPT = """You are Nexus, {user}'s personal task assistant. You manage \
their Todoist.

You are talking over Telegram, so keep replies to one or two short sentences of \
plain text. No markdown headings and no bullet points, except when you are \
listing tasks.

Right now it is {now}.

Rules:
- Use a tool whenever the user asks about or wants to change their tasks. Never \
invent a task name, id, or due date -- take them from what the tools return.
- Pass due dates to create_task in the user's own words ("tomorrow at 3pm", \
"next friday"). Todoist parses them; you should not convert them to a date.
- If a tool comes back with an error, deal with it. Either fix the input and \
call the tool again, or ask the user a short clarifying question. Do not try \
the same failing call more than twice, and never tell the user something \
worked when the tool said it did not.
- After a tool succeeds, confirm what actually happened using the values the \
tool returned, e.g. "Added 'Submit iTrade PR review', due Tue 9 Sep 3:00pm"."""


class AgentState(MessagesState):
    """MessagesState already gives us `messages` with append-on-update semantics.

    We add `steps`: how many times we have been through the agent node for this
    message. `_route` uses it as the loop's safety valve.
    """

    steps: int


def _system_prompt() -> str:
    # Rebuilt per call so the model always knows today's date -- it needs that to
    # sanity-check and echo back due dates like "tomorrow at 3pm".
    now = datetime.now().astimezone()
    return SYSTEM_PROMPT.format(user="the user", now=now.strftime("%A %d %B %Y, %H:%M %Z"))


def _agent_node(state: AgentState, model) -> dict:
    """Plan: hand the conversation to Claude and let it decide what to do."""
    # The system prompt is prepended rather than stored in state, so it is not
    # duplicated every time we come back round the loop.
    reply = model.invoke([SystemMessage(_system_prompt())] + state["messages"])
    return {"messages": [reply], "steps": state.get("steps", 0) + 1}


def _tool_node(state: AgentState) -> dict:
    """Act and observe: run the requested tools, record what happened.

    Nothing in here raises. A tool that fails produces a ToolMessage with
    status="error" containing the error text, which the next agent step reads
    like any other observation. That is what lets the model correct itself
    instead of failing silently or inventing a success.
    """
    observations = []

    for call in state["messages"][-1].tool_calls:
        tool = _TOOLS_BY_NAME.get(call["name"])
        if tool is None:
            # Shouldn't happen -- but if the model hallucinates a tool, tell it so
            # rather than crashing the whole run.
            content, status = f"There is no tool called '{call['name']}'.", "error"
        else:
            try:
                content, status = str(tool.invoke(call["args"])), "success"
            except Exception as exc:
                content, status = f"{type(exc).__name__}: {exc}", "error"

        observations.append(
            ToolMessage(
                content=content,
                tool_call_id=call["id"],
                name=call["name"],
                status=status,
            )
        )

    return {"messages": observations}


def _route(state: AgentState) -> Literal["tools", "__end__"]:
    """The conditional edge: loop again, or stop and answer?"""
    last = state["messages"][-1]

    if not getattr(last, "tool_calls", None):
        return END  # Claude replied in plain text -- that is the answer.

    if state.get("steps", 0) >= config.MAX_TOOL_LOOPS:
        return END  # Safety valve; run() turns this into an honest "I got stuck".

    return "tools"


def _build_llm() -> ChatAnthropic:
    """The raw Claude client; `_build_model` is this with the tools bound."""
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
    """Claude, with the three tools bound so it can emit tool calls."""
    return _build_llm().bind_tools(TOOLS)


def check_model() -> None:
    """One call to the free token-count endpoint, so a bad key -- or a personal
    access token that was never told its workspace -- fails at startup instead
    of on the first message. Raises whatever the SDK raises."""
    _build_llm().get_num_tokens_from_messages([HumanMessage("ping")])


def build_graph(model=None):
    """Wire the nodes together and compile.

    `model` exists so the tests can inject a scripted fake instead of calling
    the real API.
    """
    model = model if model is not None else _build_model()

    graph = StateGraph(AgentState)
    graph.add_node("agent", lambda state: _agent_node(state, model))
    graph.add_node("tools", _tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # observe, then think again

    return graph.compile()


def run(graph, user_text: str) -> str:
    """Run one Telegram message through the graph and return the reply text."""
    final = graph.invoke({"messages": [HumanMessage(user_text)], "steps": 0})
    reply = final["messages"][-1].text.strip()

    if not reply:
        # We left the loop still wanting to call tools, i.e. we hit MAX_TOOL_LOOPS.
        return (
            "I got stuck on that one - I kept trying tools without getting to an "
            "answer. Could you rephrase it?"
        )

    return reply

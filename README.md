# Nexus

A personal task assistant you message on Telegram. It runs a small LangGraph
agent on Claude that manages your Todoist through three tools -- create a task,
list open tasks, complete a task -- and replies in plain language.

```
You:    remind me to submit the iTrade PR review tomorrow at 3pm
Nexus:  Added 'Submit iTrade PR review', due Wed 9 Sep 3:00pm.

You:    what's on my list?
Nexus:  Two things: Submit iTrade PR review (due tomorrow 3pm) and Buy milk.

You:    mark the review done
Nexus:  Done - completed 'Submit iTrade PR review'.
```

This is the Week 1 cut: a real plan -> act -> observe loop, no memory between
messages, no scheduling, no multi-step planning. See [What's deliberately not
here](#whats-deliberately-not-here-yet) for how those would slot in.

## Quick start

Needs **Python 3.11 or newer** (CI runs 3.11-3.13). Check `python3 --version`
first: macOS ships 3.9 at `/usr/bin/python3`, and on that `pip` fails with
*Could not find a version that satisfies the requirement anthropic==...*
because the whole stack requires 3.10+. `brew install python@3.12` (or
`uv venv --python 3.12`) fixes it.

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # any 3.11+ interpreter
pip install -r requirements.txt

cp .env.example .env      # then fill in the three tokens
python bot.py
```

`bot.py` checks all three tokens and makes one test call to Todoist before it
starts polling, so a bad token fails at startup rather than mid-conversation.
Then open Telegram, find your bot, and send it a message.

The `.env` names are exactly:

| Variable                       | What it is                                               |
| ------------------------------ | -------------------------------------------------------- |
| `PERSONAL_ACCESS_TOKEN_CLAUDE` | Anthropic API key                                        |
| `API_TOKEN_TODOIST`            | Todoist -> Settings -> Integrations -> Developer         |
| `API_TOKEN_TELEGRAM`           | From @BotFather                                          |
| `TELEGRAM_ALLOWED_USER_IDS`    | *Optional but recommended* -- see [Lock it down](#lock-it-down) |

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest                     # 96 tests, ~1s, no network, no tokens needed
ruff check . && ruff format --check .
```

The suite never touches Claude, Todoist or Telegram: the model is a scripted
stand-in and Todoist is an in-memory fake (or, for the HTTP client itself, a
recorded `httpx` layer that returns real `Response` objects). Coverage is
enforced at 95% and a deprecated LangChain call fails the run.

[CI](.github/workflows/ci.yml) runs lint, formatting and the tests on every
push and pull request, across Python 3.11-3.13.

## How the agent loop works

Everything interesting is in [`agent.py`](agent.py); it's about 60 lines
without comments. The graph is:

```
                 ┌─────────────────────────────────────┐
                 │                                     │
  START ──► agent ──► _route ──► tools ──────────────┘
              ▲                    │
              │     no tool calls, │
              │     or loop cap    ▼
              └────────────────── END
```

Two nodes, one conditional edge, and the edge from `tools` back to `agent`.
That back-edge is the whole point -- it's what makes this an agent rather than
a single function call.

### State

The graph carries a `MessagesState`: a list of messages that only ever grows
(LangGraph's `add_messages` reducer appends rather than replaces). Plus one
extra field, `steps`, counting trips through the agent node.

Every Telegram message starts with a fresh state containing only that one
`HumanMessage`. By the time the run finishes the state holds the full
exchange -- the user's text, Claude's tool calls, the tool results, Claude's
final answer -- and then it's thrown away. That's what "no memory" means here.

### `agent` -- the plan step

```python
def _agent_node(state, model):
    reply = model.invoke([SystemMessage(_system_prompt())] + state["messages"])
    return {"messages": [reply], "steps": state.get("steps", 0) + 1}
```

Send the conversation so far to Claude (with the three tools bound to it) and
append whatever it says. Claude replies with one of two things:

- An `AIMessage` with **text** -- it has an answer for the user.
- An `AIMessage` with **`tool_calls`** -- it wants to do something first.
  Each tool call has a name, JSON arguments, and an id.

This node never touches Todoist. It only decides.

The system prompt is prepended on every call rather than stored in state, so
it isn't duplicated each time round the loop, and so it always carries the
current date -- which Claude needs to sanity-check and echo back things like
"tomorrow at 3pm".

### `_route` -- the loop

```python
def _route(state):
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END          # plain text: that's the answer
    if state.get("steps", 0) >= config.MAX_TOOL_LOOPS:
        return END          # safety valve
    return "tools"
```

A conditional edge is just a function that looks at the state and names the
next node. If Claude answered in text, we're done. If it asked for tools, go
run them. The step cap (6 by default) stops a confused model from looping on
your token bill; `run()` turns that case into an honest "I got stuck" rather
than an empty reply.

### `tools` -- the act + observe step

```python
def _tool_node(state):
    observations = []
    for call in state["messages"][-1].tool_calls:
        try:
            content, status = str(tool.invoke(call["args"])), "success"
        except Exception as exc:
            content, status = f"{type(exc).__name__}: {exc}", "error"
        observations.append(ToolMessage(content=content, tool_call_id=call["id"], status=status))
    return {"messages": observations}
```

Run each tool Claude asked for and append one `ToolMessage` per call, matched
back by `tool_call_id`. Then the edge sends us back to `agent`, where Claude
reads what happened and decides again.

The important design decision is in the `except`: **nothing raises out of this
node**. A failed tool call becomes a `ToolMessage` with `status="error"` whose
content is the error text. From the loop's point of view a failure and a
success are the same kind of thing -- an observation -- and they flow through
the same path back to Claude. That is the entire mechanism behind "retry with
corrected input or ask a clarifying question": there is no separate
error-handling branch, just a model that gets told the truth and asked again.

### What that looks like in practice

The tools are written to make good observations. They raise on failure (never
return a cheerful string for something that didn't happen), and when something
is ambiguous they raise *with the options included*. So with two tasks
containing "review":

```
You      mark the review done

agent    Claude -> complete_task(task="review")
tools    LookupError: 'review' matches 2 open tasks:
             [1] Submit iTrade PR review
             [2] Review the design doc
           Ask the user which one they meant. Do not guess.
           (appended as ToolMessage, status="error")
agent    Claude reads that, replies in text
_route   no tool calls -> END

Nexus    Which one - the iTrade PR review or the design doc review?
```

Or, when it can fix the problem itself:

```
You      mark the PR review done          (only one task matches "pr review")

agent    complete_task(task="pr review")
tools    Completed [1] Submit iTrade PR review        (status="success")
agent    "Done - completed 'Submit iTrade PR review'."
```

And the case where Todoist quietly accepts a task but can't parse the date:
`create_task` notices the missing `due` field in the response and returns
"Created ... but Todoist could not understand the due date 'sometime soonish',
so the task has no due date" -- so Claude tells you that instead of confirming
a due date that doesn't exist.

The tests in [`tests/test_agent.py`](tests/test_agent.py) walk through each of
these paths with a scripted model, including the retry (round one ambiguous,
round two uses the id from the error, round three confirms).

## Project layout

Flat on purpose -- it's a learning project, and six modules don't need a
package.

| File                | What it does                                                        |
| ------------------- | ------------------------------------------------------------------- |
| `bot.py`            | Telegram polling loop. Hands each message to the agent, replies.    |
| `agent.py`          | The LangGraph graph: state, the two nodes, routing, `run()`.        |
| `tools.py`          | `create_task`, `list_tasks`, `complete_task` as LangChain tools.    |
| `todoist.py`        | Thin HTTP client for the three Todoist calls. Raises `TodoistError`.|
| `config.py`         | Reads `.env`. Nothing raises on import; `bot.py` validates at start.|
| `tests/`            | The test suite. `support.py` holds the fakes, `conftest.py` the fixtures. |

The stack: [LangGraph](https://langchain-ai.github.io/langgraph/) for the loop,
`langchain-anthropic` (which wraps the official `anthropic` SDK) for the model,
[python-telegram-bot](https://docs.python-telegram-bot.org/) in polling mode,
and plain `httpx` against Todoist.

## Things to know

### Todoist API version

This uses Todoist's current unified API at `https://api.todoist.com/api/v1`
rather than the older REST v2 -- the [developer docs](https://developer.todoist.com/)
now show v1 as the way in. The three endpoints have the same paths on both, and
`todoist.get_tasks()` understands both response shapes (v1 paginates with a
cursor, which is followed; v2 returned a bare list), so if you ever need to
fall back, `TODOIST_API_BASE=https://api.todoist.com/rest/v2` is the only
change.

The sandbox this was built in couldn't reach `api.todoist.com`, so the client
is verified against recorded responses rather than the live API. The startup
check in `bot.py` is your first real call -- if it fails, the error message
tells you what to look at.

### Lock it down

A Telegram bot is public: anyone who finds its username can message it, and
this one edits your Todoist. Set `TELEGRAM_ALLOWED_USER_IDS` to your own user
id (message `@userinfobot` to find it) and the bot silently ignores everyone
else. It warns at startup if this isn't set.

### Model

`claude-haiku-4-5` by default (`NEXUS_MODEL` to override). Deciding which of
three tools to call is a routing job, not a reasoning one, and Haiku is fast
and cheap at it.

`max_tokens` is 1024 (`NEXUS_MAX_TOKENS` to override). That is a ceiling on
one response, not a spend, and it cannot stop the model hallucinating -- but
it bounds the damage. Telegram rejects messages over 4096 characters, about
1000 English tokens, so anything longer could not be delivered anyway, and
worst-case spend per message becomes `MAX_TOOL_LOOPS x max_tokens`. Hitting
the ceiling shows up as a reply cut off mid-sentence; a very long task list
is the only realistic way to get there.

The code sets neither `thinking` nor `temperature`, so it runs unchanged on
`claude-sonnet-5` or `claude-opus-5` if you want to compare. Those run
adaptive thinking by default (and reject `temperature` alongside it), and
thinking tokens count against `max_tokens`, so raise `NEXUS_MAX_TOKENS` to
~8192 when you switch. `run()` already handles the block-style content a
thinking model returns.

## What's deliberately not here (yet)

Each of these would be a small, contained change. They're left out because
the point of Week 1 is understanding the loop, not accreting features.

- **Memory across messages.** LangGraph does this with a *checkpointer*:
  `graph.compile(checkpointer=MemorySaver())` and pass
  `config={"configurable": {"thread_id": chat_id}}` to `invoke`. The state
  then persists per chat and "the second one" would mean something. The cost
  is that context grows every message and you need a policy for trimming it.
- **Proactive / scheduled messages** ("remind me at 3pm" that pings *you*).
  Needs a job runner alongside the polling loop and a way for the agent to
  register jobs -- a fourth tool plus something like APScheduler.
- **Multi-step planning.** The loop already handles "list, then complete the
  one that matches" because Claude chains tool calls itself. A planner node
  that writes down a plan first only pays off once tasks get long enough that
  the model loses track.

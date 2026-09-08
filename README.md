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

Everything interesting is in [`agent.py`](agent.py). The graph is:

```
                 ┌────────────────────────────────────────────────┐
                 │                                                │
  START ──► agent ──► _route ──► tools ──► observe ───────────────┘
              ▲
              │      plain text, or loop cap
              └─────────────────────────────────────► END
```

Three nodes, one conditional edge, and the back-edge from `observe` to
`agent`. That back-edge is what makes this an agent rather than a single
function call -- and `observe` is what makes it an agent with a *policy*
rather than one that hopes the model reacts well to errors.

### State

The graph carries a `MessagesState` -- a list of messages that only ever grows
(LangGraph's `add_messages` reducer appends rather than replaces, except that a
message with the same `id` replaces its predecessor, which `observe` relies
on). Plus three fields:

| Field     | Meaning                                                          |
| --------- | ---------------------------------------------------------------- |
| `steps`   | Trips through the agent node this run. The loop cap.             |
| `retries` | Retries the observe step has granted this run. The retry budget. |
| `mode`    | What the *next* agent turn is for: `act`, `retry`, or `respond`. |

Every Telegram message starts with a fresh state containing only that one
`HumanMessage`. By the time the run finishes the state holds the full
exchange and then it's thrown away. That's what "no memory" means here -- and
see the caveat about clarifying questions at the end of this section.

### `agent` -- the plan step

```python
def _agent_node(state, acting, responding):
    mode = state.get("mode", "act")
    model = responding if mode == "respond" else acting
    reply = model.invoke([SystemMessage(_system_prompt(mode))] + state["messages"])
    return {"messages": [reply], "steps": state.get("steps", 0) + 1, "mode": "act"}
```

Send the conversation so far to Claude and append whatever it says: an
`AIMessage` with **text** (it has an answer for the user) or with
**`tool_calls`** (it wants to do something first). This node never touches
Todoist. It only decides.

The Stage 2 addition is *which* Claude it sends to. `acting` has the five
tools bound. `responding` is the same model with **no tools at all**, used when
`observe` has decided the right move is to talk to the user. That turns "ask a
clarifying question instead of guessing" from a prompt instruction into a
guarantee: on a respond turn the model cannot emit a tool call, so it cannot
guess. The system prompt also gets a short mode note -- "follow the repair note
exactly" on a retry turn; "you cannot use tools this turn, ask one question or
report the failure" on a respond turn.

The system prompt is prepended on every call rather than stored in state, so
it isn't duplicated each time round the loop and always carries the current
date.

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

A conditional edge is a function that looks at the state and names the next
node. Text means done; tool calls mean go run them. The step cap (6 by default)
stops a confused model from looping on your token bill; `run()` turns that case
into an honest "I got stuck" rather than an empty reply.

### `tools` -- the act step

```python
try:
    content, status = str(tool.invoke(call["args"])), "success"
    artifact = {"outcome": "ok"}
except NeedsClarification as exc:
    content, artifact = str(exc), {"outcome": "clarify"}
except UnexpectedResult as exc:
    content, artifact = str(exc), {"outcome": "unexpected", "repair": exc.repair}
except todoist.TodoistError as exc:
    content, artifact = str(exc), {"outcome": "api_error", "status_code": exc.status_code}
except Exception as exc:
    content, artifact = f"{type(exc).__name__}: {exc}", {"outcome": "crash"}
observations.append(ToolMessage(content=content, tool_call_id=call["id"],
                                status=status, artifact=artifact))
```

Run each tool Claude asked for and append one `ToolMessage` per call, matched
back by `tool_call_id`. Two things are deliberate. **Nothing raises out of this
node** -- a failure is an observation like any other. And **this node decides
nothing.** It records what happened on two channels: `content`, the readable
result the model sees, and `artifact`, a structured record of the *kind* of
outcome that the model never sees (LangChain keeps `artifact` out of the API
request). The kind comes from which exception the tool raised -- which is why
the tools raise typed failures, not generic ones.

### `observe` -- the replan step

This is the new node, and it answers the question that matters: after a tool
call, how does the loop decide between *retry*, *ask me*, and *done*?

The decision is a pure function, `classify(artifact, retries_used)`, so it
reads as a table:

| The tool reported                                | Meaning                                                | Budget left                          | Budget spent |
| ------------------------------------------------ | ------------------------------------------------------ | ------------------------------------ | ------------ |
| `ok`                                             | It worked                                              | **done**                             | done         |
| `clarify` -- ambiguous, not found, missing info  | Only the user can resolve it                           | **ask**                              | ask          |
| `unexpected` -- 200, but the result is wrong     | A corrected call can fix it; the world already changed | **retry** via the tool's repair note | ask          |
| `bad_args` -- the model passed invalid arguments | The model can fix its own input                        | **retry**                            | ask          |
| `api_error` 400                                  | Todoist rejected something in the input                | **retry**, corrected                 | ask          |
| `api_error` 404                                  | The task is gone                                       | **ask**                              | ask          |
| `api_error` 401 / 403                            | Credentials; nothing to retry                          | **fail**                             | fail         |
| `api_error` 429 / 5xx / no status                | Transient                                              | **retry**, same input, after a pause | fail         |
| `crash` -- unknown tool, a bug                   | Not the user's problem to solve                        | **fail**                             | fail         |

Read top to bottom the questions are: *Did it work? Is this something only the
user can resolve? Could a corrected or repeated call fix it, and is there
budget? Otherwise fail, honestly.* "Ask" and "fail" both end the tool loop for
this message; they differ only in what the model is told to say.

Three decisions in there are worth being able to defend:

**The budget is one retry per message.** A second identical failure is
information, not bad luck, and the user should hear about it. It also means the
loop can never oscillate between two failing calls. (`NEXUS_MAX_RETRIES`.)

**A retry is not always a redo.** The `unexpected` row is the interesting one.
`create_task("Call mum", due_string="sometime soonish")` returns 200 and a task
*with no due date* -- Todoist dropped what it couldn't parse. The task now
exists. Calling `create_task` again with a better date would create it twice.
So the tool raises `UnexpectedResult` with a *repair*: "the task already exists
as id 900 -- call `update_task` with a simpler due string". The verdict carries
that note to the model. This is what "check whether it actually returned what
was expected, not just that the API didn't error" looks like in code: the tool
verifies the response against the request, and the failure knows how to be
fixed.

**The model executes the verdict; it doesn't choose it.** The verdict's note is
appended to the tool result (`Observer: RETRY once. ...`) so the model knows
what to do, and its action sets `mode`, which the *graph* enforces: `respond`
binds no tools. Claude decides *how* -- the corrected call, the wording of the
question, the phrasing of the failure. The policy is code, in one place, with
a table-driven test.

### What that looks like in practice

Ambiguous reference -- an **ask** verdict:

```
You      mark the review done

agent    complete_task(task="review")
tools    'review' matches 2 open tasks:
             [1] Submit iTrade PR review
             [2] Review the design doc
         Ask the user which one they meant. Do not guess.
observe  clarify -> ask.  mode = respond
         (appends "Observer: only the user can resolve this; ask them, do not retry.")
agent    no tools bound this turn -> can only reply in text

Nexus    Which one - the iTrade PR review or the design doc review?
```

Unparseable date -- a **retry** verdict whose repair is an update, not a redo:

```
You      remind me to call mum sometime soonish

agent    create_task(content="Call mum", due_string="sometime soonish")
tools    Created [900] Call mum, but Todoist did not understand the due date
         'sometime soonish', so the task has NO due date.
observe  unexpected -> retry (1 of 1).  mode = retry
         (appends "Observer: RETRY once. The task already exists as id 900 --
          do not call create_task again. Call update_task with task='900' ...")
agent    update_task(task="900", due_string="next monday")     <- not create_task
tools    Updated [900] Call mum (due 2026-09-14)
observe  ok -> done.  mode = act
agent    "Added 'Call mum' for next Monday - Todoist couldn't read 'sometime
          soonish'. Say if you meant a different day."
```

A task that doesn't exist -- **ask**, and never an invented id:

```
You      mark the PR review done          (no such open task)

agent    complete_task(task="PR review")
tools    No open task matches 'PR review'. The open tasks are: ...
         Tell the user you could not find it and ask which one they meant.
observe  clarify -> ask.  mode = respond
Nexus    I can't find a task like "PR review" - your open ones are Buy milk and
         Walk the dog. Which did you mean?
```

Rate limit -- **retry** the same call after a pause, then **fail** honestly:

```
tools    Todoist rejected the request: HTTP 429 - rate limited
observe  api_error 429 -> retry, with backoff.  sleeps RETRY_BACKOFF_SECONDS
agent    the same call again
tools    HTTP 429 again
observe  budget spent -> fail.  mode = respond
Nexus    Todoist isn't responding right now (rate limited). Try again in a minute.
```

Each of these is a test in [`tests/test_replan.py`](tests/test_replan.py) with
a scripted model and an in-memory Todoist -- including the assertion that the
retry after a dropped due date calls `update_task` and leaves exactly one task.

### What the observe step *cannot* check

Pre-tool ambiguity. "Remind me about the thing" never reaches a tool, so there
is no artifact to classify -- whether to ask, or to create a task called "the
thing", is Claude's judgment, steered by the prompt ("never create a task whose
content is a placeholder"). The one deterministic guard is that `create_task`
rejects empty content. Testing the judgment itself needs the real model, which
is what Stage 5's eval harness is for.

### The clarifying-question caveat

There is no memory between messages. When Nexus asks "which one?", your answer
arrives as a *new* run that has never seen the question. Answers have to stand
on their own -- "complete the iTrade one", not "the first one". The fix is
LangGraph's checkpointer (a few lines; see the last section), and it is the
first thing to add when memory arrives in Stage 4.

## Project layout

Flat on purpose -- it's a learning project, and six modules don't need a
package.

| File                | What it does                                                        |
| ------------------- | ------------------------------------------------------------------- |
| `bot.py`            | Telegram polling loop. Hands each message to the agent, replies.    |
| `agent.py`          | The graph: state, the three nodes, the `classify` table, `run()`.   |
| `tools.py`          | The five Todoist tools, raising typed failures that `observe` reads.|
| `todoist.py`        | Thin HTTP client for the three Todoist calls. Raises `TodoistError`.|
| `config.py`         | Reads `.env`. Nothing raises on import; `bot.py` validates at start.|
| `tests/`            | The suite. `support.py` has the fakes, `conftest.py` the fixtures, `test_replan.py` Stage 2. |
| `Dockerfile`        | Runs the bot as a worker. Used by any host that takes a Dockerfile. |
| `Procfile`          | Same thing for buildpack/nixpacks hosts. Declares a `worker`, not a `web`. |

The stack: [LangGraph](https://langchain-ai.github.io/langgraph/) for the loop,
`langchain-anthropic` (which wraps the official `anthropic` SDK) for the model,
[python-telegram-bot](https://docs.python-telegram-bot.org/) in polling mode,
and plain `httpx` against Todoist.

## Things to know

### Which Claude key

`PERSONAL_ACCESS_TOKEN_CLAUDE` can be either kind of key from the Anthropic
Console. A key created *inside a workspace* just works. An org-level
**personal access token** does not know which workspace it is acting in, and
the API refuses it with *"This API key is not scoped to a workspace..."* --
set `ANTHROPIC_WORKSPACE_ID` (Settings -> Workspaces; ids start with
`wrkspc_`) and the bot sends it as the `anthropic-workspace-id` header.
`bot.py` makes one free call to the token-count endpoint at startup, so
either problem fails on boot with that message rather than on your first
Telegram message.

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

## Deploying

Nexus is a **worker**, not a website: one long-lived process that polls Telegram
and opens no HTTP port. Everything below follows from that.

`bot.py` needs no changes to run in production. It reads `.env` if one exists
and falls back to real environment variables, so setting the tokens in your
host's dashboard is enough -- and the `.env` file itself must never be baked
into an image (`.dockerignore` excludes it).

### Two rules that matter more than the host you pick

**Deploy it as a worker / background service, not a web service.** A web
service gets health-checked on a port Nexus never opens, so the platform will
declare it unhealthy and restart it forever.

**Run exactly one instance.** Telegram allows a single `getUpdates` consumer per
bot. A second replica makes both of them fail with `Conflict: terminated by
other getUpdates request`, and your messages get split or dropped. So: no
autoscaling, replicas = 1. This is the one thing that silently breaks a
polling bot in production.

### On any of the usual hosts

[Railway](https://railway.app), [Render](https://render.com), and
[Fly.io](https://fly.io) all work the same way:

1. Point the host at this repo. Railway and Fly will use the `Dockerfile`;
   Render can use it or its native Python runtime.
2. Create the service as a **worker** (Render calls it a Background Worker;
   Railway just runs the `Procfile`'s `worker` process; on Fly, leave the
   `[http_service]` block out of `fly.toml` entirely).
3. Add the environment variables from your `.env` -- at minimum the three
   tokens, plus `ANTHROPIC_WORKSPACE_ID` if your Claude key needs it.
4. Confirm the instance count is 1.
5. Deploy, then read the logs.

A healthy start looks exactly like it does locally:

```
Todoist OK - 50 open task(s). Model: claude-haiku-4-5
Claude OK.
Nexus is polling. Ctrl-C to stop.
```

If a token is wrong the process exits immediately with the reason, so a crash
loop in the logs will tell you which one rather than leaving you guessing.

### Set the allowlist before you deploy

Locally an unlocked bot is a small risk. Deployed, it runs 24/7 against your
real Todoist, and anyone who guesses the bot's username can talk to it. Set
`TELEGRAM_ALLOWED_USER_IDS` -- see [Lock it down](#lock-it-down). The bot logs a
warning at startup while it is unset.

### Running it locally with Docker

```bash
docker build -t nexus .
docker run --rm --env-file .env nexus
```

That is the same image the host runs, so it is worth doing once before you
deploy.

## What's deliberately not here (yet)

Each of these would be a small, contained change. They're left out because
the point of these first stages is understanding the loop, not accreting features.

- **Memory across messages.** LangGraph does this with a *checkpointer*:
  `graph.compile(checkpointer=MemorySaver())` and pass
  `config={"configurable": {"thread_id": chat_id}}` to `invoke`. The state
  then persists per chat, "the second one" would mean something, and a
  clarifying question could actually be answered. The cost is that context
  grows every message and you need a policy for trimming it.
- **Proactive / scheduled messages** ("remind me at 3pm" that pings *you*).
  Needs a job runner alongside the polling loop and a way for the agent to
  register jobs -- a sixth tool plus something like APScheduler.
- **Multi-step planning.** The loop already handles "list, then complete the
  one that matches" because Claude chains tool calls itself. A planner node
  that writes down a plan first only pays off once tasks get long enough that
  the model loses track.

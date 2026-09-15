"""Buttons on the overdue nudge: Done, Tomorrow, Drop. One tap, no model.

A nudge asks "Done, or push it?", and until now the answer was a message: a
model call to work out which task, a tool call to act, a reply. The buttons
skip the model. A tap calls the same tool function the agent would have --
complete_task, update_task with "tomorrow", delete_task -- so the same checks
apply: the task must still be open, the change must actually take, and any
failure is a typed one that can be shown honestly.

Everything else about a tap is treated like a message that went through the
loop: it is logged to the conversation (so "what's left?" a minute later
knows), memory learns from the tool's event, and the record gets a `button`
row. The nudge itself is edited in place with the outcome, and the tapped
task's row of buttons disappears while any others stay.
"""

from dataclasses import dataclass

import todoist
import tools
from memory import Memory, Run, run_id

ACTIONS = ("done", "tomorrow", "drop")
LABELS = {"done": "Done", "tomorrow": "Tomorrow", "drop": "Drop"}

# Telegram allows 64 bytes of callback data; "task:<id>:tomorrow" fits with
# room for the longest Todoist id.
PREFIX = "task"


def data(task_id: str, action: str) -> str:
    return f"{PREFIX}:{task_id}:{action}"


def rows(tasks: list[dict]) -> tuple[tuple[tuple[str, str], ...], ...]:
    """One row of buttons per task. With several tasks the labels carry the
    task's number in the list above them, so "2 Done" reads unambiguously."""
    numbered = len(tasks) > 1
    return tuple(
        tuple(
            (f"{index} {LABELS[action]}" if numbered else LABELS[action], data(task["id"], action))
            for action in ACTIONS
        )
        for index, task in enumerate(tasks, start=1)
    )


def _clock(when) -> str:
    hour = when.hour % 12 or 12
    return f"{hour}:{when.minute:02d}{'am' if when.hour < 12 else 'pm'}"


@dataclass(frozen=True)
class Result:
    """What a tap did: the line appended to the nudge, and whether it worked."""

    line: str
    ok: bool


def apply(memory: Memory, chat_id: int, task_id: str, action: str) -> Result:
    """Do what the button says, through the agent's own tools, and keep the
    record exactly as a message run would."""
    if action not in ACTIONS:
        raise ValueError(f"unknown button action {action!r}")
    this, started = run_id(), memory.now()
    tool_name = {"done": "complete_task", "tomorrow": "update_task", "drop": "delete_task"}[action]
    args: dict = {"task": task_id}
    try:
        found = tools._find_task(task_id, LABELS[action].lower())
        name = found.get("content") or task_id
        if action == "done":
            text, event = tools.complete_task.func(task=task_id)
            line = f"Done: '{name}'"
        elif action == "drop":
            text, event = tools.delete_task.func(task=task_id)
            line = f"Deleted '{name}'"
        else:
            _, when = todoist.parse_due(found)
            args["due_string"] = f"tomorrow at {_clock(when)}" if when is not None else "tomorrow"
            text, event = tools.update_task.func(task=task_id, due_string=args["due_string"])
            line = f"Moved '{name}' to {args['due_string']}"
        outcome, ok = "ok", True
    except tools.NeedsClarification as exc:
        text, event, ok, outcome = str(exc), None, False, "clarify"
        line = "That task is no longer open, so there was nothing to do."
    except tools.UnexpectedResult as exc:
        text, event, ok, outcome = str(exc), exc.event, False, "unexpected"
        line = f"Todoist did not do that as asked: {exc}"
    except todoist.TodoistError as exc:
        text, event, ok, outcome = str(exc), None, False, "api_error"
        line = f"Todoist would not do that right now ({exc}). The buttons still work."
    memory.log(chat_id, "user", f"[tapped {LABELS[action]} on task {task_id}]", run_id=this)
    memory.log(
        chat_id,
        "tool",
        text,
        kind=tool_name,
        detail={"args": args, "outcome": outcome},
        run_id=this,
    )
    memory.learn(event)
    memory.log(chat_id, "assistant", line, kind="button", run_id=this)
    memory.record(
        Run(
            id=this,
            ts=started,
            chat_id=chat_id,
            kind="button",
            trigger=f"{action}:{task_id}",
            reply=line,
            outcome=outcome if ok else "failed",
            latency_ms=int((memory.now() - started).total_seconds() * 1000),
            tool_calls=1,
            trace={"tool": tool_name, "outcome": outcome, "text": text},
        )
    )
    return Result(line, ok)

"""The three tools the agent can call.

Each one returns a short string that goes straight back into the conversation as
an observation, so the wording is aimed at the model, not at a log file.

Two rules make the error handling in agent.py work:
  * Raise on failure -- never return a cheerful string for something that did not
    happen. The tool node catches the exception and marks the observation as an
    error so the model has to deal with it.
  * When something is ambiguous, raise with the *options* included. That is what
    lets the model come back and ask a sensible clarifying question instead of
    guessing which task you meant.
"""

from langchain_core.tools import tool

import todoist


def _format_task(task: dict) -> str:
    """One line per task: id, name, and due date if it has one."""
    due = task.get("due") or {}
    when = due.get("datetime") or due.get("date")
    suffix = f" (due {when})" if when else ""
    return f"[{task.get('id')}] {task.get('content')}{suffix}"


@tool(parse_docstring=True)
def create_task(content: str, due_string: str = "") -> str:
    """Create a new task in the user's Todoist.

    Args:
        content: What the task is, e.g. "Submit iTrade PR review".
        due_string: Optional due date in plain English, phrased the way the user
            phrased it, e.g. "tomorrow at 3pm" or "next friday". Todoist parses
            this itself, so do not convert it to a date. Leave empty for a task
            with no due date.
    """
    task = todoist.create_task(content, due_string or None)

    if due_string and not task.get("due"):
        # Todoist accepted the task but silently dropped a due date it could not
        # parse. Say so rather than letting the model confirm a due date that
        # does not exist.
        return (
            f"Created {_format_task(task)}, but Todoist could not understand the due "
            f"date '{due_string}' so the task has no due date. Tell the user that "
            f"and ask how they would like to phrase it."
        )

    return f"Created {_format_task(task)}"


@tool(parse_docstring=True)
def list_tasks() -> str:
    """List the user's open (not yet completed) Todoist tasks."""
    tasks = todoist.get_tasks()
    if not tasks:
        return "There are no open tasks."
    return "Open tasks:\n" + "\n".join(_format_task(task) for task in tasks)


@tool(parse_docstring=True)
def complete_task(task: str) -> str:
    """Mark one of the user's Todoist tasks as complete.

    Args:
        task: Either the task's id (as shown by list_tasks) or part of its name,
            e.g. "iTrade PR review". Matching on the name is case-insensitive.
    """
    open_tasks = todoist.get_tasks()
    if not open_tasks:
        raise LookupError("There are no open tasks, so there is nothing to complete.")

    needle = task.strip()
    matches = [item for item in open_tasks if str(item.get("id")) == needle]
    if not matches:
        matches = [
            item for item in open_tasks if needle.lower() in (item.get("content") or "").lower()
        ]

    if not matches:
        raise LookupError(
            f"No open task matches '{task}'. The open tasks are:\n"
            + "\n".join(_format_task(item) for item in open_tasks)
            + "\nAsk the user which one they meant."
        )

    if len(matches) > 1:
        raise LookupError(
            f"'{task}' matches {len(matches)} open tasks:\n"
            + "\n".join(_format_task(item) for item in matches)
            + "\nAsk the user which one they meant. Do not guess."
        )

    todoist.close_task(matches[0]["id"])
    return f"Completed {_format_task(matches[0])}"


TOOLS = [create_task, list_tasks, complete_task]

"""The five tools the agent can call.

Each returns two things: a short string that goes straight back into the
conversation as an observation (so the wording is aimed at the model, not at a
log file), and an *event* -- the task as Todoist returned it, or before and
after an update -- that the model never sees. LangChain calls the second part
the artifact. The agent loop hands events to memory, which is how "you've moved
gym four times" gets counted without the tools knowing memory exists.

Three rules make the observe step in agent.py work:

  * Raise on failure -- never return a cheerful string for something that did
    not happen.
  * Raise the *right kind* of failure. `NeedsClarification` means only the user
    can resolve it (which task? what date?). `UnexpectedResult` means Todoist
    said OK but the world is not what was asked for, and it carries the repair,
    because after a half-success "try again" is often the wrong move. Anything
    from the API itself arrives as `todoist.TodoistError` with a status code.
    The observe step turns those kinds into verdicts.
  * Verify, don't trust. After a write, check the returned task actually shows
    the change. Todoist accepts a task and silently drops a due date it could
    not parse; a 200 is not the same as "done".
"""

from langchain_core.tools import tool

import todoist

# Phrases that mean "clear the due date" rather than set one.
_REMOVE_DUE = {"no date", "no due date", "none", "remove"}


class NeedsClarification(Exception):
    """Only the user can resolve this: which task, what date, what change."""


class UnexpectedResult(Exception):
    """Todoist accepted the call but the result is not what was asked for.

    `repair` says what a corrected attempt should look like. It matters because
    the naive retry is often wrong: re-running create_task after Todoist dropped
    the due date would create the task twice.

    `event` is what *did* happen, for memory: a task created with the wrong
    date is still a task created.
    """

    def __init__(self, message: str, repair: str, event: dict | None = None):
        super().__init__(message)
        self.repair = repair
        self.event = event


# Todoist's API scores priority 1 (normal) to 4 (urgent); the app shows the
# reverse, p1 (urgent) to p4 (normal). People speak in the app's terms, so the
# tools take p-levels and convert.
def _priority_to_api(p_level: int) -> int:
    return 5 - p_level


def _format_task(task: dict) -> str:
    """One line per task: id, name, then due date and priority when set."""
    due = task.get("due") or {}
    when = due.get("datetime") or due.get("date")
    details = []
    if when:
        details.append(f"due {when}")
    api_priority = task.get("priority") or 1
    if api_priority > 1:
        details.append(f"p{5 - api_priority}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"[{task.get('id')}] {task.get('content')}{suffix}"


def _find_task(reference: str, verb: str) -> dict:
    """Resolve "the PR review" or "8123" to exactly one open task.

    Anything other than exactly one match raises NeedsClarification *with the
    candidates in the message*, so the model can ask a precise question instead
    of guessing -- and it never has an id it did not get from Todoist.
    """
    open_tasks = todoist.get_tasks()
    if not open_tasks:
        raise NeedsClarification(f"There are no open tasks, so there is nothing to {verb}.")

    needle = reference.strip()
    matches = [item for item in open_tasks if str(item.get("id")) == needle]
    if not matches:
        matches = [
            item for item in open_tasks if needle.lower() in (item.get("content") or "").lower()
        ]

    if not matches:
        raise NeedsClarification(
            f"No open task matches '{reference}'. The open tasks are:\n"
            + "\n".join(_format_task(item) for item in open_tasks)
            + "\nTell the user you could not find it and ask which one they meant."
        )

    if len(matches) > 1:
        raise NeedsClarification(
            f"'{reference}' matches {len(matches)} open tasks:\n"
            + "\n".join(_format_task(item) for item in matches)
            + "\nAsk the user which one they meant. Do not guess."
        )

    return matches[0]


@tool(parse_docstring=True, response_format="content_and_artifact")
def create_task(content: str, due_string: str = "") -> tuple[str, dict]:
    """Create a new task in the user's Todoist.

    Args:
        content: What the task is, e.g. "Submit iTrade PR review". It must be the
            actual task, never a placeholder like "the thing" -- if the user was
            vague, ask them instead of calling this.
        due_string: Optional due date in plain English, phrased the way the user
            phrased it, e.g. "tomorrow at 3pm" or "next friday". Todoist parses
            this itself, so do not convert it to a date. Leave empty for a task
            with no due date.
    """
    if not content.strip():
        raise NeedsClarification("The task has no content. Ask the user what the task is.")

    task = todoist.create_task(content.strip(), due_string or None)
    event = {"kind": "created", "task": task}

    if due_string and not task.get("due"):
        # Todoist accepted the task but silently dropped a due date it could not
        # parse. The task now exists, so the repair is an update, not a redo.
        raise UnexpectedResult(
            f"Created {_format_task(task)}, but Todoist did not understand the due "
            f"date '{due_string}', so the task has NO due date.",
            repair=(
                f"The task already exists as id {task.get('id')} -- do not call "
                f"create_task again. Call update_task with task='{task.get('id')}' and "
                f"a simpler due_string such as 'tomorrow at 3pm' or 'next monday', or "
                f"ask the user how to phrase the date."
            ),
            event=event,
        )

    return f"Created {_format_task(task)}", event


@tool(parse_docstring=True, response_format="content_and_artifact")
def list_tasks() -> tuple[str, None]:
    """List the user's open (not yet completed) Todoist tasks."""
    tasks = todoist.get_tasks()
    if not tasks:
        return "There are no open tasks.", None
    return "Open tasks:\n" + "\n".join(_format_task(task) for task in tasks), None


@tool(parse_docstring=True, response_format="content_and_artifact")
def complete_task(task: str) -> tuple[str, dict]:
    """Mark one of the user's Todoist tasks as complete.

    Args:
        task: Either the task's id (as shown by list_tasks) or part of its name,
            e.g. "iTrade PR review". Matching on the name is case-insensitive.
    """
    found = _find_task(task, "complete")
    todoist.close_task(found["id"])
    return f"Completed {_format_task(found)}", {"kind": "completed", "task": found}


@tool(parse_docstring=True, response_format="content_and_artifact")
def update_task(
    task: str, content: str = "", due_string: str = "", priority: int = 0
) -> tuple[str, dict]:
    """Change an existing task's name, due date and/or priority.

    Args:
        task: The task's id (as shown by list_tasks) or part of its name.
        content: New name for the task. Leave empty to keep the current one.
        due_string: New due date in the user's own words, e.g. "next friday 5pm";
            Todoist parses it. "no date" removes the due date. Leave empty to
            keep the current one.
        priority: New priority on Todoist's p-scale, 1 (most urgent, p1) to 4
            (normal, p4). Leave at 0 to keep the current priority.
    """
    changes: dict[str, str | int] = {}
    if content.strip():
        changes["content"] = content.strip()
    if due_string.strip():
        changes["due_string"] = due_string.strip()
    if priority:
        if priority not in (1, 2, 3, 4):
            raise NeedsClarification(
                "Priority must be 1 (most urgent) to 4 (normal). Ask the user which they want."
            )
        changes["priority"] = _priority_to_api(priority)
    if not changes:
        raise NeedsClarification(
            "Nothing to change was given. Ask the user what they want changed: the name, "
            "the due date, or the priority."
        )

    found = _find_task(task, "update")
    updated = todoist.update_task(found["id"], **changes)

    # A 200 is not proof the change took -- check each field we asked for.
    problems = []
    if "content" in changes and updated.get("content") != changes["content"]:
        problems.append(f"the name is still '{updated.get('content')}'")
    if "due_string" in changes:
        wants_no_due = str(changes["due_string"]).lower() in _REMOVE_DUE
        if not wants_no_due and not updated.get("due"):
            problems.append(
                f"Todoist did not understand the due date '{changes['due_string']}' and "
                f"the task now has no due date"
            )
    if "priority" in changes and (updated.get("priority") or 1) != changes["priority"]:
        problems.append(f"the priority is still p{5 - (updated.get('priority') or 1)}")

    if problems:
        # No event: a half-applied update is not a habit, and the repair follows.
        raise UnexpectedResult(
            f"Updated {_format_task(updated)}, but " + "; ".join(problems) + ".",
            repair=(
                f"Call update_task again for task '{found['id']}' with a corrected value "
                f"-- for a date, a simpler phrase such as 'next monday 9am' -- or ask the "
                f"user."
            ),
        )

    return f"Updated {_format_task(updated)}", {
        "kind": "updated",
        "before": found,
        "after": updated,
    }


@tool(parse_docstring=True, response_format="content_and_artifact")
def delete_task(task: str) -> tuple[str, dict]:
    """Delete a task permanently. This cannot be undone, so use it only when the
    user clearly asked to delete or remove a task, not to complete it.

    Args:
        task: The task's id (as shown by list_tasks) or part of its name.
    """
    found = _find_task(task, "delete")
    todoist.delete_task(found["id"])
    return f"Deleted {_format_task(found)}", {"kind": "deleted", "task": found}


TOOLS = [create_task, list_tasks, complete_task, update_task, delete_task]

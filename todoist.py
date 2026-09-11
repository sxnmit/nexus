"""A thin Todoist REST client -- just the calls Nexus needs.

Every failure (bad token, network trouble, a rejected request) becomes a
TodoistError. It carries `status_code` -- None when Todoist was never reached --
because the agent's observe step decides what to do next from it: a 429 is
worth one retry, a 401 is not.
"""

import json
import uuid
from datetime import date, datetime

import httpx

import config

TIMEOUT = 15.0


class TodoistError(RuntimeError):
    """A Todoist call failed. The message is written to be read by the model."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _request(method: str, path: str, **kwargs) -> dict | list | None:
    url = f"{config.TODOIST_API_BASE}{path}"
    headers = {"Authorization": f"Bearer {config.TODOIST_API_TOKEN}"}
    try:
        response = httpx.request(method, url, headers=headers, timeout=TIMEOUT, **kwargs)
    except httpx.HTTPError as exc:
        raise TodoistError(f"Could not reach Todoist ({exc}).") from exc

    if response.status_code >= 400:
        detail = response.text.strip()[:300] or "no detail"
        raise TodoistError(
            f"Todoist rejected the request: HTTP {response.status_code} - {detail}",
            status_code=response.status_code,
        )

    if not response.content:
        return None
    return response.json()


def create_task(content: str, due_string: str | None = None) -> dict:
    """Create a task. `due_string` is plain English -- Todoist parses it."""
    payload: dict[str, str] = {"content": content}
    if due_string:
        payload["due_string"] = due_string
    return _request("POST", "/tasks", json=payload)


def get_tasks() -> list[dict]:
    """Return all open (not yet completed) tasks, following pagination.

    API v1 returns {"results": [...], "next_cursor": ...} a page at a time; the
    older REST v2 returned a bare list. We handle both, and we do follow the
    cursor -- if we only read the first page, a task further down the list would
    look to the agent like it does not exist.
    """
    tasks: list[dict] = []
    cursor: str | None = None

    for _ in range(20):  # ~ thousands of tasks; a hard stop beats an infinite loop
        params = {"cursor": cursor} if cursor else None
        data = _request("GET", "/tasks", params=params)

        if not isinstance(data, dict):  # REST v2: a bare list, no pagination
            return data or []

        tasks.extend(data.get("results") or [])
        cursor = data.get("next_cursor")
        if not cursor:
            break

    return tasks


def get_task(task_id: str) -> dict:
    """Fetch one task by id."""
    return _request("GET", f"/tasks/{task_id}")


def update_task(task_id: str, **fields) -> dict:
    """Change a task's content, due_string and/or priority (API scale: 4 = urgent).

    Returns the task as Todoist now has it. API v1 sends the updated task back;
    should a server ever answer with an empty body instead, we fetch it -- the
    caller's next step is to check that the change actually took.
    """
    payload = {key: value for key, value in fields.items() if value is not None}
    data = _request("POST", f"/tasks/{task_id}", json=payload)
    if isinstance(data, dict) and data:
        return data
    return get_task(task_id)


def delete_task(task_id: str) -> None:
    """Delete a task permanently. Returns 204 with an empty body on success."""
    _request("DELETE", f"/tasks/{task_id}")


def close_task(task_id: str) -> None:
    """Mark a task complete. Returns 204 with an empty body on success."""
    _request("POST", f"/tasks/{task_id}/close")


# --- Reminders: the Sync side of the API --------------------------------------------
# Reminders never appear on the task endpoints above. They live behind the
# older Sync protocol: one POST that applies a batch of commands and/or reads
# back whole resource types. Nexus uses exactly three commands.


def sync(commands: list[dict] | None = None, resource_types: list[str] | None = None) -> dict:
    """One call to the Sync endpoint: apply `commands`, read back `resource_types`.

    A command that fails raises TodoistError carrying the error Todoist gave and
    its HTTP code, so the observe step can judge it and a caller never mistakes
    a half-applied batch for success.
    """
    form: dict[str, str] = {}
    if commands:
        form["commands"] = json.dumps(commands)
    if resource_types:
        form["sync_token"] = "*"  # a full read, not an incremental one
        form["resource_types"] = json.dumps(resource_types)
    result = _request("POST", "/sync", data=form) or {}

    for command in commands or []:
        status = (result.get("sync_status") or {}).get(command["uuid"])
        if status == "ok":
            continue
        if isinstance(status, dict):
            raise TodoistError(
                f"Todoist rejected the request: {status.get('error') or 'no detail'}",
                status_code=status.get("http_code"),
            )
        raise TodoistError("Todoist returned no status for the request.")
    return result


def _command(kind: str, args: dict) -> dict:
    # Both ids must be UUIDs; anything else is "Invalid temporary id".
    return {"type": kind, "temp_id": str(uuid.uuid4()), "uuid": str(uuid.uuid4()), "args": args}


def get_reminders(task_id: str) -> list[dict]:
    """The reminders on one task, as Todoist holds them (deleted ones dropped)."""
    result = sync(resource_types=["reminders"])
    return [
        reminder
        for reminder in result.get("reminders") or []
        if str(reminder.get("item_id")) == str(task_id) and not reminder.get("is_deleted")
    ]


def add_reminder(
    task_id: str, *, minute_offset: int | None = None, due_string: str | None = None
) -> dict | None:
    """Add one reminder: `minute_offset` minutes before the task's due time, or
    at `due_string` in the user's own words.

    Returns the reminder as Todoist holds it afterwards -- which includes the
    moment it will fire -- or None if Todoist said OK but the reminder is not
    on the task. Reading it back is the "verify, don't trust" rule again.
    """
    args: dict = {"item_id": str(task_id)}
    if minute_offset is not None:
        args.update(type="relative", minute_offset=int(minute_offset))
    else:
        args.update(type="absolute", due={"string": due_string})
    command = _command("reminder_add", args)

    result = sync(commands=[command])
    new_id = (result.get("temp_id_mapping") or {}).get(command["temp_id"])
    for reminder in get_reminders(task_id):
        if str(reminder.get("id")) == str(new_id):
            return reminder
    return None


def delete_reminder(reminder_id: str) -> None:
    sync(commands=[_command("reminder_delete", {"id": str(reminder_id)})])


# --- Reading a task's due fields ------------------------------------------------------


def parse_due(task: dict) -> tuple[date | None, datetime | None]:
    """(calendar day, exact moment) for a task, in the configured timezone.

    Todoist gives `due.date` as a calendar day and, for tasks with a time,
    `due.datetime` as RFC 3339 -- UTC with a Z, or a floating local time with
    no zone. Some API versions put the timestamp in `date` itself. All three
    shapes end up here as an aware datetime in NEXUS_TIMEZONE, or None for a
    task with a day but no time.
    """
    due = task.get("due") or {}
    raw_date = due.get("date") or ""
    raw_dt = due.get("datetime") or (raw_date if "T" in raw_date else None)

    when = None
    if raw_dt:
        when = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
        if when.tzinfo is None:
            when = when.replace(tzinfo=config.TIMEZONE)  # floating: it means local
        else:
            when = when.astimezone(config.TIMEZONE)

    if when is not None:
        return when.date(), when
    if raw_date:
        return date.fromisoformat(raw_date[:10]), None
    return None, None


def due_key(task: dict) -> str:
    """The due date exactly as Todoist states it, or "" -- for telling "the same
    task, moved" from "the same task"."""
    due = task.get("due") or {}
    return due.get("datetime") or due.get("date") or ""

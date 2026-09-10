"""A thin Todoist REST client -- just the calls Nexus needs.

Every failure (bad token, network trouble, a rejected request) becomes a
TodoistError. It carries `status_code` -- None when Todoist was never reached --
because the agent's observe step decides what to do next from it: a 429 is
worth one retry, a 401 is not.
"""

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

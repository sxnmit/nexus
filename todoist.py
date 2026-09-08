"""A thin Todoist REST client -- just the three calls Nexus needs.

Every failure (bad token, unparseable date, network trouble) becomes a
TodoistError carrying a human-readable message. tools.py lets those propagate
and the agent's tool node turns them into observations the model can react to,
which is what stops it from claiming success on a call that failed.
"""

import httpx

import config

TIMEOUT = 15.0


class TodoistError(RuntimeError):
    """A Todoist call failed. The message is written to be read by the model."""


def _request(method: str, path: str, **kwargs) -> dict | list | None:
    url = f"{config.TODOIST_API_BASE}{path}"
    headers = {"Authorization": f"Bearer {config.TODOIST_API_TOKEN}"}
    try:
        response = httpx.request(method, url, headers=headers, timeout=TIMEOUT, **kwargs)
    except httpx.HTTPError as exc:
        raise TodoistError(f"Could not reach Todoist ({exc}).") from exc

    if response.status_code >= 400:
        detail = response.text.strip()[:300] or "no detail"
        raise TodoistError(f"Todoist rejected the request: HTTP {response.status_code} - {detail}")

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


def close_task(task_id: str) -> None:
    """Mark a task complete. Returns 204 with an empty body on success."""
    _request("POST", f"/tasks/{task_id}/close")

"""Test doubles shared across the suite.

Nothing here touches the network: Claude is a scripted stand-in and Todoist is
either an in-memory fake (for the tools and the agent) or a recorded HTTP layer
that hands back real httpx.Response objects (for the client itself).
"""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx
from langchain_core.messages import AIMessage, ToolMessage

import todoist

# Due strings the fake Todoist "understands". Anything else is accepted and
# silently dropped -- which is exactly what the real API does.
PARSEABLE_DUE = {
    "tomorrow at 3pm": {"string": "tomorrow at 3pm", "datetime": "2026-09-09T15:00:00"},
    "next monday": {"string": "next monday", "date": "2026-09-14"},
    "friday 5pm": {"string": "friday 5pm", "datetime": "2026-09-11T17:00:00"},
}


class ScriptedModel:
    """Stands in for ChatAnthropic: replays canned replies, records what it saw.

    The graph calls `bind_tools` on it and then invokes either the bound view
    (acting turns) or the bare model (respond turns). `tools_bound` records
    which, per call, so a test can prove an "ask the user" turn had no tools.
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []  # the message list passed to each invoke()
        self.tools_bound = []  # parallel to `seen`: were tools bound for that call?

    def bind_tools(self, tools):
        return _BoundScriptedModel(self, tools)

    def invoke(self, messages, *, with_tools=False):
        self.seen.append(list(messages))
        self.tools_bound.append(with_tools)
        assert self.replies, "ScriptedModel ran out of replies"
        return self.replies.pop(0)

    @property
    def calls(self):
        return len(self.seen)

    def observations(self, turn=-1):
        """Every ToolMessage in the conversation as the model saw it on `turn`.

        The conversation only grows, so this is cumulative: on turn 2 it holds
        turn 1's observations too, in order. `[-1]` is always the newest.
        """
        if not self.seen:
            return []
        return [m for m in self.seen[turn] if isinstance(m, ToolMessage)]


class _BoundScriptedModel:
    def __init__(self, model, tools):
        self.model = model
        self.tools = list(tools)

    def invoke(self, messages):
        return self.model.invoke(messages, with_tools=True)


class FakeTodoist:
    """Replaces todoist._request with an in-memory task list.

    Failure injection: `fail` is a message to raise as TodoistError, with
    `fail_status` as its HTTP status (None = never reached Todoist). By default
    every call fails; `fail_times=N` fails only the first N calls, which is how
    the retry tests make "the second attempt works".
    """

    def __init__(
        self,
        tasks=None,
        fail=None,
        fail_status=None,
        fail_times=None,
        pages=None,
        bare_list=False,
        reminder_error=None,
    ):
        self.tasks = [dict(task) for task in (tasks or [])]
        self.fail = fail
        self.fail_status = fail_status
        self.fail_times = fail_times
        self.failures = 0
        self.pages = pages  # [(results, next_cursor), ...] to exercise pagination
        self.bare_list = bare_list  # mimic REST v2's unpaginated response
        self.created = []
        self.updated = []  # (task_id, payload)
        self.deleted = []
        self.closed = []
        self.calls = []
        self.reminders = []  # what the Sync endpoint holds
        self.commands = []  # every Sync command received
        self.reminder_error = reminder_error  # a sync_status error to answer reminder_add with

    def _find(self, task_id):
        for task in self.tasks:
            if str(task["id"]) == str(task_id):
                return task
        raise todoist.TodoistError(
            f"Todoist rejected the request: HTTP 404 - task {task_id} not found", status_code=404
        )

    def __call__(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))

        if self.fail and (self.fail_times is None or self.failures < self.fail_times):
            self.failures += 1
            raise todoist.TodoistError(self.fail, status_code=self.fail_status)

        parts = path.strip("/").split("/")  # e.g. ["tasks", "900", "close"]

        if method == "GET" and path == "/tasks":
            if self.bare_list:
                return list(self.tasks)
            if self.pages is not None:
                cursor = (kwargs.get("params") or {}).get("cursor")
                results, next_cursor = self.pages[0 if cursor is None else int(cursor)]
                return {"results": results, "next_cursor": next_cursor}
            return {"results": [dict(t) for t in self.tasks], "next_cursor": None}

        if method == "POST" and path == "/tasks":
            payload = kwargs["json"]
            self.created.append(payload)
            task = {"id": str(900 + len(self.created) - 1), "content": payload["content"]}
            due = PARSEABLE_DUE.get(payload.get("due_string", ""))
            if due:
                task["due"] = dict(due)
            self.tasks.append(task)
            return dict(task)

        if method == "GET" and len(parts) == 2:
            return dict(self._find(parts[1]))

        if method == "POST" and len(parts) == 2:
            task = self._find(parts[1])
            payload = kwargs["json"]
            self.updated.append((parts[1], payload))
            if "content" in payload:
                task["content"] = payload["content"]
            if "priority" in payload:
                task["priority"] = payload["priority"]
            if "due_string" in payload:
                due = PARSEABLE_DUE.get(payload["due_string"])
                if due:
                    task["due"] = dict(due)
                else:
                    task.pop("due", None)  # unparseable or "no date": dropped
            return dict(task)

        if method == "POST" and len(parts) == 3 and parts[2] == "close":
            self._find(parts[1])
            self.closed.append(parts[1])
            return None

        if method == "DELETE" and len(parts) == 2:
            task = self._find(parts[1])
            self.tasks.remove(task)
            self.deleted.append(parts[1])
            return None

        if method == "POST" and path == "/sync":
            return self._sync(kwargs.get("data") or {})

        raise AssertionError(f"unexpected Todoist call: {method} {path}")

    def _sync(self, form):
        """Apply the commands, then read back the resources asked for -- the
        same round trip the real endpoint does."""
        response = {"sync_status": {}, "temp_id_mapping": {}}
        for command in json.loads(form.get("commands", "[]")):
            self.commands.append(command)
            status = self._apply(command)
            response["sync_status"][command["uuid"]] = status
            if status == "ok" and command["type"] == "reminder_add":
                response["temp_id_mapping"][command["temp_id"]] = self.reminders[-1]["id"]
        if "reminders" in json.loads(form.get("resource_types", "[]")):
            response["reminders"] = [dict(r) for r in self.reminders]
        return response

    def _apply(self, command):
        args = command["args"]
        if command["type"] == "reminder_add":
            if self.reminder_error:
                return self.reminder_error
            task = next((t for t in self.tasks if str(t["id"]) == str(args["item_id"])), None)
            if task is None:
                return {"error": "Item not found", "error_code": 27, "http_code": 400}
            reminder = {
                "id": f"rem-{len(self.reminders) + 1}",
                "item_id": args["item_id"],
                "type": args["type"],
                "is_deleted": False,
                "notify_uid": "u1",
                "due": None,
            }
            if args["type"] == "relative":
                reminder["minute_offset"] = args["minute_offset"]
                when = _moment(task)
                if when is not None:
                    fires = when - timedelta(minutes=args["minute_offset"])
                    reminder["due"] = {"date": fires.strftime("%Y-%m-%dT%H:%M:%S")}
            else:
                due = PARSEABLE_DUE.get(args["due"]["string"])
                if due:  # an unparseable time leaves a reminder with no due, like a task
                    reminder["due"] = {"date": due.get("datetime") or due.get("date")}
            self.reminders.append(reminder)
            return "ok"
        if command["type"] == "reminder_delete":
            for reminder in self.reminders:
                if reminder["id"] == args["id"]:
                    reminder["is_deleted"] = True
            return "ok"
        return {"error": f"Unknown command {command['type']}", "http_code": 400}


def _moment(task):
    """A task's due time as a naive datetime, or None for day-only / undated."""
    due = task.get("due") or {}
    raw = due.get("datetime") or (due.get("date") if "T" in (due.get("date") or "") else None)
    return datetime.fromisoformat(raw) if raw else None


class HTTPRecorder:
    """Replaces httpx.request, returning real httpx.Response objects."""

    def __init__(self):
        self.calls = []
        self.queued = []
        self.error = None

    def queue(self, status=200, json_body=None, text=None):
        self.queued.append((status, json_body, text))
        return self

    def __call__(self, method, url, **kwargs):
        self.calls.append(
            SimpleNamespace(
                method=method,
                url=url,
                headers=kwargs.get("headers") or {},
                params=kwargs.get("params"),
                json=kwargs.get("json"),
                data=kwargs.get("data"),
                timeout=kwargs.get("timeout"),
            )
        )

        if self.error is not None:
            raise self.error

        status, json_body, text = self.queued.pop(0) if self.queued else (200, None, None)
        request = httpx.Request(method, url)
        if json_body is not None:
            return httpx.Response(status, json=json_body, request=request)
        return httpx.Response(status, text=text or "", request=request)


def tool_call(name, args, call_id="call-1"):
    """An AIMessage asking for a single tool call."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def tool_calls(*specs):
    """An AIMessage asking for several tool calls at once (parallel tool use)."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call-{i}", "type": "tool_call"}
            for i, (name, args) in enumerate(specs)
        ],
    )


def log_rows(memory):
    """The interaction log as dicts, oldest first, with the JSON detail parsed."""
    rows = memory._db.execute(
        "SELECT chat_id, role, kind, text, detail, run_id FROM interactions ORDER BY id"
    ).fetchall()
    return [
        {
            "chat_id": row["chat_id"],
            "role": row["role"],
            "kind": row["kind"],
            "text": row["text"],
            "detail": json.loads(row["detail"]) if row["detail"] else None,
            "run_id": row["run_id"],
        }
        for row in rows
    ]

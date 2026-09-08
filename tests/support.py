"""Test doubles shared across the suite.

Nothing here touches the network: Claude is a scripted stand-in and Todoist is
either an in-memory fake (for the tools and the agent) or a recorded HTTP layer
that hands back real httpx.Response objects (for the client itself).
"""

from types import SimpleNamespace

import httpx
from langchain_core.messages import AIMessage, ToolMessage

import todoist


class ScriptedModel:
    """Stands in for ChatAnthropic: replays canned replies, records what it saw."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []  # the message list passed to each invoke()

    def invoke(self, messages):
        self.seen.append(list(messages))
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


class FakeTodoist:
    """Replaces todoist._request with an in-memory task list."""

    def __init__(self, tasks=None, fail=None, pages=None, bare_list=False):
        self.tasks = list(tasks or [])
        self.fail = fail  # a TodoistError message to raise on every call
        self.pages = pages  # [(results, next_cursor), ...] to exercise pagination
        self.bare_list = bare_list  # mimic REST v2's unpaginated response
        self.created = []
        self.closed = []
        self.calls = []

    def __call__(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))

        if self.fail:
            raise todoist.TodoistError(self.fail)

        if method == "GET" and path == "/tasks":
            if self.bare_list:
                return list(self.tasks)
            if self.pages is not None:
                cursor = (kwargs.get("params") or {}).get("cursor")
                results, next_cursor = self.pages[0 if cursor is None else int(cursor)]
                return {"results": results, "next_cursor": next_cursor}
            return {"results": list(self.tasks), "next_cursor": None}

        if method == "POST" and path == "/tasks":
            payload = kwargs["json"]
            self.created.append(payload)
            task = {"id": "900", "content": payload["content"]}
            # Only this phrasing is "parseable", so tests can drive both branches.
            if payload.get("due_string") == "tomorrow at 3pm":
                task["due"] = {"string": "tomorrow at 3pm", "datetime": "2026-09-09T15:00:00"}
            self.tasks.append(task)
            return task

        if method == "POST" and path.endswith("/close"):
            self.closed.append(path.split("/")[2])
            return None

        raise AssertionError(f"unexpected Todoist call: {method} {path}")


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

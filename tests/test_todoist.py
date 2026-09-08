"""Tests for the Todoist HTTP client.

These drive the real _request() through a recorded httpx layer, so they cover
URL building, auth headers, status handling and pagination rather than mocks of
our own code.
"""

import httpx
import pytest

import config
import todoist

# --- Requests we send ---------------------------------------------------------


def test_create_task_posts_content_and_due_string(http):
    http.queue(200, {"id": "1", "content": "Buy milk"})

    todoist.create_task("Buy milk", "tomorrow at 3pm")

    call = http.calls[0]
    assert call.method == "POST"
    assert call.url == f"{config.TODOIST_API_BASE}/tasks"
    assert call.json == {"content": "Buy milk", "due_string": "tomorrow at 3pm"}
    assert call.headers["Authorization"] == f"Bearer {config.TODOIST_API_TOKEN}"
    assert call.timeout == todoist.TIMEOUT


@pytest.mark.parametrize("due", [None, ""])
def test_create_task_omits_due_string_when_absent(http, due):
    http.queue(200, {"id": "1", "content": "Buy milk"})

    todoist.create_task("Buy milk", due)

    assert http.calls[0].json == {"content": "Buy milk"}


def test_close_task_posts_to_the_close_endpoint(http):
    http.queue(204)

    todoist.close_task("12345")

    assert (http.calls[0].method, http.calls[0].url) == (
        "POST",
        f"{config.TODOIST_API_BASE}/tasks/12345/close",
    )


def test_close_task_tolerates_an_empty_204_body(http):
    http.queue(204)
    assert todoist.close_task("1") is None


def test_get_task_fetches_one_task(http):
    http.queue(200, {"id": "5", "content": "A"})

    assert todoist.get_task("5") == {"id": "5", "content": "A"}
    assert (http.calls[0].method, http.calls[0].url) == (
        "GET",
        f"{config.TODOIST_API_BASE}/tasks/5",
    )


def test_update_task_posts_only_the_given_fields(http):
    http.queue(200, {"id": "5", "content": "B", "priority": 4})

    out = todoist.update_task("5", content="B", priority=4, due_string=None)

    assert out == {"id": "5", "content": "B", "priority": 4}
    assert (http.calls[0].method, http.calls[0].url) == (
        "POST",
        f"{config.TODOIST_API_BASE}/tasks/5",
    )
    assert http.calls[0].json == {"content": "B", "priority": 4}


def test_update_task_fetches_the_task_when_the_server_returns_no_body(http):
    http.queue(204)
    http.queue(200, {"id": "5", "content": "B"})

    assert todoist.update_task("5", content="B") == {"id": "5", "content": "B"}
    assert [c.method for c in http.calls] == ["POST", "GET"]
    assert http.calls[1].url == f"{config.TODOIST_API_BASE}/tasks/5"


def test_delete_task_sends_delete(http):
    http.queue(204)

    assert todoist.delete_task("5") is None
    assert (http.calls[0].method, http.calls[0].url) == (
        "DELETE",
        f"{config.TODOIST_API_BASE}/tasks/5",
    )


def test_base_url_has_no_trailing_slash(monkeypatch):
    # config strips it, so paths never produce a double slash.
    assert not config.TODOIST_API_BASE.endswith("/")
    assert config.TODOIST_API_BASE.startswith("https://")


# --- Listing and pagination ---------------------------------------------------


def test_get_tasks_unwraps_the_v1_results_envelope(http):
    http.queue(200, {"results": [{"id": "1"}, {"id": "2"}], "next_cursor": None})

    assert [t["id"] for t in todoist.get_tasks()] == ["1", "2"]


def test_get_tasks_follows_the_cursor_across_pages(http):
    http.queue(200, {"results": [{"id": "1"}], "next_cursor": "abc"})
    http.queue(200, {"results": [{"id": "2"}], "next_cursor": None})

    assert [t["id"] for t in todoist.get_tasks()] == ["1", "2"]
    assert http.calls[0].params is None
    assert http.calls[1].params == {"cursor": "abc"}


def test_get_tasks_accepts_a_bare_rest_v2_list(http):
    http.queue(200, [{"id": "7"}])

    assert [t["id"] for t in todoist.get_tasks()] == ["7"]


def test_get_tasks_stops_instead_of_paginating_forever(http):
    for _ in range(50):  # a server that always hands back another cursor
        http.queue(200, {"results": [{"id": "x"}], "next_cursor": "more"})

    tasks = todoist.get_tasks()

    assert len(http.calls) == 20, "pagination must be capped"
    assert len(tasks) == 20


def test_get_tasks_handles_an_empty_list(http):
    http.queue(200, {"results": [], "next_cursor": None})
    assert todoist.get_tasks() == []


def test_get_tasks_handles_a_null_results_field(http):
    http.queue(200, {"results": None, "next_cursor": None})
    assert todoist.get_tasks() == []


# --- Failures -----------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500])
def test_http_errors_become_todoist_errors(http, status):
    http.queue(status, text="something went wrong")

    with pytest.raises(todoist.TodoistError) as excinfo:
        todoist.get_tasks()

    assert str(status) in str(excinfo.value)
    assert "something went wrong" in str(excinfo.value)
    assert excinfo.value.status_code == status, "the observe step decides from this"


def test_error_detail_is_truncated_so_it_cannot_flood_the_prompt(http):
    http.queue(400, text="x" * 5000)

    with pytest.raises(todoist.TodoistError) as excinfo:
        todoist.get_tasks()

    assert len(str(excinfo.value)) < 400


def test_an_empty_error_body_still_produces_a_readable_message(http):
    http.queue(500, text="")

    with pytest.raises(todoist.TodoistError, match="no detail"):
        todoist.get_tasks()


def test_network_failures_become_todoist_errors(http):
    http.error = httpx.ConnectError("name resolution failed")

    with pytest.raises(todoist.TodoistError, match="Could not reach Todoist") as excinfo:
        todoist.get_tasks()

    assert excinfo.value.status_code is None, "never reached Todoist"


def test_timeouts_become_todoist_errors(http):
    http.error = httpx.ReadTimeout("too slow")

    with pytest.raises(todoist.TodoistError, match="Could not reach Todoist"):
        todoist.create_task("Buy milk")

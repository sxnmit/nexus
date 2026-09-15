"""The nudge's buttons: Done, Tomorrow, Drop -- the agent's tools without the model."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import buttons
import config
from memory import Memory
from tests.support import log_rows

TZ = ZoneInfo("America/Toronto")
NOW = datetime(2026, 9, 9, 15, 20, tzinfo=TZ)


@pytest.fixture(autouse=True)
def toronto(monkeypatch):
    monkeypatch.setattr(config, "TIMEZONE", TZ)


def make():
    clock = {"now": NOW}
    return Memory(":memory:", clock=lambda: clock["now"]), clock


def timed(task_id, content, hour, minute=0):
    return {
        "id": task_id,
        "content": content,
        "due": {"date": "2026-09-09", "datetime": f"2026-09-09T{hour:02d}:{minute:02d}:00"},
    }


# --- Rows ----------------------------------------------------------------------------


def test_one_task_gets_plain_labels():
    assert buttons.rows([{"id": "1", "content": "A"}]) == (
        (("Done", "task:1:done"), ("Tomorrow", "task:1:tomorrow"), ("Drop", "task:1:drop")),
    )


def test_several_tasks_get_numbered_rows():
    rows = buttons.rows([{"id": "a", "content": "A"}, {"id": "b", "content": "B"}])

    assert rows == (
        (("1 Done", "task:a:done"), ("1 Tomorrow", "task:a:tomorrow"), ("1 Drop", "task:a:drop")),
        (("2 Done", "task:b:done"), ("2 Tomorrow", "task:b:tomorrow"), ("2 Drop", "task:b:drop")),
    )


def test_callback_data_fits_telegrams_limit():
    assert len(buttons.data("6X7rM8Xq9F2pWvG4", "tomorrow").encode()) <= 64


# --- Applying a tap ------------------------------------------------------------------


def test_done_completes_through_the_tool_and_keeps_the_record(todoist_api):
    todo = todoist_api(tasks=[timed("1", "Submit PR", 15)])
    memory, _ = make()

    result = buttons.apply(memory, 7, "1", "done")

    assert result == buttons.Result("Done: 'Submit PR'", True)
    assert todo.closed == ["1"]
    rows = log_rows(memory)
    assert [(row["role"], row["kind"], row["text"]) for row in rows] == [
        ("user", "message", "[tapped Done on task 1]"),
        ("tool", "complete_task", "Completed [1] Submit PR (due 2026-09-09T15:00:00)"),
        ("assistant", "button", "Done: 'Submit PR'"),
    ]
    assert rows[1]["detail"] == {"args": {"task": "1"}, "outcome": "ok"}
    assert len({row["run_id"] for row in rows}) == 1, "one run id across the three rows"
    [run] = memory.runs(NOW - timedelta(hours=1))
    assert (run.kind, run.chat_id, run.trigger, run.outcome) == ("button", 7, "done:1", "ok")
    assert (run.reply, run.tool_calls, run.id) == ("Done: 'Submit PR'", 1, rows[0]["run_id"])
    assert run.trace == {
        "tool": "complete_task",
        "outcome": "ok",
        "text": "Completed [1] Submit PR (due 2026-09-09T15:00:00)",
    }
    assert memory.recent(7) == [
        ("user", "[tapped Done on task 1]"),
        ("assistant", "Done: 'Submit PR'"),
    ], "the conversation memory knows what was tapped"


def test_done_teaches_memory_the_completion(todoist_api):
    todoist_api(tasks=[timed("1", "Gym", 14)])
    memory, _ = make()

    buttons.apply(memory, 7, "1", "done")

    [pattern] = memory.patterns()
    assert (pattern.topic, pattern.completed, pattern.completed_late) == ("gym", 1, 1)


def test_tomorrow_keeps_the_time_of_a_timed_task(todoist_api):
    todo = todoist_api(tasks=[timed("1", "Submit PR", 15)])
    todo.tasks[0]["due"] = {"date": "2026-09-09", "datetime": "2026-09-09T15:00:00"}
    memory, _ = make()
    # The fake Todoist only parses a few due strings; teach it this one.
    from tests import support

    support.PARSEABLE_DUE["tomorrow at 3:00pm"] = {
        "string": "tomorrow at 3:00pm",
        "datetime": "2026-09-10T15:00:00",
    }
    try:
        result = buttons.apply(memory, 7, "1", "tomorrow")
    finally:
        del support.PARSEABLE_DUE["tomorrow at 3:00pm"]

    assert result == buttons.Result("Moved 'Submit PR' to tomorrow at 3:00pm", True)
    assert todo.updated == [("1", {"due_string": "tomorrow at 3:00pm"})]
    assert log_rows(memory)[1]["detail"]["args"] == {
        "task": "1",
        "due_string": "tomorrow at 3:00pm",
    }
    [pattern] = memory.patterns()
    assert (pattern.rescheduled, pattern.pushed_later) == (1, 1)


def test_tomorrow_on_a_day_only_task(todoist_api):
    todo = todoist_api(tasks=[{"id": "1", "content": "Submit PR", "due": {"date": "2026-09-09"}}])
    from tests import support

    support.PARSEABLE_DUE["tomorrow"] = {"string": "tomorrow", "date": "2026-09-10"}
    try:
        result = buttons.apply(make()[0], 7, "1", "tomorrow")
    finally:
        del support.PARSEABLE_DUE["tomorrow"]

    assert result.line == "Moved 'Submit PR' to tomorrow" and result.ok
    assert todo.updated == [("1", {"due_string": "tomorrow"})]


def test_drop_deletes(todoist_api):
    todo = todoist_api(tasks=[timed("1", "Submit PR", 15)])
    memory, _ = make()

    result = buttons.apply(memory, 7, "1", "drop")

    assert result == buttons.Result("Deleted 'Submit PR'", True)
    assert todo.deleted == ["1"]
    assert memory.patterns()[0].deleted == 1


def test_a_task_that_is_gone_is_reported_not_guessed(todoist_api):
    todoist_api(tasks=[timed("2", "Something else", 15)])
    memory, _ = make()

    result = buttons.apply(memory, 7, "1", "done")

    assert result == buttons.Result(
        "That task is no longer open, so there was nothing to do.", False
    )
    [run] = memory.runs(NOW - timedelta(hours=1))
    assert run.outcome == "failed" and run.trace["outcome"] == "clarify"
    assert log_rows(memory)[1]["detail"] == {"args": {"task": "1"}, "outcome": "clarify"}


def test_an_unparsed_move_is_reported_as_todoists_doing(todoist_api):
    todoist_api(tasks=[timed("1", "Submit PR", 15)])  # "tomorrow at 3:00pm" is not in PARSEABLE_DUE
    memory, _ = make()

    result = buttons.apply(memory, 7, "1", "tomorrow")

    assert not result.ok
    assert result.line.startswith(
        "Todoist did not do that as asked: Updated [1] Submit PR, but Todoist did not understand"
    )
    [run] = memory.runs(NOW - timedelta(hours=1))
    assert run.trace["outcome"] == "unexpected"


def test_an_outage_says_the_buttons_still_work(todoist_api):
    todoist_api(fail="HTTP 503 - down", fail_status=503)
    memory, _ = make()

    result = buttons.apply(memory, 7, "1", "drop")

    assert result == buttons.Result(
        "Todoist would not do that right now (HTTP 503 - down). The buttons still work.", False
    )
    [run] = memory.runs(NOW - timedelta(hours=1))
    assert (run.outcome, run.trace["outcome"]) == ("failed", "api_error")


def test_an_unknown_action_is_a_bug_not_a_guess():
    with pytest.raises(ValueError, match="unknown button action"):
        buttons.apply(make()[0], 7, "1", "snooze")

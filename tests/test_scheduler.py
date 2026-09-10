"""Stage 3: proactive messages -- the quiet-hours gate and the three jobs.

Nothing here waits on a clock. `Proactive` takes an injectable clock and an
async `send` that records what it was handed; Todoist is the usual in-memory
fake. The jobs are coroutines, run with asyncio.run().
"""

import asyncio
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from zoneinfo import ZoneInfo

import pytest

import config
import scheduler
from scheduler import Proactive, Window

TZ = ZoneInfo("America/Toronto")
UTC = ZoneInfo("UTC")
TODAY = date(2026, 9, 9)


@pytest.fixture(autouse=True)
def toronto(monkeypatch):
    """Pin the zone so every due-date calculation below is deterministic."""
    monkeypatch.setattr(config, "TIMEZONE", TZ)


class Outbox:
    """The `send` callable: records (chat_id, text)."""

    def __init__(self):
        self.sent = []

    async def __call__(self, chat_id, text):
        self.sent.append((chat_id, text))

    @property
    def texts(self):
        return [text for _, text in self.sent]


def at(hour, minute=0, day=9):
    return datetime(2026, 9, day, hour, minute, tzinfo=TZ)


def make(chat_id=42, quiet=("22:00", "07:00"), now=None):
    """A Proactive with a fake clock you can move: `clock['now'] = at(15, 30)`."""
    clock = {"now": now or at(8)}
    outbox = Outbox()
    proactive = Proactive(
        send=outbox,
        chat_id=chat_id,
        quiet=Window(time.fromisoformat(quiet[0]), time.fromisoformat(quiet[1])),
        clock=lambda: clock["now"],
    )
    return proactive, outbox, clock


def timed(task_id, content, hour, minute=0, day=9, priority=1, utc=False):
    """A task with a time. `utc=True` mimics Todoist's Z-suffixed timestamps."""
    when = datetime(2026, 9, day, hour, minute, tzinfo=TZ)
    raw = (
        when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        if utc
        else when.strftime("%Y-%m-%dT%H:%M:%S")
    )
    return {
        "id": task_id,
        "content": content,
        "priority": priority,
        "due": {"date": when.strftime("%Y-%m-%d"), "datetime": raw},
    }


def dated(task_id, content, day=9, priority=1):
    return {
        "id": task_id,
        "content": content,
        "priority": priority,
        "due": {"date": f"2026-09-{day:02d}"},
    }


def run(coro):
    return asyncio.run(coro)


# --- Quiet hours ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hour", "minute", "quiet"),
    [
        (23, 0, True),
        (3, 0, True),
        (6, 59, True),
        (22, 0, True),
        (7, 0, False),
        (12, 0, False),
        (21, 59, False),
    ],
)
def test_default_window_crosses_midnight(hour, minute, quiet):
    assert Window(time(22, 0), time(7, 0)).is_quiet(at(hour, minute)) is quiet


def test_a_same_day_window():
    window = Window(time(13, 0), time(14, 0))
    assert window.is_quiet(at(13, 30))
    assert not window.is_quiet(at(12, 59))
    assert not window.is_quiet(at(14, 0))


def test_a_zero_length_window_means_no_quiet_hours():
    assert not Window(time(22, 0), time(22, 0)).is_quiet(at(22, 0))


# --- Reading due fields -----------------------------------------------------------


def test_parse_due_date_only():
    assert scheduler._parse_due(dated("1", "x", day=9)) == (TODAY, None)


def test_parse_due_floating_local_time():
    day, when = scheduler._parse_due(timed("1", "x", 15))
    assert (day, when) == (TODAY, at(15))


def test_parse_due_converts_utc_timestamps_to_the_configured_zone():
    day, when = scheduler._parse_due(timed("1", "x", 15, utc=True))
    assert when == at(15) and when.tzinfo is TZ and day == TODAY


def test_parse_due_accepts_a_timestamp_in_the_date_field():
    task = {"id": "1", "due": {"date": "2026-09-09T15:00:00"}}
    assert scheduler._parse_due(task) == (TODAY, at(15))


def test_parse_due_without_a_due_date():
    assert scheduler._parse_due({"id": "1", "content": "x"}) == (None, None)
    assert scheduler._parse_due({"id": "1", "due": None}) == (None, None)


def test_split_today_overdue_future_undated():
    tasks = [
        dated("today", "a", day=9),
        timed("today-timed", "b", 15),
        dated("yesterday", "c", day=8),
        dated("tomorrow", "d", day=10),
        {"id": "undated", "content": "e"},
    ]

    due_today, overdue = scheduler._split(tasks, TODAY)

    assert [t["id"] for t in due_today] == ["today", "today-timed"]
    assert [t["id"] for t in overdue] == ["yesterday"]


# --- Templates ----------------------------------------------------------------------


def test_clock_and_ago_formatting():
    assert scheduler._clock(at(15)) == "3:00pm"
    assert scheduler._clock(at(0, 5)) == "12:05am"
    assert scheduler._clock(at(12)) == "12:00pm"
    assert scheduler._ago(timedelta(minutes=20)) == "20 min"
    assert scheduler._ago(timedelta(hours=2)) == "2h"
    assert scheduler._ago(timedelta(hours=1, minutes=5)) == "1h 05m"


def test_line_shows_time_priority_and_how_overdue():
    assert (
        scheduler._line(timed("1", "Submit PR", 15, priority=4), TODAY)
        == "- Submit PR (3:00pm, p1)"
    )
    assert scheduler._line(dated("2", "Call mum", day=8), TODAY) == "- Call mum (yesterday)"
    assert (
        scheduler._line(dated("3", "Taxes", day=6, priority=3), TODAY)
        == "- Taxes (3 days overdue, p2)"
    )
    assert scheduler._line(dated("4", "Read", day=9), TODAY) == "- Read"


def test_morning_text_with_nothing_to_do():
    assert (
        scheduler.morning_text([], [], TODAY)
        == "Good morning. Nothing due today and nothing overdue."
    )


def test_morning_text_lists_due_and_overdue_and_asks():
    text = scheduler.morning_text(
        [timed("1", "Submit PR", 15)], [dated("2", "Call mum", day=8)], TODAY
    )

    assert text.startswith("Good morning.")
    assert "Due today (1):\n- Submit PR (3:00pm)" in text
    assert "Overdue (1):\n- Call mum (yesterday)" in text
    assert text.endswith("Want to move or reprioritise anything? Say which.")


def test_evening_text_done_and_still_open():
    text = scheduler.evening_text(
        [dated("1", "Done thing")], [dated("2", "Open thing")], TODAY, None
    )

    assert "Done today (1):\n- Done thing" in text
    assert "Still open (1):\n- Open thing" in text
    assert text.endswith("Push any of these to tomorrow? Say which.")


def test_evening_text_when_everything_got_done():
    text = scheduler.evening_text([dated("1", "Done thing")], [], TODAY, None)
    assert text.endswith("Nothing left open for today.")


def test_evening_text_admits_a_late_start():
    text = scheduler.evening_text([], [dated("2", "Open thing")], TODAY, at(14))
    assert "I started at 2:00pm, so I can't see what got done before that" in text


def test_nudge_text_singular_and_plural():
    now = at(15, 20)
    one = scheduler.nudge_text([(at(15), timed("1", "Submit PR", 15))], now)
    assert one == "Overdue: 'Submit PR' was due 3:00pm (20 min ago). Done, or push it?"

    two = scheduler.nudge_text([(at(14), timed("1", "A", 14)), (at(15), timed("2", "B", 15))], now)
    assert two.startswith("Overdue (2):\n- A (due 2:00pm)\n- B (due 3:00pm)")
    assert two.endswith("Done, or push them?")


# --- The gate ------------------------------------------------------------------------


def test_deliver_sends_to_the_chat_id():
    proactive, outbox, _ = make()

    assert run(proactive.deliver("morning", "hello")) is True
    assert outbox.sent == [(42, "hello")]


def test_deliver_is_disabled_without_a_chat_id():
    proactive, outbox, _ = make(chat_id=None)

    assert proactive.enabled is False
    assert run(proactive.deliver("morning", "hello")) is False
    assert outbox.sent == []


def test_deliver_holds_during_quiet_hours():
    proactive, outbox, _ = make(now=at(23))

    assert run(proactive.deliver("nudge", "hello")) is False
    assert outbox.sent == []


def test_deliver_truncates_to_telegrams_limit():
    proactive, outbox, _ = make()

    run(proactive.deliver("morning", "x" * 5000))

    assert len(outbox.texts[0]) == scheduler.MAX_MESSAGE


# --- Morning check-in -----------------------------------------------------------------


def test_morning_checkin_reports_and_snapshots(todoist_api):
    todoist_api(
        tasks=[
            timed("1", "Submit PR", 15),
            dated("2", "Call mum", day=8),
            dated("3", "Later", day=12),
        ]
    )
    proactive, outbox, _ = make()

    run(proactive.morning_checkin())

    text = outbox.texts[0]
    assert "Due today (1):\n- Submit PR (3:00pm)" in text
    assert "Overdue (1):\n- Call mum (yesterday)" in text
    assert "Later" not in text
    assert set(proactive.snapshot) == {"1", "2"}
    assert proactive.snapshot_at == at(8)
    assert proactive.nudged == {"1", "2"}, "reported now, so the nudge job must not repeat them"


def test_morning_checkin_always_sends_even_with_nothing_due(todoist_api):
    todoist_api(tasks=[dated("3", "Later", day=12)])
    proactive, outbox, _ = make()

    run(proactive.morning_checkin())

    assert outbox.texts == ["Good morning. Nothing due today and nothing overdue."]


def test_morning_checkin_is_honest_when_todoist_is_down(todoist_api):
    todoist_api(fail="HTTP 503 - down", fail_status=503)
    proactive, outbox, _ = make()

    run(proactive.morning_checkin())

    assert outbox.texts == ["Morning check-in: I couldn't reach Todoist (HTTP 503 - down)."]
    assert proactive.snapshot is None


def test_morning_checkin_still_snapshots_when_held_by_quiet_hours(todoist_api):
    todoist_api(tasks=[dated("1", "Thing", day=9)])
    proactive, outbox, _ = make(now=at(6, 30))  # morning configured inside quiet hours

    run(proactive.morning_checkin())

    assert outbox.sent == []
    assert set(proactive.snapshot) == {"1"}


# --- Evening review -----------------------------------------------------------------


def test_evening_review_diffs_against_the_morning_snapshot(todoist_api):
    todo = todoist_api(
        tasks=[
            timed("1", "Submit PR", 15),
            dated("2", "Call mum", day=8),
            dated("3", "Read", day=9),
        ]
    )
    proactive, outbox, clock = make()
    run(proactive.morning_checkin())

    todo.tasks = [task for task in todo.tasks if task["id"] != "1"]  # completed during the day
    clock["now"] = at(21)
    run(proactive.evening_review())

    text = outbox.texts[1]
    assert "Done today (1):\n- Submit PR" in text
    assert "Still open (2):\n- Read\n- Call mum (yesterday)" in text
    assert "I started at" not in text


def test_evening_review_without_a_snapshot_says_so(todoist_api):
    todoist_api(tasks=[dated("2", "Read", day=9)])
    proactive, outbox, clock = make(now=at(14))  # process started after the morning job
    clock["now"] = at(21)

    run(proactive.evening_review())

    text = outbox.texts[0]
    assert "I started at 2:00pm" in text
    assert "Still open (1):\n- Read" in text
    assert "Done today" not in text


def test_evening_review_ignores_a_snapshot_from_another_day(todoist_api):
    todoist_api(tasks=[dated("2", "Read", day=10)])
    proactive, outbox, clock = make(now=at(8, day=9))
    run(proactive.morning_checkin())

    clock["now"] = at(21, day=10)
    run(proactive.evening_review())

    assert "I started at" in outbox.texts[1]


def test_evening_review_is_silent_when_there_was_nothing_to_review(todoist_api):
    todoist_api(tasks=[dated("3", "Later", day=12)])
    proactive, outbox, clock = make(now=at(14))
    clock["now"] = at(21)

    run(proactive.evening_review())

    assert outbox.sent == []


def test_evening_review_marks_still_open_tasks_as_reported(todoist_api):
    todoist_api(tasks=[timed("1", "Submit PR", 15)])
    proactive, _, clock = make(now=at(14))
    clock["now"] = at(21)

    run(proactive.evening_review())

    assert proactive.nudged == {"1"}


def test_evening_review_is_honest_when_todoist_is_down(todoist_api):
    todoist_api(fail="HTTP 503 - down", fail_status=503)
    proactive, outbox, _ = make(now=at(21))

    run(proactive.evening_review())

    assert outbox.texts == ["Evening review: I couldn't reach Todoist (HTTP 503 - down)."]


# --- Overdue nudge --------------------------------------------------------------------


def test_nudge_reports_a_timed_task_that_just_went_overdue_once(todoist_api):
    todoist_api(tasks=[timed("1", "Submit PR", 15)])
    proactive, outbox, clock = make(now=at(15, 20))

    run(proactive.overdue_nudge())
    run(proactive.overdue_nudge())
    clock["now"] = at(18)
    run(proactive.overdue_nudge())

    assert outbox.texts == ["Overdue: 'Submit PR' was due 3:00pm (20 min ago). Done, or push it?"]
    assert proactive.nudged == {"1"}


def test_nudge_ignores_date_only_future_and_ancient_tasks(todoist_api):
    todoist_api(
        tasks=[
            dated("date-only", "Overdue by date", day=8),
            timed("future", "Later today", 18),
            timed("ancient", "Last week", 15, day=1),
            {"id": "undated", "content": "Whenever"},
        ]
    )
    proactive, outbox, _ = make(now=at(15, 20))

    run(proactive.overdue_nudge())

    assert outbox.sent == []


def test_nudge_skips_what_the_morning_checkin_already_reported(todoist_api):
    todoist_api(tasks=[timed("1", "Submit PR", 7, day=9)])  # overdue since 7am
    proactive, outbox, clock = make(now=at(8))
    run(proactive.morning_checkin())

    clock["now"] = at(8, 15)
    run(proactive.overdue_nudge())

    assert len(outbox.sent) == 1, "the morning message; no second mention by the nudge"


def test_nudge_held_in_quiet_hours_is_sent_when_the_window_opens(todoist_api):
    todoist_api(tasks=[timed("1", "Late night", 23, day=8)])
    proactive, outbox, clock = make(now=at(23, 30, day=8))

    run(proactive.overdue_nudge())
    assert outbox.sent == [] and proactive.nudged == set(), "held, not forgotten"

    clock["now"] = at(7, 5, day=9)
    run(proactive.overdue_nudge())

    assert len(outbox.sent) == 1 and proactive.nudged == {"1"}


def test_nudge_batches_several_in_time_order(todoist_api):
    todoist_api(tasks=[timed("b", "Second", 15), timed("a", "First", 14, utc=True)])
    proactive, outbox, _ = make(now=at(15, 30))

    run(proactive.overdue_nudge())

    assert outbox.texts[0].startswith("Overdue (2):\n- First (due 2:00pm)\n- Second (due 3:00pm)")


def test_nudge_only_logs_when_todoist_is_down(todoist_api):
    todoist_api(fail="HTTP 503 - down", fail_status=503)
    proactive, outbox, _ = make(now=at(15))

    run(proactive.overdue_nudge())

    assert outbox.sent == [], "a message every 15 minutes about an outage would be nagging"


# --- Wiring ---------------------------------------------------------------------------


def test_schedule_jobs_registers_three_jobs_in_the_configured_zone(monkeypatch):
    monkeypatch.setattr(config, "MORNING_TIME", time(8, 0))
    monkeypatch.setattr(config, "EVENING_TIME", time(21, 0))
    monkeypatch.setattr(config, "OVERDUE_CHECK_MINUTES", 15)
    queue = SimpleNamespace(run_daily=Mock(), run_repeating=Mock())
    proactive, _, _ = make()

    scheduler.schedule_jobs(SimpleNamespace(job_queue=queue), proactive)

    daily = {call.kwargs["name"]: call.kwargs["time"] for call in queue.run_daily.call_args_list}
    assert daily == {"morning": time(8, 0, tzinfo=TZ), "evening": time(21, 0, tzinfo=TZ)}
    repeating = queue.run_repeating.call_args.kwargs
    assert repeating["interval"] == timedelta(minutes=15) and repeating["name"] == "overdue"


def test_scheduled_callbacks_run_the_jobs():
    queue = SimpleNamespace(run_daily=Mock(), run_repeating=Mock())
    proactive, _, _ = make()
    proactive.morning_checkin = AsyncMock()
    proactive.evening_review = AsyncMock()
    proactive.overdue_nudge = AsyncMock()

    scheduler.schedule_jobs(SimpleNamespace(job_queue=queue), proactive)
    for call in queue.run_daily.call_args_list + queue.run_repeating.call_args_list:
        run(call.args[0](context=None))

    proactive.morning_checkin.assert_awaited_once()
    proactive.evening_review.assert_awaited_once()
    proactive.overdue_nudge.assert_awaited_once()


def test_now_uses_the_real_clock_in_the_configured_zone_by_default():
    proactive = Proactive(send=Outbox(), chat_id=1, quiet=Window(time(22, 0), time(7, 0)))
    assert proactive.now().tzinfo is TZ
    assert abs(proactive.now() - datetime.now(TZ)) < timedelta(seconds=5)

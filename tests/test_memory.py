"""Stage 4: the interaction log, the learned habits and the shelf -- and, from
Stage 5, the record of runs.

Every test uses a fresh in-RAM database and a fake clock, so nothing here
depends on the wall clock or the filesystem (except the one test about files).
"""

import sqlite3
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import config
import memory as memory_module
from memory import Evaluation, Memory, Pattern, Run, run_id, topic, words
from tests.support import log_rows

TZ = ZoneInfo("America/Toronto")


@pytest.fixture(autouse=True)
def toronto(monkeypatch):
    monkeypatch.setattr(config, "TIMEZONE", TZ)


def at(hour, minute=0, day=9):
    return datetime(2026, 9, day, hour, minute, tzinfo=TZ)


def make(now=None):
    """A Memory with a clock you can move: `clock['now'] = at(15)`."""
    clock = {"now": now or at(9)}
    return Memory(":memory:", clock=lambda: clock["now"]), clock


def task(content, task_id="1", date=None, dt=None):
    item = {"id": task_id, "content": content}
    if date or dt:
        item["due"] = {key: value for key, value in (("date", date), ("datetime", dt)) if value}
    return item


def moved(before_due, after_due, content="Gym"):
    """The event update_task emits when a task's due date changes."""
    return {
        "kind": "updated",
        "before": task(content, **before_due),
        "after": task(content, **after_due),
    }


def pattern(memory, key):
    return next(p for p in memory.patterns() if p.topic == key)


def seed(memory, *contents):
    for content in contents:
        memory.learn({"kind": "created", "task": task(content)})


# --- Topics -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Go to the gym tomorrow at 6pm", "gym"),
        ("Gym", "gym"),
        ("Submit the iTrade PR review", "submit itrade pr"),
        ("Pay rent (September)", "pay rent"),
        ("Call mum's birthday", "call mum's birthday"),
        ("Dentist 2pm on Friday", "dentist"),
        ("remind me to buy milk", "buy milk"),
        ("Übung machen für Café", "übung machen für"),
        ("", ""),
        ("Tomorrow at 3", ""),
    ],
)
def test_topic_strips_the_noise_and_keeps_three_words(content, expected):
    assert topic(content) == expected


def test_words_keeps_order_and_drops_times():
    assert words("Push the weekly report to next monday 9am") == ["push", "weekly", "report"]


# --- The log --------------------------------------------------------------------------


def test_log_and_recent_replay_the_conversation_oldest_first():
    memory, _ = make()
    memory.log(7, "user", "add milk")
    memory.log(7, "tool", "Created [1] milk", kind="create_task", detail={"outcome": "ok"})
    memory.log(7, "assistant", "Added milk.", kind="reply")
    memory.log(7, "assistant", "Good morning. Nothing due.", kind="morning")

    assert memory.recent(7) == [
        ("user", "add milk"),
        ("assistant", "Added milk."),
        ("assistant", "Good morning. Nothing due."),
    ], "tool rows stay out; check-ins are part of the conversation"


def test_recent_is_per_chat():
    memory, _ = make()
    memory.log(7, "user", "mine")
    memory.log(8, "user", "theirs")

    assert memory.recent(7) == [("user", "mine")]


def test_recent_keeps_only_the_last_n_messages(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_MESSAGES", 2)
    memory, _ = make()
    for text in ["one", "two", "three"]:
        memory.log(7, "user", text)

    assert memory.recent(7) == [("user", "two"), ("user", "three")]


def test_zero_messages_turns_conversation_memory_off(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_MESSAGES", 0)
    memory, _ = make()
    memory.log(7, "user", "hello")

    assert memory.recent(7) == []


def test_recent_forgets_messages_older_than_the_window(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_HOURS", 12)
    memory, clock = make(now=at(8))
    memory.log(7, "user", "this morning's thing")
    clock["now"] = at(21)
    memory.log(7, "user", "this evening's thing")

    clock["now"] = at(7, day=10)  # 23h after the first, 10h after the second
    assert memory.recent(7) == [("user", "this evening's thing")]


def test_timestamps_are_stored_in_utc_so_they_compare_as_strings():
    memory, _ = make(now=at(9))  # 09:00 in Toronto is 13:00 UTC

    memory.log(7, "user", "x")

    (ts,) = memory._db.execute("SELECT ts FROM interactions").fetchone()
    assert ts == "2026-09-09T13:00:00+00:00"


def test_tool_rows_keep_their_arguments_and_outcome():
    memory, _ = make()
    detail = {"args": {"content": "milk"}, "outcome": "ok"}

    memory.log(7, "tool", "Created [1] milk", kind="create_task", detail=detail, run_id="r1")

    assert log_rows(memory) == [
        {
            "chat_id": 7,
            "role": "tool",
            "kind": "create_task",
            "text": "Created [1] milk",
            "detail": detail,
            "run_id": "r1",
        }
    ]


def test_counts():
    memory, _ = make()
    assert memory.counts() == (0, 0)

    memory.log(7, "user", "x")
    seed(memory, "Gym")

    assert memory.counts() == (1, 1)


# --- Learning ----------------------------------------------------------------------


def test_creating_counts_under_the_topic_and_keeps_the_latest_wording():
    memory, _ = make()
    memory.learn({"kind": "created", "task": task("Go to the gym")})
    memory.learn({"kind": "created", "task": task("Gym", task_id="2")})

    found = pattern(memory, "gym")
    assert (found.created, found.sample) == (2, "Gym")
    assert not found.recurring

    seed(memory, "gym")
    assert pattern(memory, "gym").recurring


def test_completing_on_time_is_not_late():
    memory, _ = make(now=at(14))

    memory.learn({"kind": "completed", "task": task("Submit PR", dt="2026-09-09T15:00:00")})

    found = pattern(memory, "submit pr")
    assert (found.completed, found.completed_late, found.late_seconds) == (1, 0, 0)


def test_completing_a_timed_task_late_records_by_how_much():
    memory, _ = make(now=at(16, 30))

    memory.learn({"kind": "completed", "task": task("Submit PR", dt="2026-09-09T15:00:00")})

    found = pattern(memory, "submit pr")
    assert (found.completed_late, found.late_seconds) == (1, 90 * 60)


def test_utc_due_times_are_read_in_the_configured_zone():
    memory, _ = make(now=at(16))  # 20:00 UTC

    memory.learn({"kind": "completed", "task": task("Submit PR", dt="2026-09-09T19:00:00Z")})

    assert pattern(memory, "submit pr").late_seconds == 3600


def test_completing_a_day_only_task_the_next_day_is_a_day_late():
    memory, _ = make(now=at(9, day=10))

    memory.learn({"kind": "completed", "task": task("Taxes", date="2026-09-09")})

    assert pattern(memory, "taxes").late_seconds == 86400


def test_completing_a_day_only_task_on_its_day_is_on_time():
    memory, _ = make(now=at(23))

    memory.learn({"kind": "completed", "task": task("Taxes", date="2026-09-09")})

    assert pattern(memory, "taxes").completed_late == 0


def test_completing_an_undated_task_is_never_late():
    memory, _ = make()

    memory.learn({"kind": "completed", "task": task("Read")})

    found = pattern(memory, "read")
    assert (found.completed, found.completed_late) == (1, 0)


def test_moving_a_due_date_later_is_a_reschedule_pushed_later():
    memory, _ = make()

    memory.learn(moved({"date": "2026-09-09"}, {"date": "2026-09-11"}))

    found = pattern(memory, "gym")
    assert (found.rescheduled, found.pushed_later) == (1, 1)


def test_moving_a_due_date_earlier_is_a_reschedule_but_not_a_push():
    memory, _ = make()

    memory.learn(moved({"dt": "2026-09-11T15:00:00"}, {"dt": "2026-09-09T15:00:00"}))

    found = pattern(memory, "gym")
    assert (found.rescheduled, found.pushed_later) == (1, 0)


def test_clearing_a_due_date_is_a_reschedule_but_not_a_push():
    memory, _ = make()

    memory.learn(moved({"date": "2026-09-09"}, {}))

    found = pattern(memory, "gym")
    assert (found.rescheduled, found.pushed_later) == (1, 0)


def test_a_day_task_given_a_time_on_the_same_day_is_not_later():
    memory, _ = make()

    memory.learn(moved({"date": "2026-09-09"}, {"dt": "2026-09-09T15:00:00"}))

    found = pattern(memory, "gym")
    assert (found.rescheduled, found.pushed_later) == (1, 0)


def test_dating_an_undated_task_is_not_a_reschedule():
    memory, _ = make()

    memory.learn(moved({}, {"date": "2026-09-09"}))

    assert memory.patterns() == []


def test_an_update_that_keeps_the_date_teaches_nothing():
    memory, _ = make()

    memory.learn(moved({"date": "2026-09-09"}, {"date": "2026-09-09"}))

    assert memory.patterns() == []


def test_deleting_counts():
    memory, _ = make()

    memory.learn({"kind": "deleted", "task": task("Gym")})

    assert pattern(memory, "gym").deleted == 1


@pytest.mark.parametrize(
    "event",
    [
        None,
        {},
        {"kind": "sparkle"},
        {"kind": "created", "task": {"id": "1", "content": "   "}},
        {"kind": "created", "task": {"id": "1", "content": "tomorrow at 3"}},
        {"kind": "created", "task": {"id": "1"}},
    ],
)
def test_events_that_teach_nothing_are_ignored(event):
    memory, _ = make()

    memory.learn(event)

    assert memory.patterns() == []


def test_counters_accumulate_across_events():
    memory, _ = make()
    seed(memory, "Gym")
    memory.learn(moved({"date": "2026-09-09"}, {"date": "2026-09-10"}))
    memory.learn(moved({"date": "2026-09-10"}, {"date": "2026-09-12"}))
    memory.learn({"kind": "completed", "task": task("Gym")})

    found = pattern(memory, "gym")
    assert (found.created, found.rescheduled, found.pushed_later, found.completed) == (1, 2, 2, 1)


def test_patterns_come_most_recently_touched_first():
    memory, clock = make()
    for hour, name in enumerate(["Gym", "Read", "Taxes"], start=9):
        clock["now"] = at(hour)
        seed(memory, name)
    clock["now"] = at(12)
    seed(memory, "Gym")

    assert [p.topic for p in memory.patterns()] == ["gym", "taxes", "read"]


# --- Reading a pattern ------------------------------------------------------------


def test_recurring_threshold():
    assert Pattern("gym", "Gym", created=3).recurring
    assert not Pattern("gym", "Gym", created=2).recurring


def test_slipping_means_moved_a_few_times_mostly_later():
    assert Pattern("gym", "Gym", rescheduled=2, pushed_later=1).slips
    assert not Pattern("gym", "Gym", rescheduled=1, pushed_later=1).slips
    assert not Pattern("gym", "Gym", rescheduled=4, pushed_later=1).slips, "mostly pulled earlier"


def test_usually_late_means_late_a_few_times_and_at_least_half_the_time():
    assert Pattern("r", "Report", completed=3, completed_late=2, late_seconds=7200).usually_late
    assert not Pattern("r", "Report", completed=5, completed_late=2, late_seconds=7200).usually_late
    assert not Pattern("r", "Report", completed=1, completed_late=1, late_seconds=7200).usually_late


def test_lateness_is_the_average_over_the_late_completions():
    late = Pattern("r", "Report", completed=3, completed_late=2, late_seconds=7200)
    assert late.lateness == timedelta(hours=1)
    assert Pattern("r", "Report").lateness == timedelta(0)


def test_notable_means_something_worth_saying():
    assert not Pattern("gym", "Gym", created=1, completed=1).notable
    assert Pattern("gym", "Gym", created=3).notable
    assert Pattern("gym", "Gym", rescheduled=2, pushed_later=2).notable
    assert Pattern("r", "Report", completed=2, completed_late=2, late_seconds=100).notable


def test_describe_reads_as_one_plain_line():
    line = Pattern(
        "gym",
        "Gym",
        created=5,
        rescheduled=4,
        pushed_later=3,
        completed=2,
        completed_late=1,
        late_seconds=86400,
        deleted=1,
    ).describe()

    assert line == (
        "'Gym': added 5 times (recurring); moved 4 times (3 to later); "
        "done twice, late once by about 1 day on average; deleted once"
    )


def test_describe_leaves_out_zero_counters():
    assert Pattern("gym", "Gym", created=1).describe() == "'Gym': added once"


def test_advice_suggests_rather_than_nags():
    always = Pattern("gym", "Gym", rescheduled=4, pushed_later=4)
    assert always.advice("Gym") == (
        "'Gym' keeps slipping (moved 4 times before, always later). "
        "Drop it, or give it a fixed slot?"
    )

    mostly = Pattern("gym", "Gym", rescheduled=3, pushed_later=2)
    assert mostly.advice("Gym") == (
        "'Gym' keeps slipping (moved 3 times before). Drop it, or give it a fixed slot?"
    )

    late = Pattern("r", "Report", completed=2, completed_late=2, late_seconds=2 * 5400)
    assert late.advice("Weekly report") == (
        "'Weekly report' usually gets done about 2 hours late. Want to date it later up front?"
    )

    assert Pattern("gym", "Gym", created=3).advice("Gym") == "", "recurring alone is not advice"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (30, "about 1 min"),
        (20 * 60, "about 20 min"),
        (89 * 60, "about 89 min"),
        (90 * 60, "about 2 hours"),
        (23 * 3600, "about 23 hours"),
        (24 * 3600, "about 1 day"),
        (35 * 3600, "about 1 day"),
        (36 * 3600, "about 2 days"),
        (3 * 86400, "about 3 days"),
    ],
)
def test_about(seconds, expected):
    assert memory_module._about(timedelta(seconds=seconds)) == expected


# --- Relevance ----------------------------------------------------------------------


def test_patterns_for_matches_when_the_shorter_side_is_inside_the_longer():
    memory, _ = make()
    seed(memory, "Gym", "Call mum", "Submit iTrade PR review")

    def found(*texts):
        return [p.topic for p in memory.patterns_for(texts)]

    assert found("push gym to friday") == ["gym"]
    assert found("add call mum tomorrow") == ["call mum"]
    assert found("call the dentist") == [], "one shared word out of two is not a match"
    assert found("Gym session") == ["gym"], "a task name inside a longer topic"
    assert found("submit the itrade pr review") == ["submit itrade pr"]
    assert found("review") == []
    assert found("", "tomorrow at 3") == []


def test_patterns_for_honours_the_limit_most_recent_first():
    memory, clock = make()
    for hour, name in enumerate(["Gym a", "Gym b", "Gym c"], start=9):
        clock["now"] = at(hour)
        seed(memory, name)

    assert [p.topic for p in memory.patterns_for(["gym"], limit=2)] == ["gym c", "gym b"]


def test_note_is_empty_when_nothing_notable_matches():
    memory, _ = make()
    seed(memory, "Gym")

    assert memory.note(["add gym"]) == "", "added once is not a habit"
    assert memory.note(["hello"]) == ""


def test_note_lists_the_notable_topics_the_message_mentions():
    memory, _ = make()
    seed(memory, "Gym", "Gym", "Gym", "Call mum")

    text = memory.note(["push gym and call mum"])

    assert text.startswith("\n\nWhat you know about this user's habits")
    assert "- 'Gym': added 3 times (recurring)" in text
    assert "Call mum" not in text
    assert "Never nag" in text


def test_advice_for_gives_one_sentence_per_slipping_topic():
    memory, _ = make()
    memory.learn(moved({"date": "2026-09-09"}, {"date": "2026-09-10"}))
    memory.learn(moved({"date": "2026-09-10"}, {"date": "2026-09-11"}))
    tasks = [task("Gym", "1"), task("Gym", "2"), task("Read", "3")]

    assert memory.advice_for(tasks) == [
        "'Gym' keeps slipping (moved twice before, always later). Drop it, or give it a fixed slot?"
    ]
    assert memory.advice_for([task("Read")]) == []


def test_summary_without_habits():
    memory, _ = make()
    memory.log(7, "user", "hi")

    assert memory.summary() == (
        "Memory: 1 interaction logged, 0 topics tracked. No habits stand out yet; they show "
        "up after a few moves or late finishes."
    )


def test_summary_lists_the_habits():
    memory, _ = make()
    seed(memory, "Gym", "Gym", "Gym", "Read")

    text = memory.summary()

    assert text.startswith("Memory: 0 interactions logged, 2 topics tracked.\n\nHabits:\n")
    assert "- 'Gym': added 3 times (recurring)" in text
    assert "Read" not in text


# --- The shelf -----------------------------------------------------------------------


def test_the_shelf_round_trips_json():
    memory, _ = make()
    assert memory.load("reported") is None
    assert memory.load("reported", {}) == {}

    memory.save("reported", {"1": "2026-09-09"})
    memory.save("snapshot", {"at": "x", "tasks": {"1": {"id": "1"}}})
    assert memory.load("reported") == {"1": "2026-09-09"}
    assert memory.load("snapshot") == {"at": "x", "tasks": {"1": {"id": "1"}}}

    memory.save("reported", {})
    assert memory.load("reported") == {}


# --- The store itself ----------------------------------------------------------------


def test_settings_defaults():
    assert config.DB_PATH == "nexus.db"
    assert config.MEMORY_MESSAGES == 10
    assert config.MEMORY_HOURS == 12.0


def test_a_file_database_survives_reopening(tmp_path):
    path = str(tmp_path / "nexus.db")
    first = Memory(path)
    first.log(7, "user", "hello")
    seed(first, "Gym")
    first.save("k", 1)

    second = Memory(path)

    assert second.recent(7) == [("user", "hello")]
    assert second.counts() == (1, 1)
    assert second.load("k") == 1


def test_the_default_clock_is_the_configured_zone():
    assert Memory().now().tzinfo is TZ


def test_writes_from_many_threads_are_serialised():
    memory, _ = make()

    def work(n):
        for i in range(25):
            memory.log(n, "user", f"{n}-{i}")
            seed(memory, f"Gym {n}")  # a trailing digit is noise, so one topic

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert memory.counts() == (200, 1)
    assert pattern(memory, "gym").created == 200


# --- The record ---------------------------------------------------------------------


def a_run(when, **overrides):
    values = {
        "id": "abc123def456",
        "ts": when,
        "chat_id": 7,
        "kind": "message",
        "trigger": "add milk",
        "reply": "Added it.",
        "outcome": "ok",
        "latency_ms": 1234,
        "steps": 2,
        "retries": 1,
        "calls": 2,
        "tool_calls": 1,
        "input_tokens": 300,
        "output_tokens": 40,
        "cache_read_tokens": 100,
        "model": "claude-haiku-4-5",
        "prompt": "deadbeef0123",
        "build": "4c43a30",
        "error": "",
        "trace": {"steps": [{"role": "tool", "outcome": "ok"}], "reply": "Added it."},
    }
    return Run(**{**values, **overrides})


def test_record_and_runs_round_trip():
    memory, clock = make()
    run = a_run(clock["now"])

    memory.record(run)

    assert memory.runs(at(0)) == [run]
    assert memory.runs(at(0))[0].ts.tzinfo is TZ, "read back in the configured zone"


def test_runs_come_from_the_window_oldest_first():
    memory, _ = make()
    memory.record(a_run(at(12), id="noon"))
    memory.record(a_run(at(9), id="nine"))
    memory.record(a_run(at(15), id="three"))
    memory.record(a_run(at(9), id="nine-again", kind="overdue"))

    assert [run.id for run in memory.runs(at(0))] == ["nine", "nine-again", "noon", "three"]
    assert [run.id for run in memory.runs(at(9, 1), at(15))] == ["noon"], "until is exclusive"
    assert memory.runs(at(16)) == []


def test_a_run_needs_only_what_every_run_has():
    memory, clock = make()
    memory.record(Run("id1", clock["now"], 0, "morning", "clock", "", "held", 12))

    [run] = memory.runs(at(0))
    assert (run.kind, run.outcome, run.reply, run.trace) == ("morning", "held", "", {})
    assert (run.steps, run.input_tokens, run.model, run.error) == (0, 0, "", "")


def test_run_ids_are_short_and_unique():
    ids = {run_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(len(one) == 12 and int(one, 16) >= 0 for one in ids)


def test_log_rows_carry_the_run_that_made_them():
    memory, _ = make()
    memory.log(7, "user", "add milk", run_id="run-1")
    memory.log(7, "assistant", "Good morning.", kind="morning")

    assert [(row["text"], row["run_id"]) for row in log_rows(memory)] == [
        ("add milk", "run-1"),
        ("Good morning.", None),
    ]


def test_an_older_database_gains_the_run_id_column(tmp_path):
    """A database written before the record existed has an interactions table
    without run_id; opening it must add the column, not fail on the first log."""
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE interactions (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, chat_id INTEGER "
        "NOT NULL, role TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL, detail TEXT)"
    )
    old.execute(
        "INSERT INTO interactions (ts, chat_id, role, kind, text) VALUES "
        "('2026-09-09T12:00:00+00:00', 7, 'user', 'message', 'hello from before')"
    )
    old.commit()
    old.close()

    memory = Memory(path, clock=lambda: at(13))
    memory.log(7, "user", "hello from now", run_id="r1")

    assert [(row["text"], row["run_id"]) for row in log_rows(memory)] == [
        ("hello from before", None),
        ("hello from now", "r1"),
    ]
    assert memory.runs(at(0)) == [], "the runs table exists too"
    assert Memory(path).counts() == (2, 0), "reopening does not add the column twice"


def test_a_run_row_from_before_labels_gains_the_label_columns(tmp_path):
    """The runs table shipped without labels; a database from that version
    must gain them on open."""
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE runs (id TEXT PRIMARY KEY, ts TEXT NOT NULL, chat_id INTEGER NOT NULL, "
        "kind TEXT NOT NULL, trigger TEXT NOT NULL, reply TEXT NOT NULL, outcome TEXT NOT NULL, "
        "latency_ms INTEGER NOT NULL, steps INTEGER NOT NULL DEFAULT 0, retries INTEGER NOT "
        "NULL DEFAULT 0, calls INTEGER NOT NULL DEFAULT 0, tool_calls INTEGER NOT NULL DEFAULT "
        "0, input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0, "
        "cache_read_tokens INTEGER NOT NULL DEFAULT 0, model TEXT NOT NULL DEFAULT '', prompt "
        "TEXT NOT NULL DEFAULT '', build TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT "
        "'', trace TEXT NOT NULL DEFAULT '{}')"
    )
    old.execute(
        "INSERT INTO runs (id, ts, chat_id, kind, trigger, reply, outcome, latency_ms) VALUES "
        "('old1', '2026-09-09T12:00:00+00:00', 7, 'message', 'hi', 'hello', 'ok', 5)"
    )
    old.commit()
    old.close()

    memory = Memory(path, clock=lambda: at(13))

    [run] = memory.runs(at(0))
    assert (run.id, run.label, run.label_note, run.message_id) == ("old1", None, None, None)
    assert memory.label("old1", "bad", "wrong")
    assert memory.latest_run(7).label == "bad"


# --- Labels and message ids ----------------------------------------------------------


def test_latest_run_is_the_newest_reply_in_the_chat():
    memory, _ = make()
    memory.record(a_run(at(9), id="first", chat_id=7))
    memory.record(a_run(at(10), id="second", chat_id=7))
    memory.record(a_run(at(11), id="elsewhere", chat_id=8))
    memory.record(a_run(at(12), id="job", chat_id=7, kind="overdue"))

    assert memory.latest_run(7).id == "second", "the newest *reply*, not the newest row"
    assert memory.latest_run(8).id == "elsewhere"
    assert memory.latest_run(9) is None


def test_a_reply_can_be_found_by_its_telegram_message():
    memory, _ = make()
    memory.record(a_run(at(9), id="r1", chat_id=7))

    memory.attach_message("r1", 555)

    assert memory.run_for_message(7, 555).id == "r1"
    assert memory.run_for_message(7, 556) is None
    assert memory.run_for_message(8, 555) is None, "a message id is only unique within a chat"
    assert memory.latest_run(7).message_id == 555


def test_label_sets_clears_and_reports_whether_the_run_exists():
    memory, _ = make()
    memory.record(a_run(at(9), id="r1"))

    assert memory.label("r1", "bad", "it made two tasks")
    assert (memory.latest_run(7).label, memory.latest_run(7).label_note) == (
        "bad",
        "it made two tasks",
    )
    assert memory.label("r1", "good")
    assert (memory.latest_run(7).label, memory.latest_run(7).label_note) == ("good", None)
    assert memory.label("r1", None)
    assert memory.latest_run(7).label is None
    assert not memory.label("nope", "bad")


# --- The judge's grades ------------------------------------------------------------------


def an_evaluation(when, run_id="abc123def456", **overrides):
    values = {
        "run_id": run_id,
        "ts": when,
        "judge_model": "claude-haiku-4-5",
        "rubric": "rubric000001",
        "passed": {"told_the_truth": True, "kept_it_short": False},
        "reasons": {"told_the_truth": "backed by the tool", "kept_it_short": "three sentences"},
        "category": "prompt",
        "summary": "Did the job, at length.",
        "input_tokens": 900,
        "output_tokens": 80,
        "latency_ms": 700,
    }
    return Evaluation(**{**values, **overrides})


def test_evaluations_round_trip_and_grading_again_replaces():
    memory, clock = make()
    memory.record(a_run(clock["now"]))
    first = an_evaluation(clock["now"])

    memory.evaluate(first)

    assert memory.evaluations_for(["abc123def456", "other"]) == {"abc123def456": first}
    assert first.failed == ["kept_it_short"] and not first.clean

    second = an_evaluation(at(10), passed={"told_the_truth": True, "kept_it_short": True})
    memory.evaluate(second)
    assert memory.evaluations_for(["abc123def456"]) == {"abc123def456": second}
    assert second.clean and second.failed == []
    assert memory.evaluations_for([]) == {}


def test_ungraded_is_the_replies_without_a_grade_oldest_first():
    memory, _ = make()
    memory.record(a_run(at(12), id="noon"))
    memory.record(a_run(at(9), id="nine"))
    memory.record(a_run(at(10), id="ten-graded"))
    memory.record(a_run(at(11), id="crashed", outcome="crashed", reply=""))
    memory.record(a_run(at(11), id="job", kind="overdue"))
    memory.record(a_run(at(6), id="early"))
    memory.evaluate(an_evaluation(at(13), run_id="ten-graded"))

    assert [run.id for run in memory.ungraded(at(8), limit=10)] == ["nine", "noon"]
    assert [run.id for run in memory.ungraded(at(8), limit=1)] == ["nine"]
    assert [run.id for run in memory.ungraded(at(0), limit=10)] == ["early", "nine", "noon"]

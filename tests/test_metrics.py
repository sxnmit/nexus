"""Stage 5, part one: the scorecard -- counts over the record, by code alone."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import config
import metrics
from memory import Evaluation, Memory, Run

TZ = ZoneInfo("America/Toronto")
NOW = datetime(2026, 9, 9, 18, 0, tzinfo=TZ)


@pytest.fixture(autouse=True)
def toronto(monkeypatch):
    monkeypatch.setattr(config, "TIMEZONE", TZ)


def reply(
    memory,
    minutes_ago,
    trigger="add milk",
    outcome="ok",
    chat_id=1,
    steps=2,
    retries=0,
    tool_calls=1,
    tokens=(100, 10, 40),
    latency=1200,
    observations=(),
    model="claude-haiku-4-5",
    prompt="abc123",
    build="4c43a30",
):
    """Record a message run `minutes_ago` minutes before NOW."""
    run = Run(
        id=f"r-{chat_id}-{minutes_ago}",
        ts=NOW - timedelta(minutes=minutes_ago),
        chat_id=chat_id,
        kind="message",
        trigger=trigger,
        reply="ok",
        outcome=outcome,
        latency_ms=latency,
        steps=steps,
        retries=retries,
        calls=steps,
        tool_calls=tool_calls,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        cache_read_tokens=tokens[2],
        model=model,
        prompt=prompt,
        build=build,
        trace={"steps": [{"role": "tool", **observation} for observation in observations]},
    )
    memory.record(run)
    return run


def job(memory, kind, outcome, minutes_ago=30, note=""):
    memory.record(
        Run(
            id=f"{kind}-{outcome}-{minutes_ago}",
            ts=NOW - timedelta(minutes=minutes_ago),
            chat_id=1,
            kind=kind,
            trigger="clock",
            reply="",
            outcome=outcome,
            latency_ms=30,
            trace={"note": note} if note else {},
        )
    )


# --- Counting -------------------------------------------------------------------


def test_scorecard_counts_replies_by_outcome_and_sums_the_loop():
    memory = Memory()
    reply(memory, 50, steps=3, retries=1, tool_calls=2, tokens=(500, 20, 100), latency=3000)
    reply(memory, 40, outcome="asked", latency=1000)
    reply(memory, 30, outcome="ok", latency=2000, model="claude-haiku-4-5-20251001", build="b2")

    card = metrics.scorecard(memory, now=NOW)

    assert card.replies == {"ok": 2, "asked": 1}
    assert card.reply_count == 3
    assert (card.steps, card.retries, card.tool_calls) == (7, 1, 4)
    assert (card.input_tokens, card.output_tokens, card.cache_read_tokens) == (700, 40, 180)
    assert card.latencies_ms == (3000, 1000, 2000)
    assert (card.model, card.prompt, card.build) == ("claude-haiku-4-5-20251001", "abc123", "b2"), (
        "the build line describes the latest reply"
    )


def test_scorecard_reads_tool_trouble_out_of_the_traces():
    memory = Memory()
    reply(
        memory,
        50,
        observations=[
            {"name": "complete_task", "outcome": "clarify"},
            {"name": "complete_task", "outcome": "ok"},
        ],
    )
    reply(
        memory,
        40,
        observations=[
            {"name": "create_task", "outcome": "unexpected"},
            {"name": "list_tasks", "outcome": "api_error", "status_code": 503},
            {"name": "list_tasks", "outcome": "api_error", "status_code": 503},
            {"name": "update_task", "outcome": "api_error", "status_code": None},
            {"name": "update_task", "outcome": "bad_args"},
            {"name": "nuke", "outcome": "crash"},
        ],
    )

    card = metrics.scorecard(memory, now=NOW)

    assert (card.clarifications, card.unexpected) == (1, 1)
    assert card.tool_errors == {
        ("list_tasks", "503"): 2,
        ("update_task", "api_error"): 1,
        ("update_task", "bad_args"): 1,
        ("nuke", "crash"): 1,
    }, "an error keeps its HTTP status; one that never reached Todoist keeps its kind"


def test_scorecard_counts_check_ins_apart_from_replies():
    memory = Memory()
    reply(memory, 50)
    job(memory, "morning", "sent", 600)
    job(memory, "overdue", "silent", 45)
    job(memory, "overdue", "silent", 30)
    job(memory, "overdue", "held", 15)

    card = metrics.scorecard(memory, now=NOW)

    assert card.reply_count == 1
    assert card.jobs == {("morning", "sent"): 1, ("overdue", "silent"): 2, ("overdue", "held"): 1}
    assert card.steps == 2, "job rows add nothing to the loop numbers"


def test_scorecard_only_looks_at_its_window():
    memory = Memory()
    reply(memory, 60 * 30)  # thirty hours ago
    reply(memory, 60 * 24 * 6)  # six days ago
    reply(memory, 5)

    assert metrics.scorecard(memory, now=NOW).reply_count == 1
    assert metrics.scorecard(memory, days=7, now=NOW).reply_count == 3


def test_scorecard_ends_at_memorys_clock_by_default():
    clock = {"now": NOW}
    memory = Memory(clock=lambda: clock["now"])
    reply(memory, 5)

    card = metrics.scorecard(memory)

    assert card.until == NOW and card.since == NOW - timedelta(days=1)
    assert card.reply_count == 1


# --- Corrections ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second", "gap", "outcome", "expected"),
    [
        ("add milk", "no, oat milk", 1, "ok", True),
        ("add milk", "No. Oat milk.", 1, "ok", True),
        ("add milk", "undo", 1, "ok", True),
        ("add milk", "I meant almond milk", 1, "ok", True),
        ("move the PR review to friday", "move the PR review to friday 5pm", 1, "ok", True),
        ("add milk", "add milk", 6, "ok", False),
        ("move the PR review to friday", "the iTrade PR review", 1, "asked", False),
        ("add milk", "what's on my list?", 1, "ok", False),
        ("add milk", "nothing else today", 1, "ok", False),
        ("add milk", "nope", 1, "ok", True),
        ("add milk", "undo that", 1, "ok", True),
        ("add milk", "wrong", 1, "ok", True),
        ("add milk", "undone already", 1, "ok", False),
        ("", "no", 1, "ok", True),
    ],
)
def test_is_correction(first, second, gap, outcome, expected):
    earlier = Run(
        id="a",
        ts=NOW,
        chat_id=1,
        kind="message",
        trigger=first,
        reply="",
        outcome=outcome,
        latency_ms=1,
    )
    later = Run(
        id="b",
        ts=NOW + timedelta(minutes=gap),
        chat_id=1,
        kind="message",
        trigger=second,
        reply="",
        outcome="ok",
        latency_ms=1,
    )

    assert metrics.is_correction(earlier, later) is expected


def test_a_correction_needs_the_same_chat():
    memory = Memory()
    reply(memory, 10, chat_id=1)
    reply(memory, 9, chat_id=2, trigger="no, not that")

    assert metrics.scorecard(memory, now=NOW).corrections == 0


def test_corrections_are_counted_along_each_chat_in_order():
    memory = Memory()
    reply(memory, 30, trigger="add milk")
    reply(memory, 29, trigger="no, oat milk")  # corrects the first
    reply(memory, 20, trigger="move gym to friday", outcome="asked")
    reply(memory, 19, trigger="the morning gym")  # answers a question: not a correction
    reply(memory, 10, trigger="delete the old gym task")
    reply(memory, 9, trigger="delete the old gym task")  # repeated: a correction

    assert metrics.scorecard(memory, now=NOW).corrections == 2


# --- The text ---------------------------------------------------------------------


def test_text_reads_as_a_short_report():
    memory = Memory()
    reply(
        memory,
        50,
        steps=3,
        retries=1,
        tokens=(30000, 2000, 9000),
        latency=6400,
        observations=[{"name": "list_tasks", "outcome": "api_error", "status_code": 503}],
    )
    reply(
        memory,
        40,
        outcome="asked",
        latency=1500,
        observations=[{"name": "complete_task", "outcome": "clarify"}],
    )
    reply(
        memory,
        30,
        outcome="stuck",
        latency=2100,
        tokens=(11100, 1080, 2920),
        observations=[{"name": "create_task", "outcome": "unexpected"}],
    )
    reply(memory, 27, trigger="no, the other one", latency=800)  # corrects the stuck one
    job(memory, "morning", "sent", 600)
    job(memory, "evening", "silent", 5)
    job(memory, "overdue", "silent", 45)
    job(memory, "overdue", "held", 15)

    assert metrics.scorecard(memory, now=NOW).text() == (
        "Last 24 hours: 4 replies.\n"
        "Outcomes: 2 ok, 1 asked, 1 stuck.\n"
        "Loop: 9 steps, 1 retry, 1 clarification asked for, 1 unexpected result.\n"
        "Tool errors: list_tasks 503 x1.\n"
        "Corrections within 5 min: 1.\n"
        "Judge: nothing graded yet.\n"
        "Tokens: 41,300 in (12,000 from cache), 3,100 out. "
        "Replies took 1.8s typically, 6.4s at worst.\n"
        "Check-ins: evening silent x1; morning sent x1; overdue held x1, silent x1.\n"
        "Build: claude-haiku-4-5, prompt abc123, commit 4c43a30."
    )


def test_text_without_replies():
    memory = Memory()
    job(memory, "overdue", "silent", 45)

    assert metrics.scorecard(memory, now=NOW).text() == (
        "Last 24 hours: no replies.\n"
        "Check-ins: overdue silent x1.\n"
        "Build: model unknown, prompt unknown, commit unknown."
    )


def test_text_for_a_week_and_with_nothing_at_all():
    assert metrics.scorecard(Memory(), days=7, now=NOW).text() == (
        "Last 7 days: no replies.\nBuild: model unknown, prompt unknown, commit unknown."
    )


def test_plural():
    assert metrics._plural(1, "step") == "1 step"
    assert metrics._plural(2, "step") == "2 steps"
    assert metrics._plural(0, "retry", "retries") == "0 retries"


# --- The judge and the labels -----------------------------------------------------


def grade(memory, run, passed=None, category="none"):
    verdicts = {"did_what_was_asked": True, "told_the_truth": True} | (passed or {})
    memory.evaluate(
        Evaluation(
            run_id=run.id,
            ts=NOW,
            judge_model="claude-haiku-4-5",
            rubric="r1",
            passed=verdicts,
            reasons={name: "because" for name in verdicts},
            category=category,
            summary="s",
        )
    )


def test_scorecard_counts_the_judges_grades_and_the_labels():
    memory = Memory()
    graded_clean = reply(memory, 50)
    graded_flawed = reply(memory, 40)
    flawed_twice = reply(memory, 30)
    reply(memory, 20)  # ungraded
    grade(memory, graded_clean)
    grade(memory, graded_flawed, passed={"told_the_truth": False}, category="model")
    grade(
        memory,
        flawed_twice,
        passed={"told_the_truth": False, "did_what_was_asked": False},
        category="prompt",
    )
    memory.label(graded_clean.id, "bad", "it was wrong")  # disagrees with the judge
    memory.label(graded_flawed.id, "bad")  # agrees
    memory.label(flawed_twice.id, "good")  # disagrees

    card = metrics.scorecard(memory, now=NOW)

    assert (card.graded, card.flawed) == (3, 2)
    assert card.failing == {"told_the_truth": 2, "did_what_was_asked": 1}
    assert card.categories == {"model": 1, "prompt": 1}, "'none' is not a category to count"
    assert card.labels == {"bad": 2, "good": 1}
    assert (card.compared, card.agreed) == (3, 1)
    lines = [line for line in card.text().split("\n") if line.startswith(("Judge:", "Labels:"))]
    assert lines == [
        "Judge: 3 of 4 graded, 2 with a failing property (did_what_was_asked x1, "
        "told_the_truth x2); categories model x1, prompt x1.",
        "Labels: 2 bad, 1 good; the judge agreed on 1 of 3.",
    ]


def test_a_label_on_an_ungraded_reply_counts_but_is_not_compared():
    memory = Memory()
    run = reply(memory, 10)
    memory.label(run.id, "good")

    card = metrics.scorecard(memory, now=NOW)

    assert (card.labels, card.compared, card.agreed) == ({"good": 1}, 0, 0)
    assert "Labels: 1 good.\n" in card.text()


def test_judge_line_without_failures_or_categories():
    memory = Memory()
    grade(memory, reply(memory, 10))

    assert (
        "Judge: 1 of 1 graded, 0 with a failing property.\n"
        in metrics.scorecard(memory, now=NOW).text()
    )

"""Tests for the Telegram layer and startup checks.

Handlers are driven with lightweight stand-ins for Update and Context; nothing
here talks to Telegram.
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import bot
import config
from memory import Memory, Run
from tests.support import log_rows

InlineKeyboardButton = bot.InlineKeyboardButton
InlineKeyboardMarkup = bot.InlineKeyboardMarkup

SENT_MESSAGE_ID = 555


def make_update(text="hello", user_id=42, chat_id=7, has_user=True, reaction=None):
    """A Telegram update: a text message by default, or a reaction on one of
    the bot's messages (`reaction` is the new list of emoji, [] for taken back)."""
    message_reaction = None
    if reaction is not None:
        message_reaction = SimpleNamespace(
            chat=SimpleNamespace(id=chat_id),
            message_id=SENT_MESSAGE_ID,
            new_reaction=[SimpleNamespace(type="emoji", emoji=emoji) for emoji in reaction],
        )
    return SimpleNamespace(
        message=SimpleNamespace(
            text=text,
            reply_text=AsyncMock(return_value=SimpleNamespace(message_id=SENT_MESSAGE_ID)),
        ),
        message_reaction=message_reaction,
        effective_user=SimpleNamespace(id=user_id) if has_user else None,
        effective_chat=SimpleNamespace(id=chat_id),
    )


def make_context(
    graph="the-graph", memory=None, proactive=None, judge=None, reviewer=None, args=()
):
    return SimpleNamespace(
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
        bot_data={
            "graph": graph,
            "memory": memory if memory is not None else Memory(),
            "proactive": proactive,
            "judge": judge,
            "reviewer": reviewer,
        },
        args=list(args),
    )


def a_reply(memory, chat_id=7, run_id="run1", reply="Added it."):
    """A recorded reply in `memory`, as agent.run() would leave one."""
    memory.record(
        Run(
            id=run_id,
            ts=memory.now(),
            chat_id=chat_id,
            kind="message",
            trigger="add milk",
            reply=reply,
            outcome="ok",
            latency_ms=1,
        )
    )
    return run_id


@pytest.fixture(autouse=True)
def in_ram_database(monkeypatch):
    """main() opens the memory database first thing; never a real file in tests."""
    monkeypatch.setattr(config, "DB_PATH", ":memory:")


@pytest.fixture
def open_bot(monkeypatch):
    """No allowlist configured: everyone may talk to the bot."""
    monkeypatch.setattr(config, "ALLOWED_TELEGRAM_USER_IDS", set())


@pytest.fixture
def locked_bot(monkeypatch):
    """Only user 42 may talk to the bot."""
    monkeypatch.setattr(config, "ALLOWED_TELEGRAM_USER_IDS", {42})


# --- The allowlist ------------------------------------------------------------


def test_everyone_is_allowed_when_no_allowlist_is_set(open_bot):
    assert bot._is_allowed(make_update(user_id=999))


def test_only_listed_users_are_allowed(locked_bot):
    assert bot._is_allowed(make_update(user_id=42))
    assert not bot._is_allowed(make_update(user_id=43))


def test_updates_without_a_sender_are_rejected_when_locked(locked_bot):
    assert not bot._is_allowed(make_update(has_user=False))


# --- on_message ---------------------------------------------------------------


def test_on_message_runs_the_agent_and_replies(open_bot, monkeypatch):
    calls = []
    monkeypatch.setattr(bot, "run", lambda *args: calls.append(args) or "Added it.")
    update, context = make_update("add milk"), make_context()

    asyncio.run(bot.on_message(update, context))

    assert calls == [("the-graph", "add milk", context.bot_data["memory"], 7)], (
        "graph, text, memory, chat id"
    )
    context.bot.send_chat_action.assert_awaited_once()
    assert context.bot.send_chat_action.await_args.kwargs["chat_id"] == 7
    update.message.reply_text.assert_awaited_once_with("Added it.")


def test_on_message_strips_the_text_before_running_the_agent(open_bot, monkeypatch):
    seen = []
    monkeypatch.setattr(bot, "run", lambda graph, text, memory, chat_id: seen.append(text) or "ok")

    asyncio.run(bot.on_message(make_update("  add milk \n"), make_context()))

    assert seen == ["add milk"]


@pytest.mark.parametrize("text", ["", "   ", None])
def test_on_message_ignores_empty_messages(open_bot, monkeypatch, text):
    monkeypatch.setattr(bot, "run", lambda *a: pytest.fail("agent must not run"))
    update, context = make_update(text), make_context()

    asyncio.run(bot.on_message(update, context))

    update.message.reply_text.assert_not_awaited()
    context.bot.send_chat_action.assert_not_awaited()


def test_on_message_silently_drops_unauthorised_users(locked_bot, monkeypatch):
    monkeypatch.setattr(bot, "run", lambda *a: pytest.fail("agent must not run"))
    update, context = make_update(user_id=2), make_context()

    asyncio.run(bot.on_message(update, context))

    update.message.reply_text.assert_not_awaited()
    context.bot.send_chat_action.assert_not_awaited()


def test_on_message_still_serves_authorised_users_when_locked(locked_bot, monkeypatch):
    monkeypatch.setattr(bot, "run", lambda graph, text, memory, chat_id: "ok")
    update = make_update(user_id=42)

    asyncio.run(bot.on_message(update, make_context()))

    update.message.reply_text.assert_awaited_once_with("ok")


def test_on_message_replies_with_a_fallback_when_the_agent_crashes(open_bot, monkeypatch):
    def boom(graph, text, memory, chat_id):
        raise RuntimeError("api down")

    monkeypatch.setattr(bot, "run", boom)
    update = make_update()

    asyncio.run(bot.on_message(update, make_context()))

    reply = update.message.reply_text.await_args.args[0]
    assert "Something broke" in reply
    assert "api down" not in reply, "internal errors stay in the logs"


def test_on_message_remembers_which_telegram_message_carried_the_reply(open_bot, monkeypatch):
    context = make_context()
    memory = context.bot_data["memory"]

    def fake_run(graph, text, store, chat_id):
        a_reply(store, chat_id)  # what agent.run() leaves in the record
        return "Added it."

    monkeypatch.setattr(bot, "run", fake_run)

    asyncio.run(bot.on_message(make_update("add milk"), context))

    assert memory.run_for_message(7, SENT_MESSAGE_ID).id == "run1"


def test_on_message_copes_when_no_run_was_recorded(open_bot, monkeypatch):
    monkeypatch.setattr(bot, "run", lambda *a: "ok")
    context = make_context()

    asyncio.run(bot.on_message(make_update(), context))

    assert context.bot_data["memory"].run_for_message(7, SENT_MESSAGE_ID) is None


# --- /bad and reactions: the user's own verdicts ------------------------------


def test_on_bad_labels_the_last_reply_with_the_note(open_bot):
    context = make_context(args=["it", "made", "two", "tasks"])
    a_reply(context.bot_data["memory"])
    update = make_update("/bad it made two tasks")

    asyncio.run(bot.on_bad(update, context))

    run = context.bot_data["memory"].latest_run(7)
    assert (run.label, run.label_note) == ("bad", "it made two tasks")
    reply = update.message.reply_text.await_args.args[0]
    assert reply.startswith("Noted. That reply is marked as wrong, with your note")


def test_on_bad_without_a_note(open_bot):
    context = make_context()
    a_reply(context.bot_data["memory"])
    update = make_update("/bad")

    asyncio.run(bot.on_bad(update, context))

    run = context.bot_data["memory"].latest_run(7)
    assert (run.label, run.label_note) == ("bad", None)
    assert "with your note" not in update.message.reply_text.await_args.args[0]


def test_on_bad_says_so_when_there_is_nothing_to_mark(open_bot):
    update = make_update("/bad")

    asyncio.run(bot.on_bad(update, make_context()))

    update.message.reply_text.assert_awaited_once_with(
        "Nothing to mark yet - I haven't replied to anything here."
    )


def test_on_bad_is_silent_for_unauthorised_users(locked_bot):
    context = make_context()
    a_reply(context.bot_data["memory"])
    update = make_update("/bad", user_id=2)

    asyncio.run(bot.on_bad(update, context))

    update.message.reply_text.assert_not_awaited()
    assert context.bot_data["memory"].latest_run(7).label is None


@pytest.mark.parametrize(
    ("reaction", "label"),
    [
        (["\U0001f44e"], "bad"),
        (["\U0001f44d"], "good"),
        (["\U0001f525", "\U0001f44e"], "bad"),
        (["\U0001f525"], None),
        ([], None),
    ],
)
def test_on_reaction_labels_the_run_behind_the_message(open_bot, reaction, label):
    context = make_context()
    memory = context.bot_data["memory"]
    memory.attach_message(a_reply(memory), SENT_MESSAGE_ID)
    memory.label("run1", "bad")  # an earlier verdict, to be replaced or cleared

    asyncio.run(bot.on_reaction(make_update(reaction=reaction), context))

    assert memory.latest_run(7).label == label


def test_on_reaction_ignores_custom_emoji_reactions(open_bot):
    context = make_context()
    memory = context.bot_data["memory"]
    memory.attach_message(a_reply(memory), SENT_MESSAGE_ID)
    update = make_update(reaction=[])
    update.message_reaction.new_reaction = [
        SimpleNamespace(type="custom_emoji", custom_emoji_id="x")
    ]

    asyncio.run(bot.on_reaction(update, context))

    assert memory.latest_run(7).label is None


def test_on_reaction_ignores_messages_that_are_not_replies_of_ours(open_bot):
    context = make_context()
    a_reply(context.bot_data["memory"])  # recorded, but no message id attached

    asyncio.run(bot.on_reaction(make_update(reaction=["\U0001f44e"]), context))

    assert context.bot_data["memory"].latest_run(7).label is None


def test_on_reaction_ignores_unauthorised_users_and_non_reaction_updates(locked_bot):
    context = make_context()
    memory = context.bot_data["memory"]
    memory.attach_message(a_reply(memory), SENT_MESSAGE_ID)

    asyncio.run(bot.on_reaction(make_update(reaction=["\U0001f44e"], user_id=2), context))
    asyncio.run(bot.on_reaction(make_update(), context))

    assert memory.latest_run(7).label is None


# --- /review and the buttons ---------------------------------------------------


def make_query(data, user_id=42):
    query = SimpleNamespace(
        data=data,
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
        message=SimpleNamespace(reply_text=AsyncMock()),
    )
    return SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=7),
        message=None,
        message_reaction=None,
    )


def test_on_review_runs_the_weekly_review_now_and_reports(open_bot):
    reviewer = SimpleNamespace(weekly=AsyncMock(return_value="found 1, asking about 'x' (sent)"))
    update = make_update("/review")

    asyncio.run(bot.on_review(update, make_context(reviewer=reviewer)))

    reviewer.weekly.assert_awaited_once_with(trigger="/review")
    update.message.reply_text.assert_awaited_once_with("Review: found 1, asking about 'x' (sent).")


def test_on_review_is_silent_for_unauthorised_users(locked_bot):
    reviewer = SimpleNamespace(weekly=AsyncMock())
    update = make_update("/review", user_id=2)

    asyncio.run(bot.on_review(update, make_context(reviewer=reviewer)))

    reviewer.weekly.assert_not_awaited()
    update.message.reply_text.assert_not_awaited()


@pytest.mark.parametrize("action", ["yes", "no"])
def test_a_decision_button_settles_the_suggestion_and_replaces_the_ask(open_bot, action):
    reviewer = SimpleNamespace(decide=Mock(return_value=f"Settled by {action}."), evidence=Mock())
    update = make_query(f"sug:abc123:{action}")

    asyncio.run(bot.on_decision(update, make_context(reviewer=reviewer)))

    reviewer.decide.assert_called_once_with("abc123", action)
    update.callback_query.answer.assert_awaited_once()
    update.callback_query.edit_message_text.assert_awaited_once_with(f"Settled by {action}.")
    reviewer.evidence.assert_not_called()


def test_show_me_answers_with_the_evidence_and_keeps_the_buttons(open_bot):
    reviewer = SimpleNamespace(decide=Mock(), evidence=Mock(return_value="Evidence for 'x': ..."))
    update = make_query("sug:abc123:show")

    asyncio.run(bot.on_decision(update, make_context(reviewer=reviewer)))

    reviewer.evidence.assert_called_once_with("abc123")
    update.callback_query.message.reply_text.assert_awaited_once_with("Evidence for 'x': ...")
    update.callback_query.edit_message_text.assert_not_awaited()
    reviewer.decide.assert_not_called()


def test_a_button_pressed_by_a_stranger_is_acknowledged_and_ignored(locked_bot):
    reviewer = SimpleNamespace(decide=Mock(), evidence=Mock())
    update = make_query("sug:abc123:yes", user_id=2)

    asyncio.run(bot.on_decision(update, make_context(reviewer=reviewer)))

    update.callback_query.answer.assert_awaited_once()
    reviewer.decide.assert_not_called()
    update.callback_query.edit_message_text.assert_not_awaited()


def test_on_decision_ignores_updates_without_a_query(open_bot):
    reviewer = SimpleNamespace(decide=Mock(), evidence=Mock())
    update = make_update()
    update.callback_query = None

    asyncio.run(bot.on_decision(update, make_context(reviewer=reviewer)))

    reviewer.decide.assert_not_called()


# --- The nudge's buttons ---------------------------------------------------------


def make_tap(data, rows, text="Overdue: 'Submit PR' was due 3:00pm. Done, or push it?", user_id=42):
    """A callback query from a button on a nudge whose keyboard has `rows`."""
    markup = InlineKeyboardMarkup(
        [[InlineKeyboardButton(label, callback_data=item) for label, item in row] for row in rows]
    )
    query = SimpleNamespace(
        data=data,
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
        message=SimpleNamespace(
            text=text, chat=SimpleNamespace(id=7), reply_markup=markup, reply_text=AsyncMock()
        ),
    )
    return SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=7),
        message=None,
        message_reaction=None,
    )


ONE_ROW = ((("Done", "task:1:done"), ("Tomorrow", "task:1:tomorrow"), ("Drop", "task:1:drop")),)


def test_a_done_tap_completes_the_task_and_edits_the_nudge(open_bot, todoist_api):
    todo = todoist_api(tasks=[{"id": "1", "content": "Submit PR"}])
    context = make_context()
    update = make_tap("task:1:done", ONE_ROW)

    asyncio.run(bot.on_task_button(update, context))

    assert todo.closed == ["1"]
    update.callback_query.answer.assert_awaited_once_with("Done: 'Submit PR'")
    update.callback_query.edit_message_text.assert_awaited_once_with(
        "Overdue: 'Submit PR' was due 3:00pm. Done, or push it?\n\nDone: 'Submit PR'",
        reply_markup=None,
    )
    memory = context.bot_data["memory"]
    assert [(row["role"], row["kind"]) for row in log_rows(memory)] == [
        ("user", "message"),
        ("tool", "complete_task"),
        ("assistant", "button"),
    ]
    [run] = memory.runs(memory.now() - timedelta(days=1))
    assert (run.kind, run.trigger, run.outcome, run.reply) == (
        "button",
        "done:1",
        "ok",
        "Done: 'Submit PR'",
    )


def test_a_tap_on_one_of_several_tasks_keeps_the_other_rows(open_bot, todoist_api):
    todoist_api(tasks=[{"id": "1", "content": "A"}, {"id": "2", "content": "B"}])
    rows = (
        (("1 Done", "task:1:done"), ("1 Drop", "task:1:drop")),
        (("2 Done", "task:2:done"), ("2 Drop", "task:2:drop")),
    )
    update = make_tap("task:2:drop", rows, text="Overdue (2):\n1. A\n2. B")

    asyncio.run(bot.on_task_button(update, make_context()))

    kwargs = update.callback_query.edit_message_text.await_args
    assert kwargs.args[0] == "Overdue (2):\n1. A\n2. B\n\nDeleted 'B'"
    [row] = kwargs.kwargs["reply_markup"].inline_keyboard
    assert [button.callback_data for button in row] == ["task:1:done", "task:1:drop"]


def test_a_failed_tap_explains_and_leaves_the_buttons(open_bot, todoist_api):
    todoist_api(tasks=[])  # the task is gone: completed in the app, say
    update = make_tap("task:1:done", ONE_ROW)

    asyncio.run(bot.on_task_button(update, make_context()))

    update.callback_query.answer.assert_awaited_once_with(
        "That task is no longer open, so there was nothing to do."
    )
    update.callback_query.edit_message_text.assert_not_awaited()


def test_a_tap_by_a_stranger_is_acknowledged_and_ignored(locked_bot, todoist_api):
    todo = todoist_api(tasks=[{"id": "1", "content": "Submit PR"}])
    update = make_tap("task:1:done", ONE_ROW, user_id=2)

    asyncio.run(bot.on_task_button(update, make_context()))

    update.callback_query.answer.assert_awaited_once_with()
    assert todo.closed == []


def test_on_task_button_ignores_updates_without_a_query(open_bot):
    update = make_update()
    update.callback_query = None

    asyncio.run(bot.on_task_button(update, make_context()))


# --- /judge -------------------------------------------------------------------


def test_on_judge_grades_now_and_reports(open_bot):
    judge = SimpleNamespace(nightly=AsyncMock(return_value="graded 2 replies: 2 clean"))
    update = make_update("/judge")

    asyncio.run(bot.on_judge(update, make_context(judge=judge)))

    judge.nightly.assert_awaited_once_with(trigger="/judge")
    update.message.reply_text.assert_awaited_once_with("Judge: graded 2 replies: 2 clean.")


def test_on_judge_is_silent_for_unauthorised_users(locked_bot):
    judge = SimpleNamespace(nightly=AsyncMock())
    update = make_update("/judge", user_id=2)

    asyncio.run(bot.on_judge(update, make_context(judge=judge)))

    judge.nightly.assert_not_awaited()
    update.message.reply_text.assert_not_awaited()


# --- /start -------------------------------------------------------------------


def test_on_start_greets_allowed_users(open_bot):
    update = make_update("/start")

    asyncio.run(bot.on_start(update, make_context()))

    update.message.reply_text.assert_awaited_once_with(bot.GREETING)


def test_on_start_is_silent_for_unauthorised_users(locked_bot):
    update = make_update("/start", user_id=2)

    asyncio.run(bot.on_start(update, make_context()))

    update.message.reply_text.assert_not_awaited()


# --- /nudge -------------------------------------------------------------------


def test_on_nudge_runs_the_overdue_check_and_reports(open_bot):
    proactive = SimpleNamespace(overdue_nudge=AsyncMock(return_value="nudged about 1 task(s)"))
    update = make_update("/nudge")

    asyncio.run(bot.on_nudge(update, make_context(proactive=proactive)))

    proactive.overdue_nudge.assert_awaited_once_with(trigger="/nudge")
    update.message.reply_text.assert_awaited_once_with("Overdue check: nudged about 1 task(s).")


def test_on_nudge_is_silent_for_unauthorised_users(locked_bot):
    proactive = SimpleNamespace(overdue_nudge=AsyncMock())
    update = make_update("/nudge", user_id=2)

    asyncio.run(bot.on_nudge(update, make_context(proactive=proactive)))

    proactive.overdue_nudge.assert_not_awaited()
    update.message.reply_text.assert_not_awaited()


# --- /status ------------------------------------------------------------------


def test_on_status_replies_with_the_scorecard(open_bot, memory):
    update = make_update("/status")

    asyncio.run(bot.on_status(update, make_context(memory=memory)))

    text = update.message.reply_text.await_args.args[0]
    assert text.startswith("Last 24 hours: no replies.")
    assert text.endswith(f"commit {config.COMMIT or 'unknown'}.")


def test_on_status_week_covers_seven_days(open_bot, memory, monkeypatch):
    seen = {}

    def fake_scorecard(store, days):
        seen["days"] = days
        return SimpleNamespace(text=lambda: "card")

    monkeypatch.setattr(bot.metrics, "scorecard", fake_scorecard)

    asyncio.run(bot.on_status(make_update("/status week"), make_context(args=["week"])))
    assert seen["days"] == 7

    asyncio.run(bot.on_status(make_update("/status"), make_context(args=["tomorrow"])))
    assert seen["days"] == 1, "anything that is not 'week' means the last day"


def test_on_status_is_silent_for_unauthorised_users(locked_bot, memory):
    update = make_update("/status", user_id=2)

    asyncio.run(bot.on_status(update, make_context(memory=memory)))

    update.message.reply_text.assert_not_awaited()


# --- Errors -------------------------------------------------------------------


def test_on_error_logs_network_trouble_as_one_warning_line(caplog):
    from telegram.error import NetworkError

    context = SimpleNamespace(error=NetworkError("httpx.ConnectError: nodename nor servname"))

    with caplog.at_level(logging.WARNING, logger="nexus"):
        asyncio.run(bot.on_error(None, context))

    assert caplog.record_tuples == [
        (
            "nexus",
            logging.WARNING,
            "Telegram unreachable (httpx.ConnectError: nodename nor servname); retrying",
        )
    ]


def test_on_error_keeps_the_traceback_for_anything_else(caplog):
    context = SimpleNamespace(error=RuntimeError("boom"))

    with caplog.at_level(logging.ERROR, logger="nexus"):
        asyncio.run(bot.on_error(None, context))

    record = caplog.records[0]
    assert record.levelno == logging.ERROR
    assert record.exc_info[1] is context.error, "a bug keeps its traceback"


# --- /memory ------------------------------------------------------------------


def test_on_memory_replies_with_the_summary(open_bot, memory):
    memory.log(7, "user", "hi")
    update = make_update("/memory")

    asyncio.run(bot.on_memory(update, make_context(memory=memory)))

    reply = update.message.reply_text.await_args.args[0]
    assert reply.startswith("Memory: 1 interaction logged, 0 topics tracked.")


def test_on_memory_is_silent_for_unauthorised_users(locked_bot, memory):
    update = make_update("/memory", user_id=2)

    asyncio.run(bot.on_memory(update, make_context(memory=memory)))

    update.message.reply_text.assert_not_awaited()


# --- main(): startup checks ---------------------------------------------------


def test_main_exits_when_settings_are_missing(monkeypatch):
    monkeypatch.setattr(config, "missing_settings", lambda: ["API_TOKEN_TODOIST"])

    with pytest.raises(SystemExit) as excinfo:
        bot.main()

    assert "API_TOKEN_TODOIST" in str(excinfo.value)
    assert ".env.example" in str(excinfo.value)


def test_main_exits_when_todoist_is_unreachable(monkeypatch, todoist_api):
    monkeypatch.setattr(config, "missing_settings", lambda: [])
    todoist_api(fail="Todoist rejected the request: HTTP 401 - bad token")

    with pytest.raises(SystemExit) as excinfo:
        bot.main()

    assert "401" in str(excinfo.value)
    assert "TODOIST_API_BASE" in str(excinfo.value), "should hint at the API-version fallback"


def test_main_exits_when_claude_rejects_the_key(monkeypatch, todoist_api):
    monkeypatch.setattr(config, "missing_settings", lambda: [])
    todoist_api(tasks=[])

    def refuse():
        raise RuntimeError(
            "This API key is not scoped to a workspace, so this request must include ..."
        )

    monkeypatch.setattr(bot, "check_model", refuse)

    with pytest.raises(SystemExit) as excinfo:
        bot.main()

    message = str(excinfo.value)
    assert "Claude check failed" in message and "not scoped to a workspace" in message
    assert "ANTHROPIC_WORKSPACE_ID" in message, "should point at the fix"


def boot(monkeypatch, todoist_api, chat_id):
    """Everything main() needs, faked, with the proactive recipient set to `chat_id`."""
    monkeypatch.setattr(config, "missing_settings", lambda: [])
    monkeypatch.setattr(config, "TELEGRAM_CHAT_ID", chat_id)
    todoist_api(tasks=[])
    queue = SimpleNamespace(run_daily=Mock(), run_repeating=Mock())
    app = SimpleNamespace(
        bot_data={},
        add_handler=Mock(),
        add_error_handler=Mock(),
        run_polling=Mock(),
        job_queue=queue,
    )
    builder = SimpleNamespace(token=lambda token: SimpleNamespace(build=lambda: app))
    monkeypatch.setattr(bot.Application, "builder", lambda: builder)
    monkeypatch.setattr(bot, "build_graph", lambda: "the-graph")
    monkeypatch.setattr(bot, "check_model", lambda: None)
    return app


def test_main_wires_the_handlers_and_starts_polling(monkeypatch, todoist_api, caplog):
    app = boot(monkeypatch, todoist_api, chat_id=None)

    with caplog.at_level(logging.WARNING, logger="nexus"):
        bot.main()

    assert app.bot_data["graph"] == "the-graph"
    assert isinstance(app.bot_data["memory"], bot.Memory)
    handler_types = [type(call.args[0]).__name__ for call in app.add_handler.call_args_list]
    assert handler_types == ["CommandHandler"] * 7 + [
        "MessageHandler",
        "MessageReactionHandler",
        "CallbackQueryHandler",
        "CallbackQueryHandler",
    ]
    commands = [next(iter(call.args[0].commands)) for call in app.add_handler.call_args_list[:7]]
    assert commands == ["start", "memory", "nudge", "status", "bad", "judge", "review"]
    app.add_error_handler.assert_called_once_with(bot.on_error)
    assert isinstance(app.bot_data["proactive"], bot.Proactive), "/nudge needs it even when off"
    assert isinstance(app.bot_data["judge"], bot.Judge)
    assert isinstance(app.bot_data["reviewer"], bot.Reviewer)
    app.run_polling.assert_called_once_with(
        allowed_updates=["message", "message_reaction", "callback_query"]
    )
    names = [call.kwargs["name"] for call in app.job_queue.run_daily.call_args_list]
    assert names == ["judge", "review"], (
        "no recipient, so no check-ins; the jobs that grade still run"
    )
    assert "Proactive messages OFF" in caplog.text


def test_main_schedules_the_check_ins_when_there_is_a_recipient(monkeypatch, todoist_api):
    app = boot(monkeypatch, todoist_api, chat_id=42)

    bot.main()

    names = [call.kwargs["name"] for call in app.job_queue.run_daily.call_args_list]
    assert names == ["judge", "review", "morning", "evening"]
    assert app.job_queue.run_repeating.call_args.kwargs["name"] == "overdue"


def test_main_opens_memory_before_touching_the_network(monkeypatch, todoist_api, caplog):
    boot(monkeypatch, todoist_api, chat_id=None)

    with caplog.at_level(logging.INFO, logger="nexus"):
        bot.main()

    memory_line = next(line for line in caplog.messages if line.startswith("Memory:"))
    assert memory_line == "Memory: :memory: (0 interactions, 0 topics)"
    assert caplog.messages.index(memory_line) < caplog.messages.index(
        next(line for line in caplog.messages if line.startswith("Todoist OK"))
    )


def test_main_exits_when_the_database_cannot_be_opened(monkeypatch, todoist_api, tmp_path):
    boot(monkeypatch, todoist_api, chat_id=None)
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "no-such-dir" / "nexus.db"))

    with pytest.raises(SystemExit) as excinfo:
        bot.main()

    assert "Could not open the memory database" in str(excinfo.value)
    assert "NEXUS_DB_PATH" in str(excinfo.value)


def test_main_exits_on_a_setting_it_could_not_parse(monkeypatch, todoist_api):
    boot(monkeypatch, todoist_api, chat_id=None)
    monkeypatch.setattr(
        config, "config_errors", lambda: ["NEXUS_QUIET_HOURS='garbage': not enough values"]
    )

    with pytest.raises(SystemExit) as excinfo:
        bot.main()

    assert "Bad settings" in str(excinfo.value)
    assert "NEXUS_QUIET_HOURS" in str(excinfo.value)


def test_sender_sends_through_the_apps_bot():
    app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))

    asyncio.run(bot._sender(app)(42, "hi"))

    app.bot.send_message.assert_awaited_once_with(chat_id=42, text="hi", reply_markup=None)


def test_sender_turns_rows_of_buttons_into_an_inline_keyboard():
    app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))
    rows = ((("Yes", "sug:1:yes"), ("No", "sug:1:no")), (("Done", "task:9:done"),))

    asyncio.run(bot._sender(app)(42, "Build it?", rows))

    markup = app.bot.send_message.await_args.kwargs["reply_markup"]
    assert [
        [(button.text, button.callback_data) for button in row] for row in markup.inline_keyboard
    ] == [[("Yes", "sug:1:yes"), ("No", "sug:1:no")], [("Done", "task:9:done")]]


# --- config -------------------------------------------------------------------


def test_missing_settings_names_every_empty_value(monkeypatch):
    monkeypatch.setattr(config, "_REQUIRED", {"A": "", "B": "set", "C": None})
    assert config.missing_settings() == ["A", "C"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", set()),
        ("42", {42}),
        ("1, 2,3", {1, 2, 3}),
        ("1,,2", {1, 2}),
        ("abc,5", {5}),
        (" 7 ", {7}),
        ("-100123", {-100123}),  # group chat ids are negative
    ],
)
def test_parse_user_ids(raw, expected):
    assert config._parse_user_ids(raw) == expected


def test_todoist_base_url_defaults_to_the_unified_v1_api():
    assert config.TODOIST_API_BASE == "https://api.todoist.com/api/v1"


def test_model_defaults_to_haiku():
    assert config.MODEL == "claude-haiku-4-5"


def test_max_tokens_defaults_to_a_telegram_sized_ceiling():
    assert config.MAX_TOKENS == 1024


# --- config: proactive settings -----------------------------------------------


def test_parse_clock_and_window():
    assert config._parse_clock("08:00") == time(8, 0)
    assert config._parse_clock(" 21:30 ") == time(21, 30)
    assert config._parse_window("22:00-07:00") == (time(22, 0), time(7, 0))
    with pytest.raises(ValueError):
        config._parse_clock("8am")
    with pytest.raises(ValueError):
        config._parse_window("22:00")


def test_parse_chat_id():
    assert config._parse_chat_id("") is None
    assert config._parse_chat_id(" 42 ") == 42
    with pytest.raises(ValueError):
        config._parse_chat_id("me")


def test_parse_timezone_by_name_or_machine_default():
    assert str(config._parse_timezone("America/Toronto")) == "America/Toronto"
    assert config._parse_timezone("") == datetime.now().astimezone().tzinfo


def test_setting_records_a_bad_value_and_uses_the_default(monkeypatch):
    monkeypatch.setattr(config, "_ERRORS", [])
    monkeypatch.setenv("NEXUS_MORNING_TIME", "eight")

    assert config._setting("NEXUS_MORNING_TIME", "08:00", config._parse_clock) == time(8, 0)
    assert config.config_errors()[0].startswith("NEXUS_MORNING_TIME='eight'")


def test_setting_falls_back_to_the_machine_zone_for_an_unknown_timezone(monkeypatch):
    monkeypatch.setattr(config, "_ERRORS", [])
    monkeypatch.setenv("NEXUS_TIMEZONE", "Mars/Olympus")

    zone = config._setting("NEXUS_TIMEZONE", "", config._parse_timezone)

    assert zone == datetime.now().astimezone().tzinfo
    assert config.config_errors()[0].startswith("NEXUS_TIMEZONE='Mars/Olympus'")


def test_setting_is_quiet_about_a_good_value(monkeypatch):
    monkeypatch.setattr(config, "_ERRORS", [])
    monkeypatch.setenv("NEXUS_OVERDUE_CHECK_MINUTES", "30")

    assert config._setting("NEXUS_OVERDUE_CHECK_MINUTES", "15", int) == 30
    assert config.config_errors() == []


def test_proactive_defaults():
    assert time(8, 0) == config.MORNING_TIME
    assert time(21, 0) == config.EVENING_TIME
    assert (time(22, 0), time(7, 0)) == config.QUIET_HOURS
    assert config.OVERDUE_CHECK_MINUTES == 15


def test_chat_id_rule():
    assert config._chat_id(5, {9}) == 5, "an explicit id wins"
    assert config._chat_id(None, {9}) == 9, "a one-person allowlist names the recipient"
    assert config._chat_id(None, {9, 10}) is None, "two people: ambiguous, so off"
    assert config._chat_id(None, set()) is None

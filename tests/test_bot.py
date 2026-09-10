"""Tests for the Telegram layer and startup checks.

Handlers are driven with lightweight stand-ins for Update and Context; nothing
here talks to Telegram.
"""

import asyncio
import logging
from datetime import datetime, time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import bot
import config


def make_update(text="hello", user_id=42, chat_id=7, has_user=True):
    return SimpleNamespace(
        message=SimpleNamespace(text=text, reply_text=AsyncMock()),
        effective_user=SimpleNamespace(id=user_id) if has_user else None,
        effective_chat=SimpleNamespace(id=chat_id),
    )


def make_context(graph="the-graph"):
    return SimpleNamespace(
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
        bot_data={"graph": graph},
    )


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
    monkeypatch.setattr(bot, "run", lambda graph, text: calls.append((graph, text)) or "Added it.")
    update, context = make_update("add milk"), make_context()

    asyncio.run(bot.on_message(update, context))

    assert calls == [("the-graph", "add milk")]
    context.bot.send_chat_action.assert_awaited_once()
    assert context.bot.send_chat_action.await_args.kwargs["chat_id"] == 7
    update.message.reply_text.assert_awaited_once_with("Added it.")


def test_on_message_strips_the_text_before_running_the_agent(open_bot, monkeypatch):
    seen = []
    monkeypatch.setattr(bot, "run", lambda graph, text: seen.append(text) or "ok")

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
    monkeypatch.setattr(bot, "run", lambda graph, text: "ok")
    update = make_update(user_id=42)

    asyncio.run(bot.on_message(update, make_context()))

    update.message.reply_text.assert_awaited_once_with("ok")


def test_on_message_replies_with_a_fallback_when_the_agent_crashes(open_bot, monkeypatch):
    def boom(graph, text):
        raise RuntimeError("api down")

    monkeypatch.setattr(bot, "run", boom)
    update = make_update()

    asyncio.run(bot.on_message(update, make_context()))

    reply = update.message.reply_text.await_args.args[0]
    assert "Something broke" in reply
    assert "api down" not in reply, "internal errors stay in the logs"


# --- /start -------------------------------------------------------------------


def test_on_start_greets_allowed_users(open_bot):
    update = make_update("/start")

    asyncio.run(bot.on_start(update, make_context()))

    update.message.reply_text.assert_awaited_once_with(bot.GREETING)


def test_on_start_is_silent_for_unauthorised_users(locked_bot):
    update = make_update("/start", user_id=2)

    asyncio.run(bot.on_start(update, make_context()))

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
    app = SimpleNamespace(bot_data={}, add_handler=Mock(), run_polling=Mock(), job_queue=queue)
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
    handler_types = [type(call.args[0]).__name__ for call in app.add_handler.call_args_list]
    assert handler_types == ["CommandHandler", "MessageHandler"]
    app.run_polling.assert_called_once()
    assert app.job_queue.run_daily.call_count == 0, "no recipient, so no check-ins"
    assert "Proactive messages OFF" in caplog.text


def test_main_schedules_the_check_ins_when_there_is_a_recipient(monkeypatch, todoist_api):
    app = boot(monkeypatch, todoist_api, chat_id=42)

    bot.main()

    names = [call.kwargs["name"] for call in app.job_queue.run_daily.call_args_list]
    assert names == ["morning", "evening"]
    assert app.job_queue.run_repeating.call_args.kwargs["name"] == "overdue"


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

    app.bot.send_message.assert_awaited_once_with(chat_id=42, text="hi")


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

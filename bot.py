"""Telegram front end. Polls for messages and hands each one to the agent.

Polling, not webhooks -- no ngrok, no public URL, just `python bot.py`.
"""

import asyncio
import logging
import sqlite3

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import NetworkError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config
import todoist
from agent import build_graph, check_model, run
from memory import Memory
from scheduler import Proactive, Window, schedule_jobs

logging.basicConfig(format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("nexus")

GREETING = (
    "Hi, I'm Nexus. I look after your Todoist.\n\n"
    "Try:\n"
    "  remind me to submit the iTrade PR review tomorrow at 3pm\n"
    "  what's on my list?\n"
    "  mark the PR review done\n"
    "  move the PR review to friday 5pm\n"
    "  remind me 30 minutes before the PR review\n"
    "  delete the milk task\n\n"
    f"I'll also check in at {config.MORNING_TIME:%H:%M} with what's due and at "
    f"{config.EVENING_TIME:%H:%M} with what got done. "
    "Send /memory to see what I've learned about your habits."
)


def _is_allowed(update: Update) -> bool:
    """Anyone can message a public bot, so check the sender against the allowlist."""
    if not config.ALLOWED_TELEGRAM_USER_IDS:
        return True  # allowlist not configured
    user = update.effective_user
    return user is not None and user.id in config.ALLOWED_TELEGRAM_USER_IDS


async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if _is_allowed(update):
        await update.message.reply_text(GREETING)


async def on_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A window into memory: how much is logged, and which habits stand out."""
    if _is_allowed(update):
        await update.message.reply_text(context.bot_data["memory"].summary())


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    if not text:
        return

    if not _is_allowed(update):
        user = update.effective_user
        log.warning("Ignored message from user %s (not in allowlist)", user.id if user else "?")
        return

    log.info("<- %s", text)
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        # The graph and the Todoist calls inside it are synchronous, so run them
        # on a worker thread instead of blocking the polling loop.
        reply = await asyncio.to_thread(
            run, context.bot_data["graph"], text, context.bot_data["memory"], chat_id
        )
    except Exception:
        log.exception("Agent run failed")
        reply = "Something broke on my side and I couldn't finish that. Check the bot logs."

    log.info("-> %s", reply)
    await update.message.reply_text(reply)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """python-telegram-bot retries network trouble by itself -- a laptop going
    to sleep, Wi-Fi dropping -- so one line is enough for the log. Anything
    else is a bug and gets its traceback."""
    if isinstance(context.error, NetworkError):
        log.warning("Telegram unreachable (%s); retrying", context.error)
    else:
        log.error("Unhandled error in the bot", exc_info=context.error)


def _sender(app):
    """How proactive messages leave: the same bot, addressed by chat id."""

    async def send(chat_id: int, text: str) -> None:
        await app.bot.send_message(chat_id=chat_id, text=text)

    return send


def main() -> None:
    missing = config.missing_settings()
    if missing:
        raise SystemExit(
            "Missing environment variables: "
            + ", ".join(missing)
            + "\nCopy .env.example to .env and fill them in."
        )

    problems = config.config_errors()
    if problems:
        raise SystemExit(
            "Bad settings:\n  " + "\n  ".join(problems) + "\nSee .env.example for the formats."
        )

    # Local things first: the database must open before it is worth asking the
    # network anything.
    try:
        memory = Memory(config.DB_PATH)
    except sqlite3.OperationalError as exc:
        raise SystemExit(
            f"Could not open the memory database at {config.DB_PATH}: {exc}\n"
            "Set NEXUS_DB_PATH to a writable location (on a host, a mounted volume)."
        ) from exc
    interactions, topics = memory.counts()
    log.info("Memory: %s (%d interactions, %d topics)", config.DB_PATH, interactions, topics)

    # Check Todoist now rather than discovering a bad token mid-conversation.
    try:
        open_tasks = todoist.get_tasks()
    except todoist.TodoistError as exc:
        raise SystemExit(
            f"Todoist check failed: {exc}\n"
            f"Base URL is {config.TODOIST_API_BASE}. Verify API_TOKEN_TODOIST, or set "
            f"TODOIST_API_BASE=https://api.todoist.com/rest/v2 to try the older REST API."
        ) from exc

    log.info("Todoist OK - %d open task(s). Model: %s", len(open_tasks), config.MODEL)

    # Same idea for Claude. The token-count endpoint is free, so one call here
    # surfaces a bad key, or a personal access token that has not been told its
    # workspace, before anyone messages the bot.
    try:
        check_model()
    except Exception as exc:  # whatever it is, we cannot start
        raise SystemExit(
            f"Claude check failed: {exc}\n"
            "If PERSONAL_ACCESS_TOKEN_CLAUDE is an org-level personal access token, set "
            "ANTHROPIC_WORKSPACE_ID in .env (Console -> Settings -> Workspaces, ids start "
            "with wrkspc_). A key created inside a workspace needs no workspace id."
        ) from exc

    log.info("Claude OK.")
    if not config.ALLOWED_TELEGRAM_USER_IDS:
        log.warning(
            "TELEGRAM_ALLOWED_USER_IDS is not set - anyone who finds this bot can edit "
            "your Todoist."
        )

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    app.bot_data["graph"] = build_graph()
    app.bot_data["memory"] = memory
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("memory", on_memory))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_error_handler(on_error)

    log.info("Timezone: %s", config.TIMEZONE)
    proactive = Proactive(
        send=_sender(app),
        chat_id=config.TELEGRAM_CHAT_ID,
        quiet=Window(*config.QUIET_HOURS),
        memory=memory,
    )
    if proactive.enabled:
        schedule_jobs(app, proactive)
        log.info(
            "Proactive: morning %s, evening %s, overdue check every %d min, quiet %s-%s, chat %s",
            f"{config.MORNING_TIME:%H:%M}",
            f"{config.EVENING_TIME:%H:%M}",
            config.OVERDUE_CHECK_MINUTES,
            f"{config.QUIET_HOURS[0]:%H:%M}",
            f"{config.QUIET_HOURS[1]:%H:%M}",
            config.TELEGRAM_CHAT_ID,
        )
    else:
        log.warning(
            "Proactive messages OFF: set TELEGRAM_CHAT_ID (or exactly one id in "
            "TELEGRAM_ALLOWED_USER_IDS) to get the morning and evening check-ins."
        )

    log.info("Nexus is polling. Ctrl-C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()

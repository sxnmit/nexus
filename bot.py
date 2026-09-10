"""Telegram front end. Polls for messages and hands each one to the agent.

Polling, not webhooks -- no ngrok, no public URL, just `python bot.py`.
"""

import asyncio
import logging

from telegram import Update
from telegram.constants import ChatAction
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
    "  delete the milk task"
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


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    if not text:
        return

    if not _is_allowed(update):
        user = update.effective_user
        log.warning("Ignored message from user %s (not in allowlist)", user.id if user else "?")
        return

    log.info("<- %s", text)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        # The graph and the Todoist calls inside it are synchronous, so run them
        # on a worker thread instead of blocking the polling loop.
        reply = await asyncio.to_thread(run, context.bot_data["graph"], text)
    except Exception:
        log.exception("Agent run failed")
        reply = "Something broke on my side and I couldn't finish that. Check the bot logs."

    log.info("-> %s", reply)
    await update.message.reply_text(reply)


def main() -> None:
    missing = config.missing_settings()
    if missing:
        raise SystemExit(
            "Missing environment variables: "
            + ", ".join(missing)
            + "\nCopy .env.example to .env and fill them in."
        )

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
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    log.info("Nexus is polling. Ctrl-C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()

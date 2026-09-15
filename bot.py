"""Telegram front end. Polls for messages and hands each one to the agent.

Polling, not webhooks -- no ngrok, no public URL, just `python bot.py`.
"""

import asyncio
import logging
import sqlite3

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ReactionType
from telegram.error import NetworkError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    MessageReactionHandler,
    filters,
)

import buttons
import config
import metrics
import todoist
from agent import build_graph, check_model, run
from judge import Judge, schedule_nightly
from memory import Memory
from review import Reviewer, schedule_weekly
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
    "An overdue nudge comes with Done / Tomorrow / Drop buttons, so one tap settles it.\n\n"
    "Send /memory to see what I've learned about your habits, /nudge to run the "
    "overdue check right now, and /status for how the last day went.\n\n"
    "If a reply gets it wrong, react to it with a thumbs-down, or send /bad and say why: "
    "it goes in the record, next to what my own judge thought."
)

# A reaction on one of Nexus's replies, read as the user's verdict on it.
LABELS = {"\U0001f44e": "bad", "\U0001f44d": "good"}


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


async def on_nudge(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run the overdue check now and say what it saw -- so the job can be
    tested without waiting for its next tick or reading the log."""
    if _is_allowed(update):
        outcome = await context.bot_data["proactive"].overdue_nudge(trigger="/nudge")
        await update.message.reply_text(f"Overdue check: {outcome}.")


async def on_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The scorecard: the last day by default, `/status week` for seven."""
    if _is_allowed(update):
        days = 7 if (context.args or [""])[0].lower() in ("week", "7d") else 1
        await update.message.reply_text(
            metrics.scorecard(context.bot_data["memory"], days=days).text()
        )


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
    sent = await update.message.reply_text(reply)
    # Which Telegram message carried the reply: a reaction on it names the run.
    memory = context.bot_data["memory"]
    latest = memory.latest_run(chat_id)
    if latest is not None and sent is not None:
        memory.attach_message(latest.id, sent.message_id)


async def on_bad(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/bad [why]: the last reply in this chat was wrong. The note travels
    with the label, so the reviewer sees the reason and not just the verdict."""
    if not _is_allowed(update):
        return
    memory = context.bot_data["memory"]
    latest = memory.latest_run(update.effective_chat.id)
    if latest is None:
        await update.message.reply_text("Nothing to mark yet - I haven't replied to anything here.")
        return
    note = " ".join(context.args or [])
    memory.label(latest.id, "bad", note)
    log.info("label: run %s -> bad (%s)", latest.id, note or "no note")
    await update.message.reply_text(
        "Noted. That reply is marked as wrong"
        + (", with your note" if note else "")
        + "; the judge's grade for it will be checked against that."
    )


async def on_reaction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A thumbs-down or thumbs-up on one of Nexus's replies labels its run;
    taking the reaction back clears the label. Silent either way: a reaction
    is not a message, and answering one would be noise."""
    reaction = update.message_reaction
    if reaction is None or not _is_allowed(update):
        return
    memory = context.bot_data["memory"]
    found = memory.run_for_message(reaction.chat.id, reaction.message_id)
    if found is None:
        return  # not one of our replies, or from before the record began
    emojis = [item.emoji for item in reaction.new_reaction if item.type == ReactionType.EMOJI]
    label = next((LABELS[emoji] for emoji in emojis if emoji in LABELS), None)
    memory.label(found.id, label)
    log.info("label: run %s -> %s", found.id, label or "cleared")


async def on_judge(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Grade the recent replies now, and say what the judge did."""
    if _is_allowed(update):
        line = await context.bot_data["judge"].nightly(trigger="/judge")
        await update.message.reply_text(f"Judge: {line}.")


async def on_task_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Done, Tomorrow or Drop on a nudge. The tap acts through the agent's own
    tools, the nudge is edited with the outcome, and that task's row of buttons
    goes away; a failure leaves the buttons for another try."""
    query = update.callback_query
    if query is None:
        return
    if not _is_allowed(update):
        await query.answer()
        return
    _, task_id, action = query.data.split(":")
    chat_id = query.message.chat.id
    result = await asyncio.to_thread(
        buttons.apply, context.bot_data["memory"], chat_id, task_id, action
    )
    log.info("button: %s on %s -> %s", action, task_id, result.line)
    await query.answer(result.line[:200])
    if not result.ok:
        return
    markup = query.message.reply_markup
    rows = [
        row
        for row in (markup.inline_keyboard if markup else [])
        if not any(button.callback_data.startswith(f"task:{task_id}:") for button in row)
    ]
    await query.edit_message_text(
        f"{query.message.text}\n\n{result.line}",
        reply_markup=InlineKeyboardMarkup(rows) if rows else None,
    )


async def on_review(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run the weekly review now. The review itself arrives as its own message
    (through the proactive gate, buttons and all); this reply says what it did."""
    if _is_allowed(update):
        line = await context.bot_data["reviewer"].weekly(trigger="/review")
        await update.message.reply_text(f"Review: {line}.")


async def on_decision(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A button on the reviewer's ask: Yes, No, or Show me. Yes and No settle
    the suggestion and replace the ask with the outcome; Show me answers with
    the evidence and leaves the buttons in place."""
    query = update.callback_query
    if query is None:
        return
    if not _is_allowed(update):
        await query.answer()
        return
    _, suggestion_id, action = query.data.split(":")
    reviewer = context.bot_data["reviewer"]
    if action == "show":
        await query.answer()
        await query.message.reply_text(reviewer.evidence(suggestion_id))
        return
    # The decision may file a GitHub issue; keep that off the event loop.
    text = await asyncio.to_thread(reviewer.decide, suggestion_id, action)
    await query.answer()
    await query.edit_message_text(text)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """python-telegram-bot retries network trouble by itself -- a laptop going
    to sleep, Wi-Fi dropping -- so one line is enough for the log. Anything
    else is a bug and gets its traceback."""
    if isinstance(context.error, NetworkError):
        log.warning("Telegram unreachable (%s); retrying", context.error)
    else:
        log.error("Unhandled error in the bot", exc_info=context.error)


def _sender(app):
    """How proactive messages leave: the same bot, addressed by chat id. A
    message that asks something carries one row of buttons."""

    async def send(chat_id: int, text: str, rows=None) -> None:
        markup = None
        if rows:
            markup = InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton(label, callback_data=data) for label, data in row]
                    for row in rows
                ]
            )
        await app.bot.send_message(chat_id=chat_id, text=text, reply_markup=markup)

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
    app.add_handler(CommandHandler("nudge", on_nudge))
    app.add_handler(CommandHandler("status", on_status))
    app.add_handler(CommandHandler("bad", on_bad))
    app.add_handler(CommandHandler("judge", on_judge))
    app.add_handler(CommandHandler("review", on_review))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_handler(MessageReactionHandler(on_reaction))
    app.add_handler(CallbackQueryHandler(on_decision, pattern=r"^sug:"))
    app.add_handler(CallbackQueryHandler(on_task_button, pattern=r"^task:"))
    app.add_error_handler(on_error)

    judge = Judge(memory)
    app.bot_data["judge"] = judge
    schedule_nightly(app, judge)
    log.info("Judge: %s, nightly at %s", config.JUDGE_MODEL, f"{config.JUDGE_TIME:%H:%M}")

    log.info("Timezone: %s", config.TIMEZONE)
    proactive = Proactive(
        send=_sender(app),
        chat_id=config.TELEGRAM_CHAT_ID,
        quiet=Window(*config.QUIET_HOURS),
        memory=memory,
    )
    app.bot_data["proactive"] = proactive
    reviewer = Reviewer(memory, proactive)
    app.bot_data["reviewer"] = reviewer
    schedule_weekly(app, reviewer)
    log.info(
        "Reviewer: %s, weekly on day %d at %s; approved suggestions go to %s",
        config.REVIEW_MODEL,
        config.REVIEW_DAY,
        f"{config.REVIEW_TIME:%H:%M}",
        f"GitHub issues on {config.GITHUB_REPO}"
        if config.GITHUB_TOKEN and config.GITHUB_REPO
        else "you, as a brief to paste into Claude Code",
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
    # Reactions are not delivered unless asked for by name; the buttons need
    # callback queries.
    app.run_polling(
        allowed_updates=[Update.MESSAGE, Update.MESSAGE_REACTION, Update.CALLBACK_QUERY]
    )


if __name__ == "__main__":
    main()

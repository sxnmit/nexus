"""Configuration for Nexus, read once from the environment (and .env).

Nothing here raises on import, so every module stays trivially importable in
tests. `missing_settings()` is called by bot.py at startup so a missing token
fails loudly and immediately instead of halfway through a conversation.
"""

import os
from datetime import datetime, time
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

# --- Secrets. These names must match the ones in .env -------------------------
ANTHROPIC_API_KEY = os.getenv("PERSONAL_ACCESS_TOKEN_CLAUDE", "")
TODOIST_API_TOKEN = os.getenv("API_TOKEN_TODOIST", "")
TELEGRAM_BOT_TOKEN = os.getenv("API_TOKEN_TELEGRAM", "")

# Only needed when PERSONAL_ACCESS_TOKEN_CLAUDE is an org-level personal access
# token: Anthropic then insists on knowing which workspace it acts in (Console
# -> Settings -> Workspaces; ids start with wrkspc_). A key created inside a
# workspace already knows and can leave this empty.
ANTHROPIC_WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID", "")

_REQUIRED = {
    "PERSONAL_ACCESS_TOKEN_CLAUDE": ANTHROPIC_API_KEY,
    "API_TOKEN_TODOIST": TODOIST_API_TOKEN,
    "API_TOKEN_TELEGRAM": TELEGRAM_BOT_TOKEN,
}

# --- Tunables -----------------------------------------------------------------
# Haiku: this is a routing job, not a reasoning one, and it is fast and cheap.
MODEL = os.getenv("NEXUS_MODEL", "claude-haiku-4-5")

# A ceiling on one response, not a spend: unused tokens cost nothing, and it
# cannot stop the model hallucinating -- it bounds the damage. Replies are one
# or two sentences, and Telegram rejects messages over 4096 characters (about
# 1000 English tokens), so nothing longer could be delivered anyway. Worst-case
# spend per message is MAX_TOOL_LOOPS x MAX_TOKENS. Raise this to ~8192 if you
# point NEXUS_MODEL at a thinking model (claude-opus-5, claude-sonnet-5): its
# thinking tokens count against the same ceiling.
MAX_TOKENS = int(os.getenv("NEXUS_MAX_TOKENS", "1024"))

# Todoist's current unified API. The older REST v2 base
# (https://api.todoist.com/rest/v2) uses identical paths for the three endpoints
# we call, and todoist.get_tasks() understands both response shapes, so pointing
# this at v2 is the only change needed if you ever have to fall back.
TODOIST_API_BASE = os.getenv("TODOIST_API_BASE", "https://api.todoist.com/api/v1").rstrip("/")

# How many times the agent may go round the plan -> act -> observe loop for a
# single message. Stops a confused model from looping forever on your token bill.
MAX_TOOL_LOOPS = int(os.getenv("NEXUS_MAX_TOOL_LOOPS", "6"))

# After a failed tool call the observe step may grant a retry -- this many per
# message. One is deliberate: a second identical failure is information, not
# bad luck, and the user should hear about it.
MAX_RETRIES = int(os.getenv("NEXUS_MAX_RETRIES", "1"))

# Pause before retrying a rate limit or an outage, so the retry is not just a
# faster way to hit the same wall. The tests set this to 0.
RETRY_BACKOFF_SECONDS = float(os.getenv("NEXUS_RETRY_BACKOFF_SECONDS", "2"))


def _parse_user_ids(raw: str) -> set[int]:
    """Parse "123, 456" into {123, 456}. Ignores blanks and non-numeric entries."""
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if part.lstrip("-").isdigit():
            ids.add(int(part))
    return ids


# Optional comma-separated Telegram user IDs. Empty means anyone who finds the
# bot can edit your Todoist -- set this once you know your own ID.
ALLOWED_TELEGRAM_USER_IDS = _parse_user_ids(os.getenv("TELEGRAM_ALLOWED_USER_IDS", ""))


# --- Proactive messages ---------------------------------------------------------------
# A setting that cannot be parsed is recorded here and the default is used, so
# importing config never raises; bot.py reports config_errors() at startup and
# refuses to run, which is where the mistake is cheapest to fix.
_ERRORS: list[str] = []


def _setting(var: str, default: str, parse):
    raw = os.getenv(var, default)
    try:
        return parse(raw)
    except (ValueError, KeyError) as exc:  # ZoneInfoNotFoundError is a KeyError
        _ERRORS.append(f"{var}={raw!r}: {exc}")
        return parse(default)


def _parse_timezone(name: str):
    """Unset means the machine's local zone -- right on a laptop, but a container
    is usually UTC, so set NEXUS_TIMEZONE on a host or "8am" is 8am UTC."""
    return ZoneInfo(name) if name else datetime.now().astimezone().tzinfo


def _parse_clock(raw: str) -> time:
    hour, minute = raw.strip().split(":")
    return time(int(hour), int(minute))


def _parse_window(raw: str) -> tuple[time, time]:
    start, end = raw.split("-")
    return _parse_clock(start), _parse_clock(end)


def _parse_chat_id(raw: str) -> int | None:
    return int(raw) if raw.strip() else None


TIMEZONE = _setting("NEXUS_TIMEZONE", "", _parse_timezone)
MORNING_TIME = _setting("NEXUS_MORNING_TIME", "08:00", _parse_clock)
EVENING_TIME = _setting("NEXUS_EVENING_TIME", "21:00", _parse_clock)
QUIET_HOURS = _setting("NEXUS_QUIET_HOURS", "22:00-07:00", _parse_window)
OVERDUE_CHECK_MINUTES = _setting("NEXUS_OVERDUE_CHECK_MINUTES", "15", int)


def _chat_id(explicit: int | None, allowed: set[int]) -> int | None:
    """Where proactive messages go. An explicit id wins; otherwise, in a private
    chat with the bot your user id is the chat id, so a one-person allowlist is
    the obvious answer. Anything else means off."""
    if explicit is not None:
        return explicit
    return next(iter(allowed)) if len(allowed) == 1 else None


TELEGRAM_CHAT_ID = _chat_id(
    _setting("TELEGRAM_CHAT_ID", "", _parse_chat_id), ALLOWED_TELEGRAM_USER_IDS
)


def config_errors() -> list[str]:
    """Settings that could not be parsed (the default was used in their place)."""
    return list(_ERRORS)


def missing_settings() -> list[str]:
    """Names of the required environment variables that are unset or empty."""
    return [name for name, value in _REQUIRED.items() if not value]

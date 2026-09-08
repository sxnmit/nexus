"""Configuration for Nexus, read once from the environment (and .env).

Nothing here raises on import, so every module stays trivially importable in
tests. `missing_settings()` is called by bot.py at startup so a missing token
fails loudly and immediately instead of halfway through a conversation.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Secrets. These names must match the ones in .env -------------------------
ANTHROPIC_API_KEY = os.getenv("PERSONAL_ACCESS_TOKEN_CLAUDE", "")
TODOIST_API_TOKEN = os.getenv("API_TOKEN_TODOIST", "")
TELEGRAM_BOT_TOKEN = os.getenv("API_TOKEN_TELEGRAM", "")

_REQUIRED = {
    "PERSONAL_ACCESS_TOKEN_CLAUDE": ANTHROPIC_API_KEY,
    "API_TOKEN_TODOIST": TODOIST_API_TOKEN,
    "API_TOKEN_TELEGRAM": TELEGRAM_BOT_TOKEN,
}

# --- Tunables -----------------------------------------------------------------
MODEL = os.getenv("NEXUS_MODEL", "claude-opus-5")

# Generous, because on Opus 5 thinking tokens count against max_tokens. Replies
# themselves are one or two sentences.
MAX_TOKENS = 8192

# Todoist's current unified API. The older REST v2 base
# (https://api.todoist.com/rest/v2) uses identical paths for the three endpoints
# we call, and todoist.get_tasks() understands both response shapes, so pointing
# this at v2 is the only change needed if you ever have to fall back.
TODOIST_API_BASE = os.getenv("TODOIST_API_BASE", "https://api.todoist.com/api/v1").rstrip("/")

# How many times the agent may go round the plan -> act -> observe loop for a
# single message. Stops a confused model from looping forever on your token bill.
MAX_TOOL_LOOPS = int(os.getenv("NEXUS_MAX_TOOL_LOOPS", "6"))


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


def missing_settings() -> list[str]:
    """Names of the required environment variables that are unset or empty."""
    return [name for name, value in _REQUIRED.items() if not value]

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

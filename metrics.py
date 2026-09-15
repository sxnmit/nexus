"""The scorecard: what the record says about the last day or week, by code alone.

No model reads anything here. Every number is a count or a sum over the
`runs` table (memory.py), which makes this the honest half of evaluation: it
cannot be charmed by a well-phrased reply, and it cannot see whether a reply
was *right*. What it can see is the shape of trouble -- runs that got stuck or
crashed, tools that failed and with which status, retries spent, clarifying
questions asked, and what the day cost -- plus one soft signal, the
correction: a follow-up within a few minutes that starts with "no", "undo" or
"I meant", or repeats the request. A correction is a guess (it is a heuristic
over the user's words); the judge, when it comes, confirms or clears it.

Two things the scorecard keeps apart on purpose:

  * Replies from check-ins. A check-in is a template; only a reply involves
    the model. Their outcomes are counted separately (`replies`, `jobs`).
  * The agent's failures from the world's. A tool error carries its HTTP
    status, so "Todoist returned 503 twice" is not "the agent failed twice".

`/status` in Telegram prints it; the reviewer, when it comes, will cite it.
"""

import re
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise

from memory import Memory, Run, words

# A follow-up this soon after a reply may be a correction of it ...
CORRECTION_MINUTES = 5
# ... if it opens like one, or repeats this share of the request's words.
CORRECTION_OPENERS = ("no", "nope", "not that", "wrong", "undo", "i meant", "i said", "that's not")
CORRECTION_OVERLAP = 0.6

# The outcomes a reply can have, in the order the scorecard lists them.
REPLY_OUTCOMES = ("ok", "asked", "failed", "stuck", "crashed")


# --- Corrections --------------------------------------------------------------


_OPENS_LIKE_A_CORRECTION = re.compile(
    r"^(?:" + "|".join(re.escape(opener) for opener in CORRECTION_OPENERS) + r")\b"
)


def _opens_like_a_correction(text: str) -> bool:
    return _OPENS_LIKE_A_CORRECTION.match(text.strip().lower()) is not None


def _repeats(earlier: str, later: str) -> bool:
    """The later message re-uses most of the earlier one's identifying words."""
    before, after = set(words(earlier)), set(words(later))
    return bool(before) and len(before & after) / len(before | after) >= CORRECTION_OVERLAP


def is_correction(reply: Run, follow_up: Run) -> bool:
    """Does `follow_up` look like the user correcting `reply`?

    Not when the reply was a question: an answer arrives fast and repeats
    the request, and that is the conversation working, not failing.
    """
    if follow_up.chat_id != reply.chat_id or reply.outcome == "asked":
        return False
    if follow_up.ts - reply.ts > timedelta(minutes=CORRECTION_MINUTES):
        return False
    return _opens_like_a_correction(follow_up.trigger) or _repeats(reply.trigger, follow_up.trigger)


# --- The card ----------------------------------------------------------------


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def _seconds(ms: float) -> str:
    return f"{ms / 1000:.1f}s"


@dataclass(frozen=True)
class Scorecard:
    """The window's numbers. `text()` is what /status prints."""

    since: datetime
    until: datetime
    replies: Counter = field(default_factory=Counter)  # outcome -> runs
    jobs: Counter = field(default_factory=Counter)  # (kind, outcome) -> runs
    steps: int = 0
    retries: int = 0
    tool_calls: int = 0
    clarifications: int = 0  # tool calls that needed the user
    unexpected: int = 0  # tool calls Todoist accepted but got wrong
    tool_errors: Counter = field(default_factory=Counter)  # (tool, status) -> calls
    corrections: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    latencies_ms: tuple[int, ...] = ()  # one per reply
    model: str = ""
    prompt: str = ""
    build: str = ""

    @property
    def reply_count(self) -> int:
        return sum(self.replies.values())

    def text(self) -> str:
        days = round((self.until - self.since).total_seconds() / 86400)
        window = "Last 24 hours" if days == 1 else f"Last {days} days"
        lines = [f"{window}: {self._replies_line()}"]
        if self.reply_count:
            lines.append("Outcomes: " + self._outcomes() + ".")
            lines.append(
                f"Loop: {_plural(self.steps, 'step')}, "
                f"{_plural(self.retries, 'retry', 'retries')}, "
                f"{_plural(self.clarifications, 'clarification')} asked for, "
                f"{_plural(self.unexpected, 'unexpected result')}."
            )
            if self.tool_errors:
                lines.append("Tool errors: " + self._tool_errors() + ".")
            lines.append(f"Corrections within {CORRECTION_MINUTES} min: {self.corrections}.")
            lines.append(
                f"Tokens: {self.input_tokens:,} in ({self.cache_read_tokens:,} from cache), "
                f"{self.output_tokens:,} out. Replies took "
                f"{_seconds(statistics.median(self.latencies_ms))} typically, "
                f"{_seconds(max(self.latencies_ms))} at worst."
            )
        if self.jobs:
            lines.append("Check-ins: " + self._jobs() + ".")
        lines.append(
            f"Build: {self.model or 'model unknown'}, prompt {self.prompt or 'unknown'}, "
            f"commit {self.build or 'unknown'}."
        )
        return "\n".join(lines)

    def _replies_line(self) -> str:
        if not self.reply_count:
            return "no replies."
        return _plural(self.reply_count, "reply", "replies") + "."

    def _outcomes(self) -> str:
        return ", ".join(
            f"{self.replies[outcome]} {outcome}"
            for outcome in REPLY_OUTCOMES
            if self.replies[outcome]
        )

    def _tool_errors(self) -> str:
        return ", ".join(
            f"{tool} {status} x{count}"
            for (tool, status), count in sorted(self.tool_errors.items())
        )

    def _jobs(self) -> str:
        by_kind: dict[str, list[str]] = {}
        for (kind, outcome), count in sorted(self.jobs.items()):
            by_kind.setdefault(kind, []).append(f"{outcome} x{count}")
        return "; ".join(f"{kind} {', '.join(parts)}" for kind, parts in by_kind.items())


# --- Computing it -------------------------------------------------------------


def _observations(run: Run) -> Iterable[dict]:
    return (step for step in run.trace.get("steps", []) if step.get("role") == "tool")


def _count_corrections(replies: list[Run]) -> int:
    by_chat: dict[int, list[Run]] = {}
    for run in replies:
        by_chat.setdefault(run.chat_id, []).append(run)
    return sum(
        is_correction(reply, follow_up)
        for chain in by_chat.values()
        for reply, follow_up in pairwise(chain)
    )


def scorecard(memory: Memory, days: int = 1, now: datetime | None = None) -> Scorecard:
    """The scorecard for the last `days` days, ending now."""
    until = now or memory.now()
    since = until - timedelta(days=days)
    runs = memory.runs(since, until)
    replies = [run for run in runs if run.kind == "message"]

    clarifications = unexpected = 0
    tool_errors: Counter = Counter()
    for run in replies:
        for observation in _observations(run):
            outcome = observation.get("outcome")
            if outcome == "clarify":
                clarifications += 1
            elif outcome == "unexpected":
                unexpected += 1
            elif outcome in ("api_error", "bad_args", "crash"):
                status = observation.get("status_code") or outcome
                tool_errors[(observation.get("name"), str(status))] += 1

    latest = replies[-1] if replies else None
    return Scorecard(
        since=since,
        until=until,
        replies=Counter(run.outcome for run in replies),
        jobs=Counter((run.kind, run.outcome) for run in runs if run.kind != "message"),
        steps=sum(run.steps for run in replies),
        retries=sum(run.retries for run in replies),
        tool_calls=sum(run.tool_calls for run in replies),
        clarifications=clarifications,
        unexpected=unexpected,
        tool_errors=tool_errors,
        corrections=_count_corrections(replies),
        input_tokens=sum(run.input_tokens for run in replies),
        output_tokens=sum(run.output_tokens for run in replies),
        cache_read_tokens=sum(run.cache_read_tokens for run in replies),
        latencies_ms=tuple(run.latency_ms for run in replies),
        model=latest.model if latest else "",
        prompt=latest.prompt if latest else "",
        build=latest.build if latest else "",
    )

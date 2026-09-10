"""Proactive messages: time-triggered, never inferred.

Everything the bot says without being asked comes through here, under four
rules that are deliberately few:

  1. Only the clock or a state transition triggers a message -- never a guess
     about what the user might want. The morning and evening jobs fire because
     a configured time passed; a nudge fires because a task crossed from due to
     overdue.
  2. A transition is reported once. Every job records what it has reported --
     per task *and* per due date, so a task that was pushed to a new time is
     a new transition. The morning check-in may list an overdue task again
     tomorrow, but that is the day's anchor, not a repeat of the same event.
  3. Silence when there is nothing to say -- except the morning check-in, which
     is the anchor and always sends.
  4. Quiet hours are absolute for proactive messages and irrelevant for
     replies. A nudge that lands in quiet hours is not dropped: it is not marked
     as sent, so the first check after the window opens sends it.

The messages are templates, not model output. They are reports; the agent loop
is for requests; and a template cannot hallucinate a task. Your reply to a
check-in goes through the normal agent with all six tools -- and with the
check-in in its conversation memory, so "push it to tomorrow" just works.

Memory adds one thing to each report: a sentence of advice for a task the
counters say keeps slipping or usually runs late ("drop it, or give it a fixed
slot?"). That is a template too, filled from integers, so it stays honest. And
the morning snapshot and the record of what has been reported live in the same
database, so a restart between the jobs loses neither.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import config
import todoist
from memory import Memory

log = logging.getLogger("nexus.proactive")

# A task that went overdue longer ago than this is the morning check-in's job,
# not the nudge's -- and a restart must not re-nudge last week.
NUDGE_LOOKBACK = timedelta(hours=24)

# Telegram rejects messages over 4096 characters.
MAX_MESSAGE = 4000

# How late a job may still run. APScheduler's default is one second: a laptop
# that was asleep at 08:00 would skip the whole check-in, and a six-second
# stall would skip an overdue check. A check-in within the hour is still that
# morning's; an overdue check is repeated anyway, so a few minutes is enough.
DAILY_GRACE = timedelta(hours=1)
NUDGE_GRACE = timedelta(minutes=5)

Sender = Callable[[int, str], Awaitable[None]]


# --- Quiet hours ----------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """A daily quiet window. It may cross midnight: 22:00-07:00 is the default."""

    start: time
    end: time

    def is_quiet(self, now: datetime) -> bool:
        wall = now.time().replace(tzinfo=None)
        if self.start == self.end:
            return False  # a zero-length window means no quiet hours
        if self.start < self.end:
            return self.start <= wall < self.end  # e.g. 13:00-14:00
        return wall >= self.start or wall < self.end  # crosses midnight


# --- Sorting tasks by day -------------------------------------------------------------


def _split(tasks: list[dict], today: date) -> tuple[list[dict], list[dict]]:
    """Open tasks due today, and open tasks due before today."""
    due_today, overdue = [], []
    for task in tasks:
        day, _ = todoist.parse_due(task)
        if day == today:
            due_today.append(task)
        elif day is not None and day < today:
            overdue.append(task)
    return due_today, overdue


def _task_id(task: dict) -> str:
    return str(task.get("id"))


# --- Templates ---------------------------------------------------------------------


def _clock(when: datetime) -> str:
    hour = when.hour % 12 or 12
    return f"{hour}:{when.minute:02d}{'am' if when.hour < 12 else 'pm'}"


def _ago(delta: timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if minutes else f"{hours}h"


def _line(task: dict, today: date) -> str:
    """One human line per task: name, then time / how overdue / priority."""
    day, when = todoist.parse_due(task)
    bits = []
    if when is not None and day == today:
        bits.append(_clock(when))
    elif day is not None and day < today:
        behind = (today - day).days
        bits.append("yesterday" if behind == 1 else f"{behind} days overdue")
    api_priority = task.get("priority") or 1
    if api_priority > 1:
        bits.append(f"p{5 - api_priority}")
    suffix = f" ({', '.join(bits)})" if bits else ""
    return f"- {task.get('content')}{suffix}"


def _listing(title: str, tasks: list[dict], today: date) -> str:
    return f"{title} ({len(tasks)}):\n" + "\n".join(_line(task, today) for task in tasks)


def morning_text(due_today: list[dict], overdue: list[dict], today: date) -> str:
    if not due_today and not overdue:
        return "Good morning. Nothing due today and nothing overdue."
    parts = ["Good morning."]
    if due_today:
        parts.append(_listing("Due today", due_today, today))
    if overdue:
        parts.append(_listing("Overdue", overdue, today))
    parts.append("Want to move or reprioritise anything? Say which.")
    return "\n\n".join(parts)


def evening_text(
    done: list[dict], still_open: list[dict], today: date, started_at: datetime | None
) -> str:
    parts = ["Evening review."]
    if started_at is not None:
        parts.append(
            f"I started at {_clock(started_at)}, so I can't see what got done before "
            f"that -- only what is still open."
        )
    if done:
        parts.append(_listing("Done today", done, today))
    if still_open:
        parts.append(_listing("Still open", still_open, today))
        parts.append("Push any of these to tomorrow? Say which.")
    else:
        parts.append("Nothing left open for today.")
    return "\n\n".join(parts)


def nudge_text(fresh: list[tuple[datetime, dict]], now: datetime) -> str:
    if len(fresh) == 1:
        when, task = fresh[0]
        return (
            f"Overdue: '{task.get('content')}' was due {_clock(when)} "
            f"({_ago(now - when)} ago). Done, or push it?"
        )
    lines = [f"- {task.get('content')} (due {_clock(when)})" for when, task in fresh]
    return f"Overdue ({len(fresh)}):\n" + "\n".join(lines) + "\n\nDone, or push them?"


def with_advice(text: str, advice: list[str]) -> str:
    """A report, then the habits worth a word -- as a paragraph of its own."""
    return text if not advice else text + "\n\n" + "\n".join(advice)


# --- The gate and the three jobs --------------------------------------------------


class Proactive:
    """Everything the bot sends unprompted. `deliver` is the one gate."""

    def __init__(
        self,
        send: Sender,
        chat_id: int | None,
        quiet: Window,
        memory: Memory,
        clock: Callable[[], datetime] | None = None,
    ):
        self.send = send
        self.chat_id = chat_id
        self.quiet = quiet
        self.memory = memory
        self._clock = clock  # tests inject a fake clock
        self.started_at = self.now()

        # What the morning check-in saw, and what any job has already reported
        # (task id -> the due date it was reported with). Both come back from
        # the database, so a restart mid-day changes nothing.
        saved = memory.load("morning_snapshot")
        self.snapshot: dict[str, dict] | None = saved["tasks"] if saved else None
        self.snapshot_at: datetime | None = (
            datetime.fromisoformat(saved["at"]).astimezone(config.TIMEZONE) if saved else None
        )
        self.reported: dict[str, str] = memory.load("reported", {})

    @property
    def enabled(self) -> bool:
        return self.chat_id is not None

    def now(self) -> datetime:
        current = self._clock() if self._clock else datetime.now(config.TIMEZONE)
        return current.astimezone(config.TIMEZONE)

    async def deliver(self, kind: str, text: str) -> bool:
        """The gate: recipient, quiet hours, size, logging. Returns whether it went."""
        if not self.enabled:
            log.info("%s: not sent - no TELEGRAM_CHAT_ID", kind)
            return False
        now = self.now()
        if self.quiet.is_quiet(now):
            log.info("%s: held - quiet hours (%s)", kind, now.strftime("%H:%M"))
            return False
        sent = text[:MAX_MESSAGE]
        await self.send(self.chat_id, sent)
        # Part of the conversation now: the user's next message may refer to it.
        self.memory.log(self.chat_id, "assistant", sent, kind=kind)
        log.info("%s: sent", kind)
        return True

    def _is_reported(self, task: dict) -> bool:
        return self.reported.get(_task_id(task)) == todoist.due_key(task)

    def _mark_reported(self, tasks: list[dict]) -> None:
        for task in tasks:
            self.reported[_task_id(task)] = todoist.due_key(task)
        self.memory.save("reported", self.reported)

    async def _open_tasks(self) -> list[dict]:
        # The Todoist client is synchronous; keep it off the event loop.
        return await asyncio.to_thread(todoist.get_tasks)

    async def morning_checkin(self) -> None:
        """The day's anchor: what is due, what is overdue, and an offer to reshuffle."""
        now = self.now()
        today = now.date()
        try:
            tasks = await self._open_tasks()
        except todoist.TodoistError as exc:
            await self.deliver("morning", f"Morning check-in: I couldn't reach Todoist ({exc}).")
            return

        due_today, overdue = _split(tasks, today)
        self.snapshot = {_task_id(task): task for task in due_today + overdue}
        self.snapshot_at = now
        self.memory.save("morning_snapshot", {"at": now.isoformat(), "tasks": self.snapshot})

        # Rule 2: these are reported now, so the nudge job must not repeat them.
        # And a task no longer open has nothing left to report; forget it.
        open_ids = {_task_id(task) for task in tasks}
        self.reported = {tid: due for tid, due in self.reported.items() if tid in open_ids}
        self._mark_reported(due_today + overdue)

        text = morning_text(due_today, overdue, today)
        await self.deliver(
            "morning", with_advice(text, self.memory.advice_for(due_today + overdue))
        )

    async def evening_review(self) -> None:
        """Done vs still open, diffed against the morning snapshot."""
        now = self.now()
        today = now.date()
        try:
            tasks = await self._open_tasks()
        except todoist.TodoistError as exc:
            await self.deliver("evening", f"Evening review: I couldn't reach Todoist ({exc}).")
            return

        open_ids = {_task_id(task) for task in tasks}
        due_today, overdue = _split(tasks, today)
        still_open = due_today + overdue

        have_snapshot = self.snapshot is not None and self.snapshot_at.date() == today
        if have_snapshot:
            done = [task for tid, task in self.snapshot.items() if tid not in open_ids]
            started_at = None
        else:
            done, started_at = [], self.started_at

        if not done and not still_open:
            log.info("evening: nothing to review")  # rule 3
            return

        self._mark_reported(still_open)
        text = evening_text(done, still_open, today, started_at)
        await self.deliver("evening", with_advice(text, self.memory.advice_for(still_open)))

    async def overdue_nudge(self) -> None:
        """Timed tasks that just went overdue, each reported once."""
        now = self.now()
        try:
            tasks = await self._open_tasks()
        except todoist.TodoistError as exc:
            log.warning("nudge: Todoist unavailable (%s)", exc)  # every 15 min: log, don't message
            return

        fresh = []
        for task in tasks:
            _, when = todoist.parse_due(task)
            if when is None or self._is_reported(task):
                continue
            if when <= now <= when + NUDGE_LOOKBACK:
                fresh.append((when, task))
        if not fresh:
            return

        fresh.sort(key=lambda pair: pair[0])
        overdue = [task for _, task in fresh]
        text = with_advice(nudge_text(fresh, now), self.memory.advice_for(overdue))
        if await self.deliver("nudge", text):
            # Rule 4: only what was actually sent counts as reported. Held in
            # quiet hours -> tried again next check.
            self._mark_reported(overdue)


# --- Wiring into the bot -----------------------------------------------------------


def schedule_jobs(app, proactive: Proactive) -> None:
    """Register the three jobs on the bot's JobQueue (APScheduler underneath).

    The times carry NEXUS_TIMEZONE. A naive time would mean UTC to the
    JobQueue, which on a host is how "8am" quietly becomes 4am. The grace
    periods let a job that was missed -- the machine was asleep, the loop was
    busy -- still run once, late, instead of being dropped.
    """
    tz = config.TIMEZONE

    async def morning(context) -> None:
        await proactive.morning_checkin()

    async def evening(context) -> None:
        await proactive.evening_review()

    async def nudge(context) -> None:
        await proactive.overdue_nudge()

    queue = app.job_queue
    daily = {"misfire_grace_time": int(DAILY_GRACE.total_seconds())}
    queue.run_daily(
        morning, time=config.MORNING_TIME.replace(tzinfo=tz), name="morning", job_kwargs=daily
    )
    queue.run_daily(
        evening, time=config.EVENING_TIME.replace(tzinfo=tz), name="evening", job_kwargs=daily
    )
    queue.run_repeating(
        nudge,
        interval=timedelta(minutes=config.OVERDUE_CHECK_MINUTES),
        first=timedelta(minutes=1),
        name="overdue",
        job_kwargs={"misfire_grace_time": int(NUDGE_GRACE.total_seconds())},
    )

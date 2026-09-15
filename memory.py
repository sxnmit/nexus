"""Lightweight memory: an interaction log and a few learned habits, in SQLite.

Five tables and no embeddings.

`interactions`  Everything said and done, in order: each user message, each
                tool call with its arguments and outcome, each reply, and each
                check-in the scheduler sent. Its last few rows are the
                conversation memory. Before a run, `recent()` hands the agent
                the last N messages of the chat, which is what makes "push it
                to friday" mean something the morning after a check-in. Each
                row names the run that produced it.

`runs`          One row per unit of work -- a message answered, a check-in
                sent or withheld -- and how it went: the outcome, the steps
                and retries, what the model calls cost, how long it took, and
                a trace of what the model saw and did. The prompt version and
                the build are on every row, so a change to either shows up in
                what followed. The user's own verdict on a reply -- a thumbs
                down, a /bad -- is a label on the row. This is the record the
                judge and the scorecard read.

`evaluations`   The judge's grades (judge.py), one row per graded run: pass
                or fail per rubric property with a one-line reason, a failure
                category, a summary, and what the grading cost. Kept apart
                from `runs` because grades come later, from another model,
                under a rubric that has its own version.

`patterns`      One row per *topic* -- a task name with the noise stripped, so
                "Go to the gym tomorrow" and "gym" share a row -- holding
                counters the tool outcomes update: created, rescheduled (and
                how many of those were pushed later), completed (and how many
                were late, by how much in total), deleted. Nothing is inferred.
                Every field counts something that happened through Nexus.

`state`         A key/value shelf for the scheduler: the morning snapshot and
                what it has already reported, so a restart forgets neither.

The tools attach a structured event to each successful call (see tools.py) and
`learn()` turns those into counter updates. Two things read the counters:
`note()` renders the topics a message mentions as a short block for the
planning step, and `advice_for()` gives the proactive templates one sentence
per task worth commenting on.

Why counters and not a model: memory here exists to *change a suggestion*, not
to narrate. "You've moved gym four times; want a fixed slot?" needs four
integers, and four integers cannot hallucinate. Where counters fall short is
grouping -- "gym", "workout" and "go running" are three topics -- and that is
exactly the gap embeddings-based retrieval would fill. Flagged, not built.

One connection, one lock: the Telegram handler runs the agent on worker
threads while the scheduler's jobs run on the event loop, and a SQLite
connection must not be used from two threads at once. Writes are tiny, so the
lock costs nothing.
"""

import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime, timedelta

import config
import todoist

# A topic is "recurring" once it has been added this many times ...
RECURRING_AT = 3
# ... and "keeps slipping" once it has been moved this many times, mostly to later.
SLIPS_AT = 2
# How many topics a note for the model may carry.
NOTE_LIMIT = 5

# A word: letters or digits in any script, with an apostrophe allowed inside ("mum's").
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")
# Words that carry no identity: articles, prepositions, request verbs, and
# anything that names a time rather than a task.
_NOISE_WORDS = (
    "a an the to for of my me i we our your at on in and with up out about this that it "
    "is be do go get from by some please need add task remind reminder "
    "today tomorrow tonight morning afternoon evening night next last every am pm "
    "monday tuesday wednesday thursday friday saturday sunday mon tue wed thu fri sat sun "
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sep sept oct nov dec"
)
_NOISE = frozenset(_NOISE_WORDS.split())

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id      INTEGER PRIMARY KEY,
    ts      TEXT    NOT NULL,  -- UTC, ISO 8601
    chat_id INTEGER NOT NULL,
    role    TEXT    NOT NULL,  -- user | assistant | tool
    kind    TEXT    NOT NULL,  -- message | reply | morning | evening | nudge | a tool name
    text    TEXT    NOT NULL,
    detail  TEXT,              -- JSON; for a tool call, its arguments and outcome
    run_id  TEXT               -- the run this row belongs to (runs.id)
);
CREATE INDEX IF NOT EXISTS interactions_by_chat ON interactions (chat_id, ts);

CREATE TABLE IF NOT EXISTS runs (
    id                TEXT    PRIMARY KEY,   -- short random id; interaction rows carry it
    ts                TEXT    NOT NULL,      -- UTC, ISO 8601: when the run started
    chat_id           INTEGER NOT NULL,
    kind              TEXT    NOT NULL,      -- message | morning | evening | overdue
    trigger           TEXT    NOT NULL,      -- the user's message, or what fired the job
    reply             TEXT    NOT NULL,      -- what went back; '' when nothing was sent
    outcome           TEXT    NOT NULL,      -- see Run
    latency_ms        INTEGER NOT NULL,
    steps             INTEGER NOT NULL DEFAULT 0,
    retries           INTEGER NOT NULL DEFAULT 0,
    calls             INTEGER NOT NULL DEFAULT 0,   -- model calls
    tool_calls        INTEGER NOT NULL DEFAULT 0,
    input_tokens      INTEGER NOT NULL DEFAULT 0,
    output_tokens     INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    model             TEXT    NOT NULL DEFAULT '',
    prompt            TEXT    NOT NULL DEFAULT '',  -- agent.PROMPT_VERSION
    build             TEXT    NOT NULL DEFAULT '',  -- the git commit, when known
    error             TEXT    NOT NULL DEFAULT '',
    trace             TEXT    NOT NULL DEFAULT '{}', -- JSON; see agent._trace
    label             TEXT,                          -- the user's verdict: good | bad
    label_note        TEXT,                          -- what they said with it
    message_id        INTEGER                        -- Telegram's id for the reply, for reactions
);
CREATE INDEX IF NOT EXISTS runs_by_time ON runs (ts);

CREATE TABLE IF NOT EXISTS evaluations (
    run_id        TEXT    PRIMARY KEY,           -- runs.id
    ts            TEXT    NOT NULL,              -- UTC: when it was graded
    judge_model   TEXT    NOT NULL,
    rubric        TEXT    NOT NULL,              -- judge.RUBRIC_VERSION
    passed        TEXT    NOT NULL,              -- JSON: property -> true | false
    reasons       TEXT    NOT NULL,              -- JSON: property -> one line
    category      TEXT    NOT NULL,              -- judge.CATEGORIES
    summary       TEXT    NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS patterns (
    topic          TEXT PRIMARY KEY,
    sample         TEXT    NOT NULL,  -- the latest task name behind the topic
    created        INTEGER NOT NULL DEFAULT 0,
    rescheduled    INTEGER NOT NULL DEFAULT 0,
    pushed_later   INTEGER NOT NULL DEFAULT 0,
    completed      INTEGER NOT NULL DEFAULT 0,
    completed_late INTEGER NOT NULL DEFAULT 0,
    late_seconds   INTEGER NOT NULL DEFAULT 0,
    deleted        INTEGER NOT NULL DEFAULT 0,
    first_seen     TEXT    NOT NULL,
    last_seen      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL  -- JSON
);
"""

# Columns added after a table first shipped. CREATE TABLE IF NOT EXISTS leaves
# an existing table alone, so a database from an earlier version gets them here.
_ADDED_COLUMNS = {
    "interactions": [("run_id", "TEXT")],
    "runs": [("label", "TEXT"), ("label_note", "TEXT"), ("message_id", "INTEGER")],
}


# --- Topics -------------------------------------------------------------------------


def words(text: str) -> list[str]:
    """The words that identify a task, lowercased and in order: no articles, no
    request verbs, nothing that names a time."""
    return [w for w in _WORD.findall(text.lower()) if w not in _NOISE and not w[0].isdigit()]


def topic(content: str) -> str:
    """The key a task is remembered under: its first three identifying words."""
    return " ".join(words(content)[:3])


# --- Patterns -----------------------------------------------------------------------


def _times(count: int) -> str:
    return {1: "once", 2: "twice"}.get(count, f"{count} times")


def _about(delta: timedelta) -> str:
    """A duration the way a person says it: 'about 2 hours', 'about 3 days'."""
    minutes = max(1, round(delta.total_seconds() / 60))
    if minutes < 90:
        return f"about {minutes} min"
    hours = round(minutes / 60)
    if hours < 24:
        return f"about {hours} hours"
    days = round(hours / 24)
    return f"about {days} day{'s' if days != 1 else ''}"


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


@dataclass(frozen=True)
class Pattern:
    """One topic's counters. The properties are the only interpretation."""

    topic: str
    sample: str
    created: int = 0
    rescheduled: int = 0
    pushed_later: int = 0
    completed: int = 0
    completed_late: int = 0
    late_seconds: int = 0
    deleted: int = 0

    @property
    def recurring(self) -> bool:
        return self.created >= RECURRING_AT

    @property
    def slips(self) -> bool:
        """Moved at least SLIPS_AT times, and at least half of those to later."""
        return self.rescheduled >= SLIPS_AT and self.pushed_later * 2 >= self.rescheduled

    @property
    def usually_late(self) -> bool:
        """Finished late at least twice, and at least half the time."""
        return self.completed_late >= 2 and self.completed_late * 2 >= self.completed

    @property
    def lateness(self) -> timedelta:
        """Average lateness of the late completions; zero if there were none."""
        if not self.completed_late:
            return timedelta(0)
        return timedelta(seconds=self.late_seconds / self.completed_late)

    @property
    def notable(self) -> bool:
        """Worth a word to the model or the user. Plain counts are not."""
        return self.recurring or self.slips or self.usually_late

    def describe(self) -> str:
        """One line for the model: every non-zero counter, plainly."""
        bits = []
        if self.created:
            bits.append(
                f"added {_times(self.created)}" + (" (recurring)" if self.recurring else "")
            )
        if self.rescheduled:
            bits.append(f"moved {_times(self.rescheduled)} ({self.pushed_later} to later)")
        if self.completed:
            done = f"done {_times(self.completed)}"
            if self.completed_late:
                done += (
                    f", late {_times(self.completed_late)} by {_about(self.lateness)} on average"
                )
            bits.append(done)
        if self.deleted:
            bits.append(f"deleted {_times(self.deleted)}")
        return f"'{self.sample}': " + "; ".join(bits)

    def advice(self, name: str) -> str:
        """One sentence for a template, or nothing. A suggestion, not a nag."""
        if self.slips:
            always = ", always later" if self.pushed_later == self.rescheduled else ""
            return (
                f"'{name}' keeps slipping (moved {_times(self.rescheduled)} before{always}). "
                f"Drop it, or give it a fixed slot?"
            )
        if self.usually_late:
            return (
                f"'{name}' usually gets done {_about(self.lateness)} late. "
                f"Want to date it later up front?"
            )
        return ""


# --- Runs -----------------------------------------------------------------------------


def run_id() -> str:
    """A short random id for a run, minted before the run starts so every row
    it logs can carry it."""
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class Run:
    """One unit of work, as the record keeps it.

    `kind` says what started it: "message" for a Telegram message answered by
    the agent, or a job name ("morning", "evening", "overdue"). `outcome` is
    how it ended, from a small fixed vocabulary so it can be counted:

      message   ok       every tool call succeeded (or none was needed)
                asked    a tool needed the user's help; the reply is a question
                failed   a tool failed and no retry could help
                stuck    the loop hit its step cap without a reply
                crashed  an exception escaped the graph; `error` says which
      job       sent | held (quiet hours) | off (no recipient) | silent
                (nothing to say) | unavailable (Todoist down)

    The counters and token fields are zero for a job: the templates never call
    the model. `trace` is JSON the evaluator reads; for a message run it holds
    what the model saw and every step it took (see agent._trace).

    `label` is the user's own verdict on a reply -- "good" or "bad", from a
    reaction or /bad -- with what they said in `label_note`. `message_id` is
    Telegram's id for the reply, which is how a reaction finds its run.
    """

    id: str
    ts: datetime
    chat_id: int
    kind: str
    trigger: str
    reply: str
    outcome: str
    latency_ms: int
    steps: int = 0
    retries: int = 0
    calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    model: str = ""
    prompt: str = ""
    build: str = ""
    error: str = ""
    trace: dict = field(default_factory=dict)
    label: str | None = None
    label_note: str | None = None
    message_id: int | None = None


@dataclass(frozen=True)
class Evaluation:
    """The judge's grades for one run: each rubric property passed or failed,
    with a one-line reason; a category for what went wrong (or "none"); and a
    one-sentence summary for the reviewer. `judge_model` and `rubric` say who
    graded it and under which rubric, so grades from different rubrics are
    never averaged together by mistake."""

    run_id: str
    ts: datetime
    judge_model: str
    rubric: str
    passed: dict  # property -> bool
    reasons: dict  # property -> str
    category: str
    summary: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0

    @property
    def failed(self) -> list[str]:
        return [name for name, ok in self.passed.items() if not ok]

    @property
    def clean(self) -> bool:
        return not self.failed


# --- What a tool event means --------------------------------------------------------


def _late_by(task: dict, now: datetime) -> int:
    """Seconds past due at completion: 0 for on time, no due date, or a
    day-only task finished on its day."""
    day, when = todoist.parse_due(task)
    if when is not None:
        return max(0, int((now - when).total_seconds()))
    if day is not None:
        return max(0, (now.date() - day).days) * 86400
    return 0


def _is_later(after: dict, before: dict) -> bool:
    a_day, a_when = todoist.parse_due(after)
    b_day, b_when = todoist.parse_due(before)
    if a_day is None or b_day is None:
        return False  # the due date was cleared, not moved
    if a_when is not None and b_when is not None:
        return a_when > b_when
    return a_day > b_day


# --- The store -----------------------------------------------------------------------


class Memory:
    """The store. `path` is a SQLite file, or ":memory:" for one that dies with
    the process."""

    def __init__(self, path: str = ":memory:", clock: Callable[[], datetime] | None = None):
        self.path = path
        self._clock = clock  # tests inject a fake clock
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock, self._db:
            self._db.executescript(_SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """Add the columns a database from an earlier version is missing."""
        for table, columns in _ADDED_COLUMNS.items():
            present = {row["name"] for row in self._db.execute(f"PRAGMA table_info({table})")}
            for name, kind in columns:
                if name not in present:
                    self._db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")

    def now(self) -> datetime:
        current = self._clock() if self._clock else datetime.now(config.TIMEZONE)
        return current.astimezone(config.TIMEZONE)

    @staticmethod
    def _stamp(when: datetime) -> str:
        # UTC, so timestamps compare correctly as strings -- across a DST change too.
        return when.astimezone(UTC).isoformat(timespec="seconds")

    # --- The log ---------------------------------------------------------------------

    def log(
        self,
        chat_id: int,
        role: str,
        text: str,
        kind: str = "message",
        detail: dict | None = None,
        run_id: str | None = None,
    ) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO interactions (ts, chat_id, role, kind, text, detail, run_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self._stamp(self.now()),
                    chat_id,
                    role,
                    kind,
                    text,
                    None if detail is None else json.dumps(detail),
                    run_id,
                ),
            )

    def recent(self, chat_id: int) -> list[tuple[str, str]]:
        """The conversation so far: the last MEMORY_MESSAGES user/assistant
        messages of this chat from the last MEMORY_HOURS, oldest first. Tool
        calls are left out -- the replies already say what was done."""
        since = self._stamp(self.now() - timedelta(hours=config.MEMORY_HOURS))
        with self._lock:
            rows = self._db.execute(
                "SELECT role, text FROM interactions"
                " WHERE chat_id = ? AND role IN ('user', 'assistant') AND ts >= ?"
                " ORDER BY id DESC LIMIT ?",
                (chat_id, since, config.MEMORY_MESSAGES),
            ).fetchall()
        return [(row["role"], row["text"]) for row in reversed(rows)]

    def counts(self) -> tuple[int, int]:
        """(interactions logged, topics tracked)."""
        with self._lock:
            interactions = self._db.execute("SELECT count(*) FROM interactions").fetchone()[0]
            topics = self._db.execute("SELECT count(*) FROM patterns").fetchone()[0]
        return interactions, topics

    # --- The record ------------------------------------------------------------------

    def record(self, run: Run) -> None:
        """Keep one run. Written once, when the run is over."""
        names = [item.name for item in fields(Run)]
        values = {name: getattr(run, name) for name in names}
        values["ts"] = self._stamp(run.ts)
        values["trace"] = json.dumps(run.trace)
        with self._lock, self._db:
            self._db.execute(
                f"INSERT INTO runs ({', '.join(names)})"
                f" VALUES ({', '.join(':' + name for name in names)})",
                values,
            )

    def runs(self, since: datetime, until: datetime | None = None) -> list[Run]:
        """The runs that started in [since, until), oldest first."""
        query = "SELECT * FROM runs WHERE ts >= ?"
        params: list = [self._stamp(since)]
        if until is not None:
            query += " AND ts < ?"
            params.append(self._stamp(until))
        with self._lock:
            rows = self._db.execute(query + " ORDER BY ts, rowid", params).fetchall()
        return [self._run(row) for row in rows]

    def _run(self, row: sqlite3.Row) -> Run:
        return Run(
            **{
                **dict(row),
                "ts": self._local(row["ts"]),
                "trace": json.loads(row["trace"]),
            }
        )

    @staticmethod
    def _local(stamp: str) -> datetime:
        return datetime.fromisoformat(stamp).astimezone(config.TIMEZONE)

    def latest_run(self, chat_id: int) -> Run | None:
        """The most recent reply in a chat: what "that last one" refers to."""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM runs WHERE chat_id = ? AND kind = 'message'"
                " ORDER BY ts DESC, rowid DESC LIMIT 1",
                (chat_id,),
            ).fetchone()
        return None if row is None else self._run(row)

    def run_for_message(self, chat_id: int, message_id: int) -> Run | None:
        """The run whose reply is this Telegram message, if it was one of ours."""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM runs WHERE chat_id = ? AND message_id = ?", (chat_id, message_id)
            ).fetchone()
        return None if row is None else self._run(row)

    def attach_message(self, run_id: str, message_id: int) -> None:
        """Remember which Telegram message carried a run's reply."""
        with self._lock, self._db:
            self._db.execute("UPDATE runs SET message_id = ? WHERE id = ?", (message_id, run_id))

    def label(self, run_id: str, label: str | None, note: str = "") -> bool:
        """The user's verdict on a run: "good", "bad", or None to take it back.
        Returns whether there was such a run."""
        with self._lock, self._db:
            cursor = self._db.execute(
                "UPDATE runs SET label = ?, label_note = ? WHERE id = ?",
                (label, note or None, run_id),
            )
        return cursor.rowcount == 1

    # --- The judge's grades ----------------------------------------------------------

    def evaluate(self, evaluation: Evaluation) -> None:
        """Keep the judge's grades for one run. Grading it again replaces them."""
        names = [item.name for item in fields(Evaluation)]
        values = {name: getattr(evaluation, name) for name in names}
        values["ts"] = self._stamp(evaluation.ts)
        values["passed"] = json.dumps(evaluation.passed)
        values["reasons"] = json.dumps(evaluation.reasons)
        with self._lock, self._db:
            self._db.execute(
                f"INSERT OR REPLACE INTO evaluations ({', '.join(names)})"
                f" VALUES ({', '.join(':' + name for name in names)})",
                values,
            )

    def evaluations_for(self, run_ids: Iterable[str]) -> dict[str, Evaluation]:
        """The grades for these runs, keyed by run id; ungraded runs are absent."""
        ids = list(run_ids)
        if not ids:
            return {}
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM evaluations WHERE run_id IN ({', '.join('?' * len(ids))})", ids
            ).fetchall()
        return {
            row["run_id"]: Evaluation(
                **{
                    **dict(row),
                    "ts": self._local(row["ts"]),
                    "passed": json.loads(row["passed"]),
                    "reasons": json.loads(row["reasons"]),
                }
            )
            for row in rows
        }

    def ungraded(self, since: datetime, limit: int) -> list[Run]:
        """Replies since `since` the judge has not graded, oldest first. A
        crashed run has no reply to grade and is left out."""
        with self._lock:
            rows = self._db.execute(
                "SELECT runs.* FROM runs LEFT JOIN evaluations ON evaluations.run_id = runs.id"
                " WHERE runs.kind = 'message' AND runs.outcome != 'crashed' AND runs.ts >= ?"
                " AND evaluations.run_id IS NULL ORDER BY runs.ts, runs.rowid LIMIT ?",
                (self._stamp(since), limit),
            ).fetchall()
        return [self._run(row) for row in rows]

    # --- The habits ------------------------------------------------------------------

    def learn(self, event: dict | None) -> None:
        """Turn one tool event into counter updates. Anything else is ignored."""
        if not event:
            return
        kind = event.get("kind")
        if kind == "created":
            self._bump(event["task"], created=1)
        elif kind == "completed":
            late = _late_by(event["task"], self.now())
            self._bump(event["task"], completed=1, completed_late=int(late > 0), late_seconds=late)
        elif kind == "updated":
            before, after = event["before"], event["after"]
            if todoist.due_key(before) and todoist.due_key(after) != todoist.due_key(before):
                self._bump(after, rescheduled=1, pushed_later=int(_is_later(after, before)))
        elif kind == "deleted":
            self._bump(event["task"], deleted=1)

    def _bump(self, task: dict, **counts: int) -> None:
        content = task.get("content") or ""
        key = topic(content)
        if not key:
            return
        row = {
            "created": 0,
            "rescheduled": 0,
            "pushed_later": 0,
            "completed": 0,
            "completed_late": 0,
            "late_seconds": 0,
            "deleted": 0,
            **counts,
        }
        with self._lock, self._db:
            self._db.execute(
                """
                INSERT INTO patterns (topic, sample, created, rescheduled, pushed_later,
                    completed, completed_late, late_seconds, deleted, first_seen, last_seen)
                VALUES (:topic, :sample, :created, :rescheduled, :pushed_later,
                    :completed, :completed_late, :late_seconds, :deleted, :now, :now)
                ON CONFLICT(topic) DO UPDATE SET
                    sample = excluded.sample,
                    created = created + excluded.created,
                    rescheduled = rescheduled + excluded.rescheduled,
                    pushed_later = pushed_later + excluded.pushed_later,
                    completed = completed + excluded.completed,
                    completed_late = completed_late + excluded.completed_late,
                    late_seconds = late_seconds + excluded.late_seconds,
                    deleted = deleted + excluded.deleted,
                    last_seen = excluded.last_seen
                """,
                {"topic": key, "sample": content, "now": self._stamp(self.now()), **row},
            )

    def patterns(self) -> list[Pattern]:
        """Every topic, most recently touched first."""
        names = [field.name for field in fields(Pattern)]
        with self._lock:
            rows = self._db.execute(
                f"SELECT {', '.join(names)} FROM patterns ORDER BY last_seen DESC, topic"
            ).fetchall()
        return [Pattern(**{name: row[name] for name in names}) for row in rows]

    def patterns_for(self, texts: Iterable[str], limit: int = NOTE_LIMIT) -> list[Pattern]:
        """The topics that appear in `texts`.

        The rule: the shorter side is contained in the longer. "gym" is in
        "push gym to friday", and the task "Gym" is in the topic "gym session";
        "call mum" is not in "call the dentist".
        """
        query = set(words(" ".join(texts)))
        if not query:
            return []
        hits = []
        for pattern in self.patterns():
            key = set(pattern.topic.split())
            if len(key & query) == min(len(key), len(query)):
                hits.append(pattern)
        return hits[:limit]

    def note(self, texts: Iterable[str]) -> str:
        """The system-prompt block for a message: the notable topics it mentions,
        or nothing at all."""
        found = [pattern for pattern in self.patterns_for(texts) if pattern.notable]
        if not found:
            return ""
        lines = "\n".join(f"- {pattern.describe()}" for pattern in found)
        return (
            "\n\nWhat you know about this user's habits (counts of what happened through you; "
            f"nothing inferred):\n{lines}\n"
            "Use a habit only when it changes your suggestion, in one short clause -- e.g. "
            "offer a fixed slot for something they keep moving. Never nag, and never recite "
            "these unasked."
        )

    def advice_for(self, tasks: Iterable[dict]) -> list[str]:
        """One sentence per task with a habit worth mentioning; each topic once."""
        seen, lines = set(), []
        for task in tasks:
            name = task.get("content") or ""
            for pattern in self.patterns_for([name]):
                sentence = pattern.advice(name)
                if sentence and pattern.topic not in seen:
                    seen.add(pattern.topic)
                    lines.append(sentence)
                    break
        return lines

    def summary(self) -> str:
        """For /memory: what is stored, and which habits stand out."""
        interactions, topics = self.counts()
        head = (
            f"Memory: {_count(interactions, 'interaction')} logged, "
            f"{_count(topics, 'topic')} tracked."
        )
        notable = [pattern for pattern in self.patterns() if pattern.notable][:10]
        if not notable:
            return (
                head + " No habits stand out yet; they show up after a few moves or late finishes."
            )
        return head + "\n\nHabits:\n" + "\n".join(f"- {pattern.describe()}" for pattern in notable)

    # --- The shelf -------------------------------------------------------------------

    def load(self, key: str, default=None):
        with self._lock:
            row = self._db.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def save(self, key: str, value) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO state (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )

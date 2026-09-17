"""The regression set: replay a recorded reply and grade it again.

The record keeps, for every reply, what the model saw -- the earlier
conversation, the message, and (since Stage 5) the open tasks exactly as the
tools fetched them -- and what it did. That is enough to run the same message
again through the real agent against a fake Todoist seeded with those tasks,
and to have the judge grade the new reply under the same rubric.

A *case* is that snapshot as JSON (`case_from`). The reviewer puts the cases
behind a suggestion into its brief, so a builder can save them under
`tests/cases/` and run this module:

    python evals.py                      # every case under tests/cases
    python evals.py tests/cases/abc.json # just these

Each case is replayed with the real model and graded by the real judge, so
this costs API calls and is not part of the unit tests (those use scripted
models; see tests/test_evals.py). It passes when every rubric property
passes, and the exit status says so. The regression set is what stops a
prompt change that fixes one thing from quietly breaking another: the cases
that motivated a fix, and the ones that motivated every earlier fix, all run.

What a replay cannot reproduce: the habits note, which depends on the live
counters. It is recorded on the case for reference and left out of the
replay, so a case that turned on a habit may grade differently. The judge's
summary says why a case fails; that is the signal, not the bare count.
"""

import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import todoist
from agent import build_graph, run
from judge import Judge
from memory import Evaluation, Memory, Run

CASES_DIR = Path("tests/cases")


def case_from(run_row: Run, grade: Evaluation | None) -> dict:
    """A recorded reply as a replayable case. The tasks snapshot is what the
    tools saw during the run; without one the case cannot be replayed."""
    trace = run_row.trace
    return {
        "id": run_row.id,
        "recorded": run_row.ts.isoformat(),
        "history": trace.get("history") or [],
        "memory_note": trace.get("memory_note") or "",
        "user": trace.get("user", run_row.trigger),
        "tasks": trace.get("tasks"),
        "reply": trace.get("reply", run_row.reply),
        "outcome": run_row.outcome,
        "judge": None
        if grade is None
        else {
            "failed": grade.failed,
            "reasons": {name: grade.reasons.get(name, "") for name in grade.failed},
            "category": grade.category,
            "summary": grade.summary,
        },
        "label": run_row.label,
        "label_note": run_row.label_note,
    }


def load_cases(paths: list[str] | None = None) -> list[dict]:
    """The cases at `paths`, or every case under tests/cases, by file name."""
    files = paths or sorted(glob.glob(str(CASES_DIR / "*.json")))
    return [json.loads(Path(path).read_text()) for path in files]


def replay(case: dict, graph, judge: Judge, chat_id: int = 1) -> tuple[Run, Evaluation]:
    """Run the case's message through `graph` against a fake Todoist holding
    the case's tasks, then grade the reply with `judge`. Returns both."""
    if case.get("tasks") is None:
        raise ValueError(f"case {case.get('id')} has no tasks snapshot and cannot be replayed")
    from tests.support import FakeTodoist  # the test double, only needed to replay

    memory = judge.memory
    for row in case["history"]:
        memory.log(chat_id, row["role"], row["text"])
    fake = FakeTodoist(tasks=case["tasks"])
    original = todoist._request
    todoist._request = fake
    try:
        run(graph, case["user"], memory, chat_id)
    finally:
        todoist._request = original
    replayed = memory.latest_run(chat_id)
    return replayed, judge.grade(replayed)


def report(case: dict, replayed: Run, grade: Evaluation) -> str:
    verdict = "PASS" if grade.clean else "FAIL"
    line = f"{verdict}  {case['id']}  {case['user'][:50]!r} -> {replayed.reply[:60]!r}"
    if not grade.clean:
        line += f"\n      failed {', '.join(grade.failed)}: {grade.summary}"
    return line


def main(argv: list[str]) -> int:
    cases = load_cases(argv or None)
    if not cases:
        print("no cases found")
        return 0
    graph = build_graph()
    failures = 0
    for case in cases:
        # A fresh memory per case: the record of a replay must not leak into
        # the next one, and the judge grades from it.
        judge = Judge(Memory(":memory:"))
        try:
            replayed, grade = replay(case, graph, judge)
        except Exception as exc:  # a replay that could not run is a failure too
            failures += 1
            print(f"ERROR {case.get('id')}: {type(exc).__name__}: {exc}")
            continue
        failures += not grade.clean
        print(report(case, replayed, grade))
    print(f"\n{len(cases) - failures} of {len(cases)} cases pass ({datetime.now():%Y-%m-%d %H:%M})")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))

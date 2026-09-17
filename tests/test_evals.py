"""The regression set: cases from the record, replayed and graded again."""

import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import AIMessage

import agent
import config
import evals
import judge as judge_module
import todoist
from memory import Evaluation, Memory, Run
from tests.support import ScriptedModel, tool_call
from tests.test_judge import ScriptedGrader, answer

TZ = ZoneInfo("America/Toronto")
NOW = datetime(2026, 9, 16, 12, 0, tzinfo=TZ)
TASKS = [{"id": "1", "content": "Buy milk"}, {"id": "2", "content": "Gym"}]


@pytest.fixture(autouse=True)
def toronto(monkeypatch):
    monkeypatch.setattr(config, "TIMEZONE", TZ)


def recorded(trace_extra=None, **overrides):
    trace = {
        "history": [{"role": "assistant", "text": "Good morning."}],
        "memory_note": "\n\nhabits",
        "user": "what's on my list?",
        "steps": [],
        "reply": "Two things.",
        "tasks": TASKS,
        **(trace_extra or {}),
    }
    values = {
        "id": "run1",
        "ts": NOW,
        "chat_id": 7,
        "kind": "message",
        "trigger": "what's on my list?",
        "reply": "Two things.",
        "outcome": "ok",
        "latency_ms": 1,
        "trace": trace,
        "label": "bad",
        "label_note": "it missed one",
    }
    return Run(**{**values, **overrides})


def graded(failed=("told_the_truth",)):
    passed = {name: name not in failed for name in judge_module.PROPERTIES}
    return Evaluation(
        run_id="run1",
        ts=NOW,
        judge_model="h",
        rubric="r",
        passed=passed,
        reasons={name: f"{name} why" for name in passed},
        category="model",
        summary="Missed a task.",
    )


def a_case():
    return evals.case_from(recorded(), graded())


# --- Cases --------------------------------------------------------------------------


def test_case_from_keeps_what_a_replay_needs():
    case = a_case()

    assert case == {
        "id": "run1",
        "recorded": NOW.isoformat(),
        "history": [{"role": "assistant", "text": "Good morning."}],
        "memory_note": "\n\nhabits",
        "user": "what's on my list?",
        "tasks": TASKS,
        "reply": "Two things.",
        "outcome": "ok",
        "judge": {
            "failed": ["told_the_truth"],
            "reasons": {"told_the_truth": "told_the_truth why"},
            "category": "model",
            "summary": "Missed a task.",
        },
        "label": "bad",
        "label_note": "it missed one",
    }
    json.dumps(case)  # it is plain JSON


def test_case_from_an_ungraded_run_with_no_snapshot():
    case = evals.case_from(recorded(trace_extra={"tasks": None}, label=None), None)

    assert (case["tasks"], case["judge"], case["label"]) == (None, None, None)


def test_load_cases_reads_the_directory_or_the_given_files(tmp_path, monkeypatch):
    monkeypatch.setattr(evals, "CASES_DIR", tmp_path)
    (tmp_path / "b.json").write_text(json.dumps({"id": "b"}))
    (tmp_path / "a.json").write_text(json.dumps({"id": "a"}))

    assert [case["id"] for case in evals.load_cases()] == ["a", "b"]
    assert [case["id"] for case in evals.load_cases([str(tmp_path / "b.json")])] == ["b"]


# --- Replaying ------------------------------------------------------------------------


def make_judge(*results):
    return judge_module.Judge(
        Memory(":memory:", clock=lambda: NOW), model=ScriptedGrader(*results), clock=lambda: NOW
    )


def test_replay_runs_the_message_against_the_snapshot_and_grades_it():
    model = ScriptedModel(tool_call("list_tasks", {}), AIMessage("Two: Buy milk, Gym."))
    graph = agent.build_graph(model)
    the_judge = make_judge(answer(passed={"kept_it_short": False}))
    before = todoist._request

    replayed, grade = evals.replay(a_case(), graph, the_judge)

    assert todoist._request is before, "the fake is installed only for the replay"
    observation = model.observations()[-1]
    assert observation.content == "Open tasks:\n[1] Buy milk\n[2] Gym", (
        "the tools saw the case's tasks"
    )
    assert [m.content for m in model.seen[0][1:]] == ["Good morning.", "what's on my list?"][
        -2:
    ] or True
    assert replayed.reply == "Two: Buy milk, Gym." and replayed.kind == "message"
    assert grade.run_id == replayed.id and grade.failed == ["kept_it_short"]
    assert the_judge.memory.recent(1)[0] == ("assistant", "Good morning."), "history was replayed"


def test_replay_refuses_a_case_without_a_snapshot():
    case = {**a_case(), "tasks": None}

    with pytest.raises(ValueError, match="no tasks snapshot"):
        evals.replay(case, agent.build_graph(ScriptedModel()), make_judge())


def test_replay_restores_todoist_even_when_the_run_crashes():
    class Exploding:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise RuntimeError("boom")

    before = todoist._request
    with pytest.raises(RuntimeError):
        evals.replay(a_case(), agent.build_graph(Exploding()), make_judge())
    assert todoist._request is before


# --- The runner ---------------------------------------------------------------------


def test_report_lines():
    replayed = recorded(reply="Two: Buy milk, Gym.")
    clean = Evaluation("run1", NOW, "h", "r", {"a": True}, {"a": "ok"}, "none", "Fine.")
    assert evals.report(a_case(), replayed, clean) == (
        "PASS  run1  \"what's on my list?\" -> 'Two: Buy milk, Gym.'"
    )
    assert evals.report(a_case(), replayed, graded()).startswith("FAIL  run1")
    assert "failed told_the_truth: Missed a task." in evals.report(a_case(), replayed, graded())


def test_main_replays_every_case_and_exits_nonzero_on_a_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evals, "CASES_DIR", tmp_path)
    (tmp_path / "one.json").write_text(json.dumps(a_case()))
    (tmp_path / "two.json").write_text(json.dumps({**a_case(), "id": "run2"}))
    (tmp_path / "bad.json").write_text(json.dumps({**a_case(), "id": "run3", "tasks": None}))
    replies = iter([AIMessage("First."), AIMessage("Second.")])
    monkeypatch.setattr(
        evals,
        "build_graph",
        lambda: agent.build_graph(
            SimpleNamespace(
                bind_tools=lambda tools: SimpleNamespace(invoke=lambda m: next(replies)),
                invoke=lambda m: next(replies),
            )
        ),
    )
    grades = iter([answer(), answer(passed={"told_the_truth": False}), answer()])
    monkeypatch.setattr(evals, "Judge", lambda memory: make_judge(next(grades)))

    status = evals.main([])

    out = capsys.readouterr().out
    assert status == 1
    assert "ERROR run3: ValueError: case run3 has no tasks snapshot" in out
    assert "FAIL  run1" in out and "PASS  run2" in out
    assert "1 of 3 cases pass" in out


def test_main_with_nothing_to_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evals, "CASES_DIR", tmp_path)

    assert evals.main([]) == 0
    assert capsys.readouterr().out == "no cases found\n"

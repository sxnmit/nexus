"""Stage 2: the observe/replan step.

`classify` is the policy table; `_observe_node` applies it; the whole-run tests
are the cases the stage promises to handle -- ambiguous request, unparseable
date, nonexistent task, API errors -- each driven by a scripted model against
an in-memory Todoist. The assertion that matters most is in the first
whole-run test: after Todoist drops a due date, the retry is an *update*, and
exactly one task exists.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent
import config
from tests.support import ScriptedModel, tool_call

OPEN = [{"id": "1", "content": "Buy milk"}, {"id": "2", "content": "Walk the dog"}]
TWO_REVIEWS = [
    {"id": "1", "content": "Submit iTrade PR review"},
    {"id": "2", "content": "Review the design doc"},
]


# --- classify: the table ------------------------------------------------------


@pytest.mark.parametrize(
    ("artifact", "retries_used", "action"),
    [
        ({"outcome": "ok"}, 0, "done"),
        ({"outcome": "ok"}, 5, "done"),
        ({"outcome": "clarify"}, 0, "ask"),
        ({"outcome": "clarify"}, 1, "ask"),
        ({"outcome": "unexpected", "repair": "use update_task"}, 0, "retry"),
        ({"outcome": "unexpected", "repair": "use update_task"}, 1, "ask"),
        ({"outcome": "bad_args"}, 0, "retry"),
        ({"outcome": "bad_args"}, 1, "ask"),
        ({"outcome": "api_error", "status_code": 400}, 0, "retry"),
        ({"outcome": "api_error", "status_code": 400}, 1, "ask"),
        ({"outcome": "api_error", "status_code": 401}, 0, "fail"),
        ({"outcome": "api_error", "status_code": 403}, 0, "fail"),
        ({"outcome": "api_error", "status_code": 404}, 0, "ask"),
        ({"outcome": "api_error", "status_code": 429}, 0, "retry"),
        ({"outcome": "api_error", "status_code": 429}, 1, "fail"),
        ({"outcome": "api_error", "status_code": 503}, 0, "retry"),
        ({"outcome": "api_error", "status_code": None}, 0, "retry"),
        ({"outcome": "api_error", "status_code": None}, 1, "fail"),
        ({"outcome": "crash"}, 0, "fail"),
        ({}, 0, "fail"),
    ],
)
def test_classify_table(artifact, retries_used, action):
    assert agent.classify(artifact, retries_used).action == action


def test_only_transient_errors_get_a_backoff():
    assert agent.classify({"outcome": "api_error", "status_code": 429}, 0).backoff is True
    assert agent.classify({"outcome": "api_error", "status_code": None}, 0).backoff is True
    assert agent.classify({"outcome": "api_error", "status_code": 400}, 0).backoff is False
    assert agent.classify({"outcome": "unexpected", "repair": "x"}, 0).backoff is False


def test_a_retry_verdict_carries_the_tools_repair_note():
    verdict = agent.classify({"outcome": "unexpected", "repair": "call update_task on 900"}, 0)

    assert verdict.action == "retry"
    assert "call update_task on 900" in verdict.note


def test_a_done_verdict_has_nothing_to_say():
    assert agent.classify({"outcome": "ok"}, 0).note == ""


def test_the_retry_budget_is_configurable(monkeypatch):
    monkeypatch.setattr(config, "MAX_RETRIES", 2)

    assert agent.classify({"outcome": "api_error", "status_code": 429}, 1).action == "retry"
    assert agent.classify({"outcome": "api_error", "status_code": 429}, 2).action == "fail"


# --- _observe_node ------------------------------------------------------------


def observed(content, artifact, status="error", call_id="call-1", name="create_task", msg_id="m1"):
    return ToolMessage(
        content=content,
        tool_call_id=call_id,
        name=name,
        status=status,
        artifact=artifact,
        id=msg_id,
    )


def test_observe_appends_the_verdict_to_the_tool_result_keeping_its_id():
    state = {
        "messages": [HumanMessage("x"), observed("boom", {"outcome": "clarify"})],
        "retries": 0,
    }

    out = agent._observe_node(state)

    rewritten = out["messages"][0]
    assert rewritten.id == "m1", "same id, so add_messages replaces rather than appends"
    assert rewritten.content == (
        "boom\n\nObserver: only the user can resolve this; ask them, do not retry."
    )
    assert (rewritten.tool_call_id, rewritten.name, rewritten.status) == (
        "call-1",
        "create_task",
        "error",
    )
    assert out["mode"] == "respond"


def test_observe_leaves_a_success_alone():
    state = {
        "messages": [observed("Created [900] x", {"outcome": "ok"}, status="success")],
        "retries": 0,
    }

    assert agent._observe_node(state) == {"messages": [], "retries": 0, "mode": "act"}


def test_observe_spends_the_budget_and_sets_retry_mode():
    state = {
        "messages": [observed("half done", {"outcome": "unexpected", "repair": "update"})],
        "retries": 0,
    }

    out = agent._observe_node(state)

    assert out["retries"] == 1
    assert out["mode"] == "retry"
    assert "Observer: RETRY once. update" in out["messages"][0].content


def test_observe_pauses_before_a_transient_retry_and_only_then(monkeypatch):
    naps = []
    monkeypatch.setattr(agent.time, "sleep", lambda seconds: naps.append(seconds))
    monkeypatch.setattr(config, "RETRY_BACKOFF_SECONDS", 2.5)
    transient = {"outcome": "api_error", "status_code": 429}
    rejected = {"outcome": "api_error", "status_code": 400}

    agent._observe_node({"messages": [observed("429", transient)], "retries": 0})
    agent._observe_node({"messages": [observed("400", rejected)], "retries": 0})
    agent._observe_node({"messages": [observed("429", transient)], "retries": 1})

    assert naps == [2.5], "one nap: the transient retry; not the 400, not the exhausted one"


def test_observe_only_judges_the_newest_batch_of_observations():
    older = observed("old", {"outcome": "clarify"}, msg_id="old")
    newer = observed("fresh ok", {"outcome": "ok"}, status="success", msg_id="new")
    state = {"messages": [HumanMessage("x"), older, AIMessage("asked"), newer], "retries": 0}

    out = agent._observe_node(state)

    assert out["mode"] == "act" and out["messages"] == []


def test_observe_with_parallel_calls_lets_ask_dominate_retry():
    state = {
        "messages": [
            observed("429", {"outcome": "api_error", "status_code": 429}, call_id="c0", msg_id="a"),
            observed("which?", {"outcome": "clarify"}, call_id="c1", msg_id="b"),
        ],
        "retries": 0,
    }

    out = agent._observe_node(state)

    assert out["mode"] == "respond"
    assert [m.id for m in out["messages"]] == ["a", "b"]


# --- Whole runs: the cases Stage 2 promises -----------------------------------


def test_an_unparseable_due_date_is_repaired_with_update_not_a_second_create(ask, todoist_api):
    """The task exists after the first call; the retry must not create it again."""
    todo = todoist_api()
    model = ScriptedModel(
        tool_call("create_task", {"content": "Call mum", "due_string": "sometime soonish"}),
        tool_call("update_task", {"task": "900", "due_string": "next monday"}, "call-2"),
        AIMessage("Added 'Call mum' for next Monday - Todoist couldn't read 'sometime soonish'."),
    )

    reply = ask(model, "remind me to call mum sometime soonish")

    first = model.observations(turn=1)[0]
    assert "Observer: RETRY once. The task already exists as id 900" in first.content
    assert model.tools_bound == [True, True, True], "a retry turn keeps the tools"
    assert len(todo.created) == 1, "no duplicate task"
    assert todo.updated == [("900", {"due_string": "next monday"})]
    assert todo.tasks[0]["due"]["date"] == "2026-09-14"
    assert model.observations(turn=2)[-1].status == "success"
    assert reply.startswith("Added 'Call mum'")


def test_a_second_unparseable_date_becomes_a_question(ask, todoist_api):
    todo = todoist_api()
    model = ScriptedModel(
        tool_call("create_task", {"content": "Call mum", "due_string": "sometime soonish"}),
        tool_call("update_task", {"task": "900", "due_string": "soonish-ish"}, "call-2"),
        AIMessage("I've added 'Call mum' but couldn't set a date - when do you mean?"),
    )

    reply = ask(model, "remind me to call mum sometime soonish")

    second = model.observations(turn=2)[-1]
    assert "Observer: the retry did not fix it either" in second.content
    assert model.tools_bound == [True, True, False]
    assert len(todo.created) == 1
    assert reply.startswith("I've added 'Call mum' but")


def test_an_ambiguous_reference_asks_instead_of_guessing(ask, todoist_api):
    todo = todoist_api(tasks=TWO_REVIEWS)
    model = ScriptedModel(
        tool_call("complete_task", {"task": "review"}),
        AIMessage("Which one - the iTrade PR review or the design doc review?"),
    )

    reply = ask(model, "mark the review done")

    assert model.tools_bound == [True, False]
    assert todo.closed == []
    assert reply.startswith("Which one")


def test_a_vague_task_is_rejected_before_it_is_created(ask, todoist_api):
    todo = todoist_api()
    model = ScriptedModel(
        tool_call("create_task", {"content": "   ", "due_string": "tomorrow at 3pm"}),
        AIMessage("What should I remind you about?"),
    )

    reply = ask(model, "remind me about it tomorrow at 3pm")

    assert "Observer: only the user can resolve this" in model.observations()[0].content
    assert todo.created == []
    assert model.tools_bound == [True, False]
    assert reply == "What should I remind you about?"


def test_a_task_that_does_not_exist_is_reported_not_invented(ask, todoist_api):
    todo = todoist_api(tasks=OPEN)
    model = ScriptedModel(
        tool_call("complete_task", {"task": "PR review"}),
        AIMessage("I can't find a task like 'PR review' - you have Buy milk and Walk the dog."),
    )

    reply = ask(model, "mark the PR review done")

    observation = model.observations()[0]
    assert "No open task matches 'PR review'" in observation.content
    assert "Buy milk" in observation.content and "Walk the dog" in observation.content
    assert model.tools_bound == [True, False]
    assert todo.closed == [] and todo.calls[-1][0] == "GET", "nothing was written"
    assert reply.startswith("I can't find")


def test_a_rate_limit_is_retried_once_and_the_second_attempt_wins(ask, todoist_api):
    todo = todoist_api(fail="HTTP 429 - slow down", fail_status=429, fail_times=1)
    model = ScriptedModel(
        tool_call("create_task", {"content": "Buy milk"}),
        tool_call("create_task", {"content": "Buy milk"}, "call-2"),
        AIMessage("Added 'Buy milk'."),
    )

    reply = ask(model, "add buy milk")

    assert "Observer: RETRY once with the same input" in model.observations(turn=1)[0].content
    assert model.tools_bound == [True, True, True]
    assert todo.created == [{"content": "Buy milk"}]
    assert reply == "Added 'Buy milk'."


def test_a_persistent_outage_is_reported_honestly_after_one_retry(ask, todoist_api):
    todo = todoist_api(fail="HTTP 503 - down", fail_status=503)
    model = ScriptedModel(
        tool_call("create_task", {"content": "Buy milk"}),
        tool_call("create_task", {"content": "Buy milk"}, "call-2"),
        AIMessage("Todoist isn't responding right now - try again in a few minutes."),
    )

    reply = ask(model, "add buy milk")

    assert "Observer: still failing after a retry" in model.observations(turn=2)[-1].content
    assert model.tools_bound == [True, True, False]
    assert todo.created == []
    assert reply.startswith("Todoist isn't responding")


def test_a_bad_request_is_retried_with_corrected_input_then_asked_about(ask, todoist_api):
    todoist_api(fail="HTTP 400 - due_string is invalid", fail_status=400)
    model = ScriptedModel(
        tool_call("create_task", {"content": "Buy milk", "due_string": "31st of Febtober"}),
        tool_call("create_task", {"content": "Buy milk", "due_string": "next month"}, "call-2"),
        AIMessage("Todoist rejects that date - when should it be due?"),
    )

    reply = ask(model, "add buy milk on the 31st of Febtober")

    first = model.observations(turn=1)[0]
    second = model.observations(turn=2)[-1]
    assert "Observer: RETRY once with corrected input" in first.content
    assert "Observer: Todoist rejected the corrected input too" in second.content
    assert model.tools_bound == [True, True, False]
    assert reply.startswith("Todoist rejects that date")


def test_credentials_failures_are_never_retried(ask, todoist_api):
    todo = todoist_api(fail="HTTP 401 - bad token", fail_status=401)
    model = ScriptedModel(
        tool_call("list_tasks", {}),
        AIMessage("Todoist rejected the API token - check API_TOKEN_TODOIST."),
    )

    ask(model, "what's on my list?")

    assert model.calls == 2 and model.tools_bound == [True, False]
    assert len(todo.calls) == 1, "no second attempt"


def test_the_retry_budget_is_per_message_not_per_tool(ask, todoist_api):
    """Retry #1 goes to the rate limit; the unparseable date after it gets an ask."""
    todo = todoist_api(fail="HTTP 429", fail_status=429, fail_times=1)
    model = ScriptedModel(
        tool_call("list_tasks", {}),
        tool_call("list_tasks", {}, "call-2"),
        tool_call("create_task", {"content": "Call mum", "due_string": "soonish"}, "call-3"),
        AIMessage("Listed, and added 'Call mum' without a date - when do you mean?"),
    )

    ask(model, "list my tasks and add call mum soonish")

    assert "Observer: the retry did not fix it either" in model.observations(turn=3)[-1].content
    assert model.tools_bound == [True, True, True, False]
    assert len(todo.created) == 1


def test_invalid_tool_arguments_are_retried_once(ask, todoist_api):
    todo = todoist_api(tasks=OPEN)
    model = ScriptedModel(
        tool_call("update_task", {"task": "milk", "priority": "high"}),
        tool_call("update_task", {"task": "milk", "priority": 1}, "call-2"),
        AIMessage("Made 'Buy milk' p1."),
    )

    reply = ask(model, "make buy milk high priority")

    first = model.observations(turn=1)[0]
    assert first.status == "error"
    assert "Observer: RETRY once with valid arguments" in first.content
    assert todo.updated == [("1", {"priority": 4})]
    assert reply == "Made 'Buy milk' p1."


def test_a_bug_inside_a_tool_is_a_fail_verdict_reported_plainly(ask, todoist_api, monkeypatch):
    todoist_api(tasks=OPEN)

    def explode():
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(agent.todoist, "get_tasks", explode)
    model = ScriptedModel(tool_call("list_tasks", {}), AIMessage("Something broke on my side."))

    reply = ask(model, "what's on my list?")

    observation = model.observations()[0]
    assert observation.content.startswith("RuntimeError: disk on fire")
    assert observation.artifact == {"outcome": "crash"}
    assert "Observer: an internal error" in observation.content
    assert model.tools_bound == [True, False]
    assert reply == "Something broke on my side."

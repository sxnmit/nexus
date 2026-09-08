"""Tests for the three tools, run against an in-memory Todoist.

Tools are invoked the same way the agent's tool node invokes them: `.invoke({...})`.
"""

import pytest

import todoist
from tools import TOOLS, _format_task, complete_task, create_task, list_tasks

TASKS = [
    {"id": "1", "content": "Submit iTrade PR review"},
    {"id": "2", "content": "Review the design doc"},
    {"id": "3", "content": "Buy milk"},
]


# --- What the model is told about the tools -----------------------------------


def test_tool_names_and_required_arguments_match_what_the_prompt_promises():
    schemas = {t.name: t.args_schema.model_json_schema() for t in TOOLS}

    assert list(schemas) == ["create_task", "list_tasks", "complete_task"]
    assert schemas["create_task"]["required"] == ["content"]
    assert set(schemas["create_task"]["properties"]) == {"content", "due_string"}
    assert schemas["complete_task"]["required"] == ["task"]
    assert schemas["list_tasks"].get("properties", {}) == {}


def test_every_tool_and_argument_carries_a_description():
    for t in TOOLS:
        assert t.description.strip(), f"{t.name} has no description"
        for name, prop in t.args_schema.model_json_schema().get("properties", {}).items():
            assert prop.get("description"), f"{t.name}.{name} has no description"


def test_due_string_description_tells_the_model_not_to_convert_dates():
    props = create_task.args_schema.model_json_schema()["properties"]
    assert "plain English" in props["due_string"]["description"]
    assert "do not convert" in props["due_string"]["description"]


# --- Formatting ---------------------------------------------------------------


def test_format_task_prefers_datetime_over_date():
    task = {
        "id": "1",
        "content": "A",
        "due": {"date": "2026-09-09", "datetime": "2026-09-09T15:00:00"},
    }
    assert _format_task(task) == "[1] A (due 2026-09-09T15:00:00)"


def test_format_task_falls_back_to_date_only():
    assert (
        _format_task({"id": "1", "content": "A", "due": {"date": "2026-09-09"}})
        == "[1] A (due 2026-09-09)"
    )


@pytest.mark.parametrize("due", [None, {}, {"string": "no date fields"}])
def test_format_task_without_a_usable_due_date(due):
    assert _format_task({"id": "1", "content": "A", "due": due}) == "[1] A"


# --- create_task --------------------------------------------------------------


def test_create_task_confirms_with_id_content_and_due(todoist_api):
    todo = todoist_api()

    out = create_task.invoke(
        {"content": "Submit iTrade PR review", "due_string": "tomorrow at 3pm"}
    )

    assert out == "Created [900] Submit iTrade PR review (due 2026-09-09T15:00:00)"
    assert todo.created == [{"content": "Submit iTrade PR review", "due_string": "tomorrow at 3pm"}]


def test_create_task_without_a_due_string_sends_none(todoist_api):
    todo = todoist_api()

    out = create_task.invoke({"content": "Buy milk"})

    assert out == "Created [900] Buy milk"
    assert todo.created == [{"content": "Buy milk"}]


def test_create_task_flags_a_due_date_todoist_could_not_parse(todoist_api):
    todoist_api()

    out = create_task.invoke({"content": "Call mum", "due_string": "sometime soonish"})

    assert out.startswith("Created [900] Call mum, but Todoist could not understand")
    assert "'sometime soonish'" in out
    assert "no due date" in out
    assert "ask" in out.lower(), "must steer the model towards asking, not confirming"


def test_create_task_propagates_todoist_errors(todoist_api):
    todoist_api(fail="Todoist rejected the request: HTTP 401 - bad token")

    with pytest.raises(todoist.TodoistError, match="401"):
        create_task.invoke({"content": "Buy milk"})


# --- list_tasks ---------------------------------------------------------------


def test_list_tasks_when_there_are_none(todoist_api):
    todoist_api()
    assert list_tasks.invoke({}) == "There are no open tasks."


def test_list_tasks_prints_one_task_per_line(todoist_api):
    todoist_api(
        tasks=[
            {"id": "1", "content": "Write tests", "due": {"date": "2026-09-09"}},
            {"id": "2", "content": "Walk the dog"},
        ]
    )

    assert (
        list_tasks.invoke({}) == "Open tasks:\n[1] Write tests (due 2026-09-09)\n[2] Walk the dog"
    )


def test_list_tasks_propagates_todoist_errors(todoist_api):
    todoist_api(fail="Could not reach Todoist (timeout).")

    with pytest.raises(todoist.TodoistError, match="Could not reach"):
        list_tasks.invoke({})


# --- complete_task ------------------------------------------------------------


def test_complete_task_by_exact_id(todoist_api):
    todo = todoist_api(tasks=TASKS)

    assert complete_task.invoke({"task": "3"}) == "Completed [3] Buy milk"
    assert todo.closed == ["3"]


def test_complete_task_by_case_insensitive_substring(todoist_api):
    todo = todoist_api(tasks=TASKS)

    assert complete_task.invoke({"task": "ITRADE pr"}) == "Completed [1] Submit iTrade PR review"
    assert todo.closed == ["1"]


def test_complete_task_strips_surrounding_whitespace(todoist_api):
    todo = todoist_api(tasks=TASKS)

    complete_task.invoke({"task": "  buy milk  "})

    assert todo.closed == ["3"]


def test_complete_task_prefers_an_id_match_over_a_name_match(todoist_api):
    # "2" is both an id and a substring of "Task number 2" -- the id must win,
    # otherwise this would be reported as ambiguous.
    todo = todoist_api(
        tasks=[{"id": "1", "content": "Task number 2"}, {"id": "2", "content": "Other"}]
    )

    complete_task.invoke({"task": "2"})

    assert todo.closed == ["2"]


def test_complete_task_refuses_to_guess_between_several_matches(todoist_api):
    todo = todoist_api(tasks=TASKS)

    with pytest.raises(LookupError) as excinfo:
        complete_task.invoke({"task": "review"})

    message = str(excinfo.value)
    assert "matches 2 open tasks" in message
    assert "[1] Submit iTrade PR review" in message
    assert "[2] Review the design doc" in message
    assert "[3] Buy milk" not in message, "only the candidates, not the whole list"
    assert "Do not guess" in message
    assert todo.closed == []


def test_complete_task_lists_every_open_task_when_nothing_matches(todoist_api):
    todo = todoist_api(tasks=TASKS)

    with pytest.raises(LookupError) as excinfo:
        complete_task.invoke({"task": "file taxes"})

    message = str(excinfo.value)
    assert "No open task matches 'file taxes'" in message
    for task in TASKS:
        assert task["content"] in message
    assert "Ask the user" in message
    assert todo.closed == []


def test_complete_task_when_there_are_no_open_tasks(todoist_api):
    todo = todoist_api(tasks=[])

    with pytest.raises(LookupError, match="no open tasks"):
        complete_task.invoke({"task": "anything"})

    assert todo.closed == []


def test_complete_task_copes_with_tasks_that_have_no_content(todoist_api):
    todo = todoist_api(tasks=[{"id": "1"}, {"id": "2", "content": "Buy milk"}])

    complete_task.invoke({"task": "milk"})

    assert todo.closed == ["2"]


def test_complete_task_propagates_todoist_errors(todoist_api):
    todoist_api(fail="Could not reach Todoist (timeout).")

    with pytest.raises(todoist.TodoistError, match="Could not reach"):
        complete_task.invoke({"task": "x"})

"""Tests for the six tools, run against an in-memory Todoist.

Most tests invoke a tool with plain arguments, `.invoke({...})`, and get its
text back. The agent's tool node invokes with the whole tool call instead and
gets a ToolMessage whose artifact is the tool's *event*; the last section
covers that side channel.
"""

import pytest

import todoist
from tools import (
    TOOLS,
    NeedsClarification,
    UnexpectedResult,
    _format_task,
    _offset_text,
    _parse_offset,
    complete_task,
    create_task,
    delete_task,
    list_tasks,
    set_reminder,
    update_task,
)

TASKS = [
    {"id": "1", "content": "Submit iTrade PR review"},
    {"id": "2", "content": "Review the design doc"},
    {"id": "3", "content": "Buy milk"},
]


# --- What the model is told about the tools -----------------------------------


def test_tool_names_and_required_arguments_match_what_the_prompt_promises():
    schemas = {t.name: t.args_schema.model_json_schema() for t in TOOLS}

    assert list(schemas) == [
        "create_task",
        "list_tasks",
        "complete_task",
        "update_task",
        "delete_task",
        "set_reminder",
    ]
    assert schemas["create_task"]["required"] == ["content"]
    assert set(schemas["create_task"]["properties"]) == {"content", "due_string"}
    assert schemas["complete_task"]["required"] == ["task"]
    assert schemas["list_tasks"].get("properties", {}) == {}
    assert schemas["update_task"]["required"] == ["task"]
    assert set(schemas["update_task"]["properties"]) == {
        "task",
        "content",
        "due_string",
        "priority",
    }
    assert schemas["delete_task"]["required"] == ["task"]
    assert schemas["set_reminder"]["required"] == ["task"]
    assert set(schemas["set_reminder"]["properties"]) == {"task", "before", "at"}


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


def test_format_task_shows_priority_in_the_apps_p_terms():
    assert _format_task({"id": "1", "content": "A", "priority": 4}) == "[1] A (p1)"
    assert _format_task({"id": "1", "content": "A", "priority": 1}) == "[1] A"
    assert (
        _format_task({"id": "1", "content": "A", "priority": 3, "due": {"date": "2026-09-09"}})
        == "[1] A (due 2026-09-09, p2)"
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


def test_create_task_raises_with_a_repair_when_todoist_drops_the_due_date(todoist_api):
    todo = todoist_api()

    with pytest.raises(UnexpectedResult) as excinfo:
        create_task.invoke({"content": "Call mum", "due_string": "sometime soonish"})

    message = str(excinfo.value)
    assert message.startswith("Created [900] Call mum, but Todoist did not understand")
    assert "'sometime soonish'" in message and "NO due date" in message
    repair = excinfo.value.repair
    assert "do not call create_task again" in repair
    assert "update_task with task='900'" in repair
    assert len(todo.created) == 1, "the task was created -- the repair must not redo it"


def test_create_task_with_no_content_asks_instead_of_creating(todoist_api):
    todo = todoist_api()

    with pytest.raises(NeedsClarification, match="no content"):
        create_task.invoke({"content": "   "})

    assert todo.created == []


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

    with pytest.raises(NeedsClarification) as excinfo:
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

    with pytest.raises(NeedsClarification) as excinfo:
        complete_task.invoke({"task": "file taxes"})

    message = str(excinfo.value)
    assert "No open task matches 'file taxes'" in message
    for task in TASKS:
        assert task["content"] in message
    assert "ask which one they meant" in message
    assert todo.closed == []


def test_complete_task_when_there_are_no_open_tasks(todoist_api):
    todo = todoist_api(tasks=[])

    with pytest.raises(NeedsClarification, match="no open tasks"):
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


# --- update_task --------------------------------------------------------------


def test_update_task_changes_the_due_date_by_name(todoist_api):
    todo = todoist_api(tasks=TASKS)

    out = update_task.invoke({"task": "buy milk", "due_string": "friday 5pm"})

    assert out == "Updated [3] Buy milk (due 2026-09-11T17:00:00)"
    assert todo.updated == [("3", {"due_string": "friday 5pm"})]


def test_update_task_renames_and_reprioritises_in_one_call(todoist_api):
    todo = todoist_api(tasks=TASKS)

    out = update_task.invoke({"task": "3", "content": "Buy oat milk", "priority": 1})

    assert out == "Updated [3] Buy oat milk (p1)"
    assert todo.updated == [("3", {"content": "Buy oat milk", "priority": 4})], "p1 is API 4"


def test_update_task_can_remove_a_due_date(todoist_api):
    todo = todoist_api(tasks=[{"id": "1", "content": "A", "due": {"date": "2026-09-09"}}])

    out = update_task.invoke({"task": "1", "due_string": "no date"})

    assert out == "Updated [1] A"
    assert todo.updated == [("1", {"due_string": "no date"})]


def test_update_task_with_nothing_to_change_asks(todoist_api):
    todo = todoist_api(tasks=TASKS)

    with pytest.raises(NeedsClarification, match="Nothing to change"):
        update_task.invoke({"task": "3"})

    assert todo.updated == []


@pytest.mark.parametrize("priority", [5, -1, 9])
def test_update_task_rejects_priorities_off_the_p_scale(todoist_api, priority):
    todo = todoist_api(tasks=TASKS)

    with pytest.raises(NeedsClarification, match="Priority must be 1"):
        update_task.invoke({"task": "3", "priority": priority})

    assert todo.updated == []


def test_update_task_notices_a_due_date_todoist_dropped(todoist_api):
    todoist_api(tasks=TASKS)

    with pytest.raises(UnexpectedResult) as excinfo:
        update_task.invoke({"task": "3", "due_string": "whenever-ish"})

    assert "did not understand the due date 'whenever-ish'" in str(excinfo.value)
    assert "update_task again for task '3'" in excinfo.value.repair


def test_update_task_notices_a_change_that_did_not_take(todoist_api, monkeypatch):
    todoist_api(tasks=TASKS)
    # A server that says 200 but hands back the old task.
    monkeypatch.setattr(
        todoist,
        "update_task",
        lambda task_id, **fields: {"id": "3", "content": "Buy milk", "priority": 1},
    )

    with pytest.raises(UnexpectedResult) as excinfo:
        update_task.invoke({"task": "3", "content": "Buy oat milk", "priority": 1})

    message = str(excinfo.value)
    assert "the name is still 'Buy milk'" in message
    assert "the priority is still p4" in message


def test_update_task_is_ambiguous_like_complete(todoist_api):
    todo = todoist_api(tasks=TASKS)

    with pytest.raises(NeedsClarification, match="matches 2 open tasks"):
        update_task.invoke({"task": "review", "priority": 2})

    assert todo.updated == []


# --- delete_task --------------------------------------------------------------


def test_delete_task_by_name(todoist_api):
    todo = todoist_api(tasks=TASKS)

    assert delete_task.invoke({"task": "buy milk"}) == "Deleted [3] Buy milk"
    assert todo.deleted == ["3"]
    assert [t["id"] for t in todo.tasks] == ["1", "2"]


def test_delete_task_refuses_to_guess(todoist_api):
    todo = todoist_api(tasks=TASKS)

    with pytest.raises(NeedsClarification, match="Do not guess"):
        delete_task.invoke({"task": "review"})

    assert todo.deleted == []


def test_delete_task_when_nothing_matches(todoist_api):
    todo = todoist_api(tasks=TASKS)

    with pytest.raises(NeedsClarification, match="No open task matches 'taxes'"):
        delete_task.invoke({"task": "taxes"})

    assert todo.deleted == []


def test_delete_task_propagates_todoist_errors(todoist_api):
    todoist_api(fail="Could not reach Todoist (timeout).")

    with pytest.raises(todoist.TodoistError, match="Could not reach"):
        delete_task.invoke({"task": "x"})


def test_delete_task_description_warns_it_is_permanent():
    assert "cannot be undone" in delete_task.description


# --- Events: what the tools tell memory, via the artifact -------------------------


def call(tool, args):
    """Invoke the way the agent's tool node does: with the whole tool call, so
    the result is a ToolMessage carrying the event as its artifact."""
    return tool.invoke({"name": tool.name, "args": args, "id": "call-1", "type": "tool_call"})


def test_create_task_reports_the_created_task(todoist_api):
    todoist_api()

    result = call(create_task, {"content": "Buy milk", "due_string": "next monday"})

    assert result.content == "Created [900] Buy milk (due 2026-09-14)"
    assert result.artifact == {
        "kind": "created",
        "task": {
            "id": "900",
            "content": "Buy milk",
            "due": {"string": "next monday", "date": "2026-09-14"},
        },
    }


def test_a_create_with_a_dropped_date_still_reports_the_created_task(todoist_api):
    todoist_api()

    with pytest.raises(UnexpectedResult) as excinfo:
        call(create_task, {"content": "Call mum", "due_string": "sometime soonish"})

    assert excinfo.value.event == {"kind": "created", "task": {"id": "900", "content": "Call mum"}}


def test_list_tasks_has_no_event(todoist_api):
    todoist_api(tasks=[])
    assert call(list_tasks, {}).artifact is None


def test_complete_task_reports_the_task_as_it_was(todoist_api):
    todoist_api(tasks=[{"id": "3", "content": "Buy milk", "due": {"date": "2026-09-09"}}])

    result = call(complete_task, {"task": "milk"})

    assert result.artifact == {
        "kind": "completed",
        "task": {"id": "3", "content": "Buy milk", "due": {"date": "2026-09-09"}},
    }


def test_update_task_reports_before_and_after(todoist_api):
    todoist_api(tasks=[{"id": "3", "content": "Buy milk", "due": {"date": "2026-09-09"}}])

    event = call(update_task, {"task": "3", "due_string": "friday 5pm"}).artifact

    assert event["kind"] == "updated"
    assert event["before"]["due"] == {"date": "2026-09-09"}
    assert event["after"]["due"] == {"string": "friday 5pm", "datetime": "2026-09-11T17:00:00"}


def test_a_half_applied_update_carries_no_event(todoist_api):
    todoist_api(tasks=[{"id": "3", "content": "Buy milk"}])

    with pytest.raises(UnexpectedResult) as excinfo:
        call(update_task, {"task": "3", "due_string": "whenever-ish"})

    assert excinfo.value.event is None


def test_delete_task_reports_the_deleted_task(todoist_api):
    todoist_api(tasks=[{"id": "3", "content": "Buy milk"}])

    assert call(delete_task, {"task": "3"}).artifact == {
        "kind": "deleted",
        "task": {"id": "3", "content": "Buy milk"},
    }


# --- set_reminder -----------------------------------------------------------------

TIMED = {"id": "3", "content": "Submit PR", "due": {"date": "2026-09-21T23:59:00"}}
DAY_ONLY = {"id": "4", "content": "Taxes", "due": {"date": "2026-09-21"}}


@pytest.mark.parametrize(
    ("text", "minutes"),
    [
        ("30 minutes", 30),
        ("30 min", 30),
        ("30m", 30),
        ("2 hours", 120),
        ("an hour", 60),
        ("a day", 1440),
        ("1 day before", 1440),
        ("1 week", 10080),
        ("1h30m", 90),
        ("1 hour 30 minutes", 90),
        ("2.5 hours", 150),
    ],
)
def test_parse_offset_reads_plain_durations(text, minutes):
    assert _parse_offset(text) == minutes


@pytest.mark.parametrize("text", ["soon", "", "0 minutes", "a while"])
def test_parse_offset_asks_when_it_cannot_read_the_duration(text):
    with pytest.raises(NeedsClarification, match="how long before"):
        _parse_offset(text)


@pytest.mark.parametrize(
    ("minutes", "text"),
    [
        (30, "30 min"),
        (60, "1 hour"),
        (120, "2 hours"),
        (90, "90 min"),
        (1440, "1 day"),
        (4320, "3 days"),
        (10080, "1 week"),
    ],
)
def test_offset_text(minutes, text):
    assert _offset_text(minutes) == text


def test_set_reminder_needs_exactly_one_of_before_and_at(todoist_api):
    todoist_api(tasks=[TIMED])
    for args in ({"task": "3"}, {"task": "3", "before": "30 minutes", "at": "9am"}):
        with pytest.raises(NeedsClarification, match="exactly one"):
            set_reminder.invoke(args)


def test_set_reminder_before_the_due_time(todoist_api):
    todo = todoist_api(tasks=[TIMED])

    out = set_reminder.invoke({"task": "submit pr", "before": "30 minutes"})

    assert out == (
        "Reminder set on [3] Submit PR (due 2026-09-21T23:59:00): 30 min before "
        "(fires 2026-09-21T23:29:00)."
    )
    assert todo.commands[0]["type"] == "reminder_add"
    assert todo.commands[0]["args"] == {"item_id": "3", "type": "relative", "minute_offset": 30}


def test_set_reminder_at_a_time_in_the_users_words(todoist_api):
    todo = todoist_api(tasks=[TIMED])

    out = set_reminder.invoke({"task": "3", "at": "tomorrow at 3pm"})

    assert out == "Reminder set on [3] Submit PR (due 2026-09-21T23:59:00): at 2026-09-09T15:00:00."
    assert todo.commands[0]["args"] == {
        "item_id": "3",
        "type": "absolute",
        "due": {"string": "tomorrow at 3pm"},
    }


def test_set_reminder_before_needs_a_task_with_a_due_time(todoist_api):
    todo = todoist_api(tasks=[DAY_ONLY])

    with pytest.raises(NeedsClarification, match="no due time"):
        set_reminder.invoke({"task": "taxes", "before": "1 hour"})

    assert todo.commands == []


def test_set_reminder_at_works_on_a_task_without_a_time(todoist_api):
    todoist_api(tasks=[DAY_ONLY])

    out = set_reminder.invoke({"task": "taxes", "at": "next monday"})

    assert out == "Reminder set on [4] Taxes (due 2026-09-21): at 2026-09-14."


def test_set_reminder_does_not_duplicate_an_existing_one(todoist_api):
    todo = todoist_api(tasks=[TIMED])
    set_reminder.invoke({"task": "3", "before": "30 minutes"})
    set_reminder.invoke({"task": "3", "before": "1 day"})

    out = set_reminder.invoke({"task": "3", "before": "30 min"})

    assert out == (
        "[3] Submit PR (due 2026-09-21T23:59:00) already has a reminder 30 min before "
        "(fires 2026-09-21T23:29:00). Nothing added. Other reminders on it: 1 day before "
        "(fires 2026-09-20T23:59:00)."
    )
    assert len(todo.commands) == 2, "no third reminder_add"


def test_set_reminder_mentions_the_reminders_already_there(todoist_api):
    todoist_api(tasks=[TIMED])
    set_reminder.invoke({"task": "3", "before": "1 day"})

    out = set_reminder.invoke({"task": "3", "before": "30 minutes"})

    assert out.endswith("Other reminders on it: 1 day before (fires 2026-09-20T23:59:00).")


def test_set_reminder_undoes_a_reminder_todoist_could_not_place_in_time(todoist_api):
    todo = todoist_api(tasks=[TIMED])

    with pytest.raises(UnexpectedResult) as excinfo:
        set_reminder.invoke({"task": "3", "at": "sometime soonish"})

    assert "did not understand the reminder time 'sometime soonish'" in str(excinfo.value)
    assert "simpler phrase" in excinfo.value.repair
    assert [c["type"] for c in todo.commands] == ["reminder_add", "reminder_delete"]
    assert todo.reminders[0]["is_deleted"] is True, "no dud left on the task"


def test_set_reminder_reports_a_reminder_that_did_not_land(todoist_api, monkeypatch):
    todoist_api(tasks=[TIMED])
    monkeypatch.setattr(todoist, "add_reminder", lambda *a, **k: None)

    with pytest.raises(UnexpectedResult) as excinfo:
        set_reminder.invoke({"task": "3", "before": "1 hour"})

    assert "not on the task" in str(excinfo.value)
    assert "Do not call set_reminder again" in excinfo.value.repair


def test_set_reminder_is_ambiguous_like_the_other_tools(todoist_api):
    todoist_api(tasks=TASKS)

    with pytest.raises(NeedsClarification, match="matches 2 open tasks"):
        set_reminder.invoke({"task": "review", "before": "1 hour"})


def test_set_reminder_propagates_a_rejected_command(todoist_api):
    todoist_api(
        tasks=[TIMED],
        reminder_error={"error": "Premium required", "error_code": 1, "http_code": 403},
    )

    with pytest.raises(todoist.TodoistError, match="Premium required") as excinfo:
        set_reminder.invoke({"task": "3", "before": "1 hour"})

    assert excinfo.value.status_code == 403


def test_set_reminder_has_no_event_for_memory(todoist_api):
    todoist_api(tasks=[TIMED])
    assert call(set_reminder, {"task": "3", "before": "1 hour"}).artifact is None

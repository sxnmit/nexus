"""Tests for the LangGraph agent loop: routing, the two nodes, and whole runs."""

from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END

import agent
import config
from tests.support import ScriptedModel, tool_call, tool_calls

TWO_REVIEWS = [
    {"id": "1", "content": "Submit iTrade PR review"},
    {"id": "2", "content": "Review the design doc"},
]


# --- The graph itself ---------------------------------------------------------


def test_graph_has_exactly_the_documented_shape():
    graph = agent.build_graph(ScriptedModel()).get_graph()

    assert set(graph.nodes) == {"__start__", "agent", "tools", "__end__"}
    assert {(e.source, e.target) for e in graph.edges} == {
        ("__start__", "agent"),
        ("agent", "tools"),
        ("agent", "__end__"),
        ("tools", "agent"),
    }


def test_build_graph_binds_the_real_model_to_the_three_tools_by_default():
    model = agent._build_model()

    assert model.bound.model == config.MODEL
    assert [t["name"] for t in model.kwargs["tools"]] == [
        "create_task",
        "list_tasks",
        "complete_task",
    ]
    assert agent.build_graph() is not None


# --- _route: the conditional edge --------------------------------------------


def test_route_ends_on_a_plain_text_reply():
    state = {"messages": [HumanMessage("hi"), AIMessage("hello")], "steps": 1}
    assert agent._route(state) == END


def test_route_goes_to_tools_when_the_model_asks_for_them():
    state = {"messages": [tool_call("list_tasks", {})], "steps": 1}
    assert agent._route(state) == "tools"


def test_route_stops_at_the_loop_cap_even_with_tool_calls_pending():
    state = {"messages": [tool_call("list_tasks", {})], "steps": config.MAX_TOOL_LOOPS}
    assert agent._route(state) == END


def test_route_treats_a_missing_step_count_as_zero():
    assert agent._route({"messages": [tool_call("list_tasks", {})]}) == "tools"


# --- _agent_node: plan --------------------------------------------------------


def test_agent_node_prepends_the_system_prompt_and_counts_the_step():
    model = ScriptedModel(AIMessage("hi"))

    out = agent._agent_node({"messages": [HumanMessage("hello")], "steps": 2}, model)

    sent = model.seen[0]
    assert isinstance(sent[0], SystemMessage)
    assert sent[0].content.startswith("You are Nexus")
    assert sent[1:] == [HumanMessage("hello")]
    assert out == {"messages": [AIMessage("hi")], "steps": 3}


def test_system_prompt_carries_todays_date():
    assert datetime.now().astimezone().strftime("%d %B %Y") in agent._system_prompt()


def test_system_prompt_spells_out_the_error_handling_contract():
    prompt = agent._system_prompt()
    assert "never tell the user something worked when the tool said it did not" in prompt
    assert "clarifying question" in prompt


# --- _tool_node: act + observe -------------------------------------------------


def test_tool_node_returns_one_observation_per_call_with_matching_ids(todoist_api):
    todoist_api(tasks=[{"id": "1", "content": "Buy milk"}])
    state = {"messages": [tool_calls(("list_tasks", {}), ("complete_task", {"task": "milk"}))]}

    out = agent._tool_node(state)["messages"]

    assert all(isinstance(m, ToolMessage) for m in out)
    assert [m.tool_call_id for m in out] == ["call-0", "call-1"]
    assert [m.name for m in out] == ["list_tasks", "complete_task"]
    assert [m.status for m in out] == ["success", "success"]
    assert out[1].content == "Completed [1] Buy milk"


def test_tool_node_turns_an_exception_into_an_error_observation(todoist_api):
    todoist_api(fail="Todoist rejected the request: HTTP 500 - boom")

    out = agent._tool_node({"messages": [tool_call("list_tasks", {})]})["messages"]

    assert out[0].status == "error"
    assert out[0].content == "TodoistError: Todoist rejected the request: HTTP 500 - boom"
    assert out[0].tool_call_id == "call-1"


def test_tool_node_reports_an_unknown_tool_instead_of_crashing():
    out = agent._tool_node({"messages": [tool_call("nuke_everything", {})]})["messages"]

    assert out[0].status == "error"
    assert "no tool called 'nuke_everything'" in out[0].content


def test_tool_node_keeps_going_after_one_call_fails(todoist_api):
    todoist_api(tasks=[])
    state = {"messages": [tool_calls(("complete_task", {"task": "x"}), ("list_tasks", {}))]}

    out = agent._tool_node(state)["messages"]

    assert [m.status for m in out] == ["error", "success"]


# --- Whole runs: the happy paths ----------------------------------------------


def test_the_success_criterion_create_flow(ask, todoist_api):
    todo = todoist_api()
    model = ScriptedModel(
        tool_call(
            "create_task", {"content": "Submit iTrade PR review", "due_string": "tomorrow at 3pm"}
        ),
        AIMessage("Added 'Submit iTrade PR review', due tomorrow at 3pm."),
    )

    reply = ask(model, "remind me to submit the iTrade PR review tomorrow at 3pm")

    assert reply == "Added 'Submit iTrade PR review', due tomorrow at 3pm."
    assert todo.created == [{"content": "Submit iTrade PR review", "due_string": "tomorrow at 3pm"}]
    observation = model.observations()[0]
    assert observation.status == "success"
    assert "Submit iTrade PR review" in observation.content


def test_a_plain_answer_never_touches_todoist(ask, todoist_api):
    todo = todoist_api(fail="must not be called")
    model = ScriptedModel(AIMessage("I manage your Todoist - ask me to add or list a task."))

    reply = ask(model, "who are you?")

    assert reply.startswith("I manage your Todoist")
    assert model.calls == 1
    assert todo.calls == []


def test_state_accumulates_the_whole_exchange_but_never_the_system_prompt(todoist_api):
    todoist_api()
    model = ScriptedModel(tool_call("list_tasks", {}), AIMessage("Nothing open."))

    final = agent.build_graph(model).invoke({"messages": [HumanMessage("list")], "steps": 0})

    assert [type(m).__name__ for m in final["messages"]] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]
    assert final["steps"] == 2
    assert not any(isinstance(m, SystemMessage) for m in final["messages"])


def test_the_system_prompt_is_sent_exactly_once_per_model_call(ask, todoist_api):
    todoist_api()
    model = ScriptedModel(
        tool_call("list_tasks", {}), tool_call("list_tasks", {}, "call-2"), AIMessage("ok")
    )

    ask(model, "list twice")

    assert model.calls == 3
    for turn in model.seen:
        assert sum(isinstance(m, SystemMessage) for m in turn) == 1
        assert isinstance(turn[0], SystemMessage)


# --- Whole runs: the error paths ----------------------------------------------


def test_the_model_can_recover_from_a_failed_call_by_retrying(ask, todoist_api):
    """Round 1 is ambiguous, round 2 uses the id the error showed it."""
    todo = todoist_api(tasks=TWO_REVIEWS)
    model = ScriptedModel(
        tool_call("complete_task", {"task": "review"}),
        tool_call("complete_task", {"task": "1"}, "call-2"),
        AIMessage("Done - completed 'Submit iTrade PR review'."),
    )

    reply = ask(model, "mark the PR review done")

    assert [o.status for o in model.observations(turn=1)] == ["error"]
    assert [o.status for o in model.observations(turn=2)] == ["error", "success"]
    assert todo.closed == ["1"]
    assert reply.startswith("Done")


def test_an_ambiguous_match_lets_the_model_ask_instead_of_guessing(ask, todoist_api):
    todo = todoist_api(tasks=TWO_REVIEWS)
    model = ScriptedModel(
        tool_call("complete_task", {"task": "review"}),
        AIMessage("Which one - the iTrade PR review or the design doc review?"),
    )

    reply = ask(model, "mark the review done")

    observation = model.observations()[0]
    assert observation.status == "error"
    assert "Submit iTrade PR review" in observation.content
    assert "Review the design doc" in observation.content
    assert todo.closed == []
    assert reply.startswith("Which one")


def test_a_todoist_outage_reaches_the_model_as_an_error_not_a_crash(ask, todoist_api):
    todoist_api(fail="Todoist rejected the request: HTTP 401 - unauthorised")
    model = ScriptedModel(
        tool_call("create_task", {"content": "Buy milk", "due_string": ""}),
        AIMessage("Todoist rejected that - your API token looks wrong."),
    )

    reply = ask(model, "add buy milk")

    observation = model.observations()[0]
    assert observation.status == "error"
    assert "401" in observation.content
    assert reply.startswith("Todoist rejected that")


def test_an_unparsed_due_date_is_reported_rather_than_confirmed(ask, todoist_api):
    todoist_api()
    model = ScriptedModel(
        tool_call("create_task", {"content": "Call mum", "due_string": "sometime soonish"}),
        AIMessage("Added it, but Todoist couldn't read that date - when did you mean?"),
    )

    ask(model, "remind me to call mum sometime soonish")

    assert "could not understand the due date 'sometime soonish'" in model.observations()[0].content


def test_the_loop_is_capped_and_the_user_is_told(ask, todoist_api):
    todoist_api()
    model = ScriptedModel(
        *[tool_call("list_tasks", {}, f"call-{i}") for i in range(config.MAX_TOOL_LOOPS + 5)]
    )

    reply = ask(model, "loop forever")

    assert model.calls == config.MAX_TOOL_LOOPS
    assert "got stuck" in reply


def test_the_loop_cap_is_configurable(ask, todoist_api, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_LOOPS", 2)
    todoist_api()
    model = ScriptedModel(*[tool_call("list_tasks", {}, f"call-{i}") for i in range(10)])

    ask(model, "loop forever")

    assert model.calls == 2


# --- run(): turning the final state into a Telegram reply ---------------------


def test_run_strips_surrounding_whitespace(ask):
    assert ask(ScriptedModel(AIMessage("  hi  \n")), "x") == "hi"


def test_run_extracts_text_from_content_blocks(ask):
    """With thinking on, Anthropic returns a list of blocks rather than a string."""
    model = ScriptedModel(
        AIMessage(
            content=[
                {"type": "thinking", "thinking": "...", "signature": "sig"},
                {"type": "text", "text": "Added it."},
            ]
        )
    )

    assert ask(model, "x") == "Added it."

"""Tests for the LangGraph agent loop: routing, the nodes, and whole runs.

The observe/replan step has its own file, tests/test_replan.py."""

from datetime import datetime, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END

import agent
import config
from tests.support import ScriptedModel, log_rows, tool_call, tool_calls

TWO_REVIEWS = [
    {"id": "1", "content": "Submit iTrade PR review"},
    {"id": "2", "content": "Review the design doc"},
]


# --- The graph itself ---------------------------------------------------------


def test_graph_has_exactly_the_documented_shape():
    graph = agent.build_graph(ScriptedModel()).get_graph()

    assert set(graph.nodes) == {"__start__", "agent", "tools", "observe", "__end__"}
    assert {(e.source, e.target) for e in graph.edges} == {
        ("__start__", "agent"),
        ("agent", "tools"),
        ("agent", "__end__"),
        ("tools", "observe"),
        ("observe", "agent"),
    }


def test_build_llm_sends_no_workspace_header_by_default():
    assert agent.build_llm().default_headers is None


def test_build_llm_sends_the_workspace_header_for_a_personal_access_token(monkeypatch):
    monkeypatch.setattr(agent.config, "ANTHROPIC_WORKSPACE_ID", "wrkspc_01TEST")

    assert agent.build_llm().default_headers == {"anthropic-workspace-id": "wrkspc_01TEST"}


def test_check_model_makes_exactly_one_tiny_request(monkeypatch):
    calls = []

    class Stub:
        def get_num_tokens_from_messages(self, messages):
            calls.append(messages)
            return 1

    monkeypatch.setattr(agent, "build_llm", lambda: Stub())

    agent.check_model()

    assert len(calls) == 1 and len(calls[0]) == 1


def test_build_graph_binds_the_real_model_to_the_six_tools_by_default():
    model = agent._build_model()

    assert model.bound.model == config.MODEL
    assert [t["name"] for t in model.kwargs["tools"]] == [
        "create_task",
        "list_tasks",
        "complete_task",
        "update_task",
        "delete_task",
        "set_reminder",
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
    scripted = AIMessage("hi")
    model = ScriptedModel(scripted)

    out = agent._agent_node({"messages": [HumanMessage("hello")], "steps": 2}, model, model)

    sent = model.seen[0]
    assert isinstance(sent[0], SystemMessage)
    assert sent[0].content.startswith("You are Nexus")
    assert sent[1:] == [HumanMessage("hello")]
    assert (out["steps"], out["mode"]) == (3, "act")
    [reply] = out["messages"]
    assert reply.content == "hi"
    assert reply.response_metadata["latency_ms"] >= 0, "how long the call took, for the record"
    assert scripted.response_metadata == {}, "recorded on a copy; the model's object is untouched"


def test_agent_node_picks_the_model_by_mode_and_says_so_in_the_prompt():
    acting, responding = ScriptedModel(AIMessage("act")), ScriptedModel(AIMessage("respond"))
    state = {"messages": [HumanMessage("x")], "steps": 0}

    agent._agent_node({**state, "mode": "retry"}, acting, responding)
    agent._agent_node({**state, "mode": "respond"}, acting, responding)

    assert acting.calls == 1 and responding.calls == 1
    assert "follow its repair note" in acting.seen[0][0].content.lower()
    assert "cannot use tools" in responding.seen[0][0].content


def test_system_prompt_carries_todays_date():
    assert datetime.now().astimezone().strftime("%d %B %Y") in agent._system_prompt()


def test_system_prompt_spells_out_the_error_handling_contract():
    prompt = agent._system_prompt().lower()
    assert "never tell the user something worked when the tool said it did not" in prompt
    assert "clarifying question" in prompt
    assert "observer:" in prompt, "the model must be told what the verdict line is"


def test_system_prompt_tells_the_model_when_a_reminder_is_really_a_task():
    prompt = agent._system_prompt()
    assert "set_reminder" in prompt
    assert '"Remind me to X tomorrow at 3pm" is a task' in prompt
    assert "Todoist sends the notification, not you" in prompt


def test_system_prompt_explains_the_conversation_memory():
    prompt = agent._system_prompt()
    assert "recent conversation" in prompt
    assert agent.CHECK_IN_MARKER in prompt, "the model must know what the stand-in line means"


def test_system_prompt_carries_the_memory_note():
    assert "habits" not in agent._system_prompt()
    assert agent._system_prompt(memory_note="\n\nhabits: x").endswith("habits: x")
    assert agent._system_prompt("respond", "\n\nhabits: x").endswith(
        "Do not claim anything was done."
    )


# --- _tool_node: act ------------------------------------------------------------


def test_tool_node_returns_one_observation_per_call_with_matching_ids(todoist_api):
    todoist_api(tasks=[{"id": "1", "content": "Buy milk"}])
    state = {"messages": [tool_calls(("list_tasks", {}), ("complete_task", {"task": "milk"}))]}

    out = agent._tool_node(state)["messages"]

    assert all(isinstance(m, ToolMessage) for m in out)
    assert [m.tool_call_id for m in out] == ["call-0", "call-1"]
    assert [m.name for m in out] == ["list_tasks", "complete_task"]
    assert [m.status for m in out] == ["success", "success"]
    assert out[1].content == "Completed [1] Buy milk"
    assert out[0].artifact == {"outcome": "ok", "event": None}
    assert out[1].artifact == {
        "outcome": "ok",
        "event": {"kind": "completed", "task": {"id": "1", "content": "Buy milk"}},
    }, "the event rides in the artifact, which the model never sees"


def test_tool_node_turns_an_exception_into_an_error_observation(todoist_api):
    todoist_api(fail="Todoist rejected the request: HTTP 500 - boom", fail_status=500)

    out = agent._tool_node({"messages": [tool_call("list_tasks", {})]})["messages"]

    assert out[0].status == "error"
    assert out[0].content == "Todoist rejected the request: HTTP 500 - boom"
    assert out[0].tool_call_id == "call-1"
    assert out[0].artifact == {"outcome": "api_error", "status_code": 500}


def test_tool_node_records_how_long_each_call_took(todoist_api):
    todoist_api(tasks=[])

    out = agent._tool_node({"messages": [tool_call("list_tasks", {})]})["messages"]

    assert out[0].response_metadata == {"latency_ms": out[0].response_metadata["latency_ms"]}
    assert out[0].response_metadata["latency_ms"] >= 0


def test_tool_node_reports_an_unknown_tool_instead_of_crashing():
    out = agent._tool_node({"messages": [tool_call("nuke_everything", {})]})["messages"]

    assert out[0].status == "error"
    assert "no tool called 'nuke_everything'" in out[0].content


def test_tool_node_keeps_going_after_one_call_fails(todoist_api):
    todoist_api(tasks=[])
    state = {"messages": [tool_calls(("complete_task", {"task": "x"}), ("list_tasks", {}))]}

    out = agent._tool_node(state)["messages"]

    assert [m.status for m in out] == ["error", "success"]
    assert [m.artifact["outcome"] for m in out] == ["clarify", "ok"]


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
    assert "Observer: only the user can resolve this" in observation.content
    assert model.tools_bound == [True, False], "the asking turn must not be able to call tools"
    assert todo.closed == []
    assert reply.startswith("Which one")


def test_a_rejected_token_reaches_the_model_as_a_fail_verdict_not_a_crash(ask, todoist_api):
    todoist_api(fail="Todoist rejected the request: HTTP 401 - unauthorised", fail_status=401)
    model = ScriptedModel(
        tool_call("create_task", {"content": "Buy milk", "due_string": ""}),
        AIMessage("Todoist rejected that - your API token looks wrong."),
    )

    reply = ask(model, "add buy milk")

    observation = model.observations()[0]
    assert observation.status == "error"
    assert "401" in observation.content
    assert "Observer: Todoist refused our credentials" in observation.content
    assert model.tools_bound == [True, False], "nothing to retry, so no tools on the reply turn"
    assert reply.startswith("Todoist rejected that")


def test_an_unparsed_due_date_is_reported_rather_than_confirmed(ask, todoist_api):
    todoist_api()
    model = ScriptedModel(
        tool_call("create_task", {"content": "Call mum", "due_string": "sometime soonish"}),
        AIMessage("Added it, but Todoist couldn't read that date - when did you mean?"),
    )

    ask(model, "remind me to call mum sometime soonish")

    observation = model.observations()[0]
    assert "did not understand the due date 'sometime soonish'" in observation.content
    assert "Observer: RETRY once" in observation.content
    assert "do not call create_task again" in observation.content


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
    """A thinking model (NEXUS_MODEL=claude-opus-5) returns content blocks, not a string."""
    model = ScriptedModel(
        AIMessage(
            content=[
                {"type": "thinking", "thinking": "...", "signature": "sig"},
                {"type": "text", "text": "Added it."},
            ]
        )
    )

    assert ask(model, "x") == "Added it."


# --- Memory around the graph --------------------------------------------------


def shape(messages):
    """(type, content) pairs -- messages that went through the graph carry ids."""
    return [(type(m).__name__, m.content) for m in messages]


def test_run_replays_the_recent_conversation_before_the_new_message(ask, memory):
    memory.log(7, "user", "add gym tomorrow")
    memory.log(7, "assistant", "Added 'Gym', due tomorrow.", kind="reply")
    model = ScriptedModel(AIMessage("Moved it to friday."))

    ask(model, "actually make it friday")

    seen = model.seen[0]
    assert isinstance(seen[0], SystemMessage)
    assert shape(seen[1:]) == [
        ("HumanMessage", "add gym tomorrow"),
        ("AIMessage", "Added 'Gym', due tomorrow."),
        ("HumanMessage", "actually make it friday"),
    ]


def test_run_puts_a_user_turn_in_front_of_a_leading_check_in(ask, memory):
    memory.log(7, "assistant", "Good morning. Due today (1):\n- Gym (2:00pm)", kind="morning")
    model = ScriptedModel(AIMessage("Pushed 'Gym' to tomorrow."))

    ask(model, "push it to tomorrow")

    seen = shape(model.seen[0][1:])
    assert seen[0] == ("HumanMessage", agent.CHECK_IN_MARKER), (
        "the first message must be the user's"
    )
    assert seen[1][0] == "AIMessage" and seen[1][1].startswith("Good morning")
    assert seen[2] == ("HumanMessage", "push it to tomorrow")


def test_run_keeps_conversations_apart_by_chat(ask, memory):
    memory.log(8, "user", "someone else's message")
    model = ScriptedModel(AIMessage("hi"))

    ask(model, "hello", chat_id=7)

    assert shape(model.seen[0][1:]) == [("HumanMessage", "hello")]


def test_run_logs_the_exchange_and_learns_from_the_tool_event(ask, memory, todoist_api):
    todoist_api()
    model = ScriptedModel(
        tool_call("create_task", {"content": "Gym", "due_string": "tomorrow at 3pm"}),
        AIMessage("Added 'Gym', due tomorrow at 3pm."),
    )

    ask(model, "add gym tomorrow at 3pm")

    assert [(row["role"], row["kind"]) for row in log_rows(memory)] == [
        ("user", "message"),
        ("tool", "create_task"),
        ("assistant", "reply"),
    ]
    assert memory.recent(7) == [
        ("user", "add gym tomorrow at 3pm"),
        ("assistant", "Added 'Gym', due tomorrow at 3pm."),
    ]
    gym = memory.patterns()[0]
    assert (gym.topic, gym.created) == ("gym", 1)


def test_run_logs_a_tool_call_with_its_arguments_and_outcome(ask, memory, todoist_api):
    todoist_api(tasks=[])
    model = ScriptedModel(tool_call("complete_task", {"task": "x"}), AIMessage("Nothing to do."))

    ask(model, "mark x done")

    row = log_rows(memory)[1]
    assert row["kind"] == "complete_task"
    assert row["text"].startswith("There are no open tasks")
    assert "Observer:" in row["text"], "the log keeps the tool result as the model saw it"
    assert row["detail"] == {"args": {"task": "x"}, "outcome": "clarify"}


def test_run_learns_from_a_half_success_but_not_from_its_repair(ask, memory, todoist_api):
    """A task created with a dropped due date is still a task created -- and the
    repair that gives it a date is not a reschedule."""
    todoist_api()
    model = ScriptedModel(
        tool_call("create_task", {"content": "Gym", "due_string": "soonish"}),
        tool_call("update_task", {"task": "900", "due_string": "tomorrow at 3pm"}, "call-2"),
        AIMessage("Added 'Gym', due tomorrow at 3pm."),
    )

    ask(model, "add gym soonish")

    gym = memory.patterns()[0]
    assert (gym.created, gym.rescheduled) == (1, 0)


def test_run_puts_the_relevant_habits_in_the_system_prompt(ask, memory):
    for _ in range(3):
        memory.learn({"kind": "created", "task": {"id": "1", "content": "Gym"}})
    model = ScriptedModel(AIMessage("ok"))

    ask(model, "add gym tomorrow")

    prompt = model.seen[0][0].content
    assert "'Gym': added 3 times (recurring)" in prompt
    assert "Never nag" in prompt


def test_run_leaves_the_prompt_alone_when_nothing_is_known(ask, memory):
    model = ScriptedModel(AIMessage("ok"))

    ask(model, "add gym tomorrow")

    assert "habits" not in model.seen[0][0].content


def test_run_looks_for_habits_in_the_earlier_messages_too(ask, memory):
    for _ in range(3):
        memory.learn({"kind": "created", "task": {"id": "1", "content": "Gym"}})
    memory.log(7, "user", "add gym tomorrow")
    memory.log(7, "assistant", "Added.", kind="reply")
    model = ScriptedModel(AIMessage("ok"))

    ask(model, "actually make it friday")

    assert "'Gym'" in model.seen[0][0].content


def test_run_logs_the_stuck_reply_too(ask, memory, todoist_api):
    todoist_api()
    model = ScriptedModel(
        *[tool_call("list_tasks", {}, f"call-{i}") for i in range(config.MAX_TOOL_LOOPS + 5)]
    )

    reply = ask(model, "loop forever")

    assert memory.recent(7)[-1] == ("assistant", reply)


def test_history_conversion():
    assert agent._history([]) == []
    assert agent._history([("user", "a"), ("assistant", "b")]) == [
        HumanMessage("a"),
        AIMessage("b"),
    ]
    assert agent._history([("assistant", "b")]) == [
        HumanMessage(agent.CHECK_IN_MARKER),
        AIMessage("b"),
    ]


# --- The record ---------------------------------------------------------------


def usage(input_tokens, output_tokens, cache_read=0):
    """usage_metadata the way langchain-anthropic fills it in."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_details": {"cache_read": cache_read},
    }


def costed(message, input_tokens, output_tokens, cache_read=0, **metadata):
    """A scripted reply with token usage and response metadata attached."""
    return message.model_copy(
        update={
            "usage_metadata": usage(input_tokens, output_tokens, cache_read),
            "response_metadata": metadata,
        }
    )


def latest_run(memory):
    [run] = memory.runs(memory.now() - timedelta(days=1))
    return run


class ExplodingModel:
    """A model whose every call raises -- an outage, an invalid key."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        raise RuntimeError("boom")


def test_run_records_what_happened_and_what_it_cost(ask, memory, todoist_api):
    todoist_api(tasks=[{"id": "1", "content": "Buy milk"}])
    model = ScriptedModel(
        costed(tool_call("list_tasks", {}), 500, 20, 100, model="claude-haiku-4-5-20251001"),
        costed(AIMessage("One task: Buy milk."), 600, 15, stop_reason="end_turn"),
    )

    ask(model, "what's on my list?")

    run = latest_run(memory)
    assert (run.kind, run.chat_id, run.trigger) == ("message", 7, "what's on my list?")
    assert (run.reply, run.outcome, run.error) == ("One task: Buy milk.", "ok", "")
    assert (run.steps, run.retries, run.calls, run.tool_calls) == (2, 0, 2, 1)
    assert (run.input_tokens, run.output_tokens, run.cache_read_tokens) == (1100, 35, 100)
    assert run.latency_ms >= 0
    assert run.model == "claude-haiku-4-5-20251001", "the model that answered, per the response"
    assert (run.prompt, run.build) == (agent.PROMPT_VERSION, config.COMMIT)


def test_run_falls_back_to_the_configured_model_name(ask, memory):
    ask(ScriptedModel(AIMessage("hi")), "hello")

    run = latest_run(memory)
    assert run.model == config.MODEL
    assert (run.input_tokens, run.output_tokens, run.calls) == (0, 0, 1), "no usage reported"


def test_run_traces_what_the_model_saw_and_did(ask, memory, todoist_api):
    todoist_api(tasks=[{"id": "1", "content": "Buy milk"}])
    memory.log(7, "assistant", "Good morning. Nothing due today.", kind="morning")
    for _ in range(3):
        memory.learn({"kind": "created", "task": {"id": "9", "content": "Gym"}})
    model = ScriptedModel(
        costed(tool_call("list_tasks", {}), 500, 20, stop_reason="tool_use"),
        AIMessage("One task: Buy milk."),
    )

    ask(model, "what's on my list, and gym?")

    trace = latest_run(memory).trace
    assert trace["history"] == [{"role": "assistant", "text": "Good morning. Nothing due today."}]
    assert trace["memory_note"].startswith("\n\nWhat you know about this user's habits")
    assert trace["user"] == "what's on my list, and gym?"
    assert trace["reply"] == "One task: Buy milk."
    planned, observed, answered = trace["steps"]
    assert planned == {
        "role": "assistant",
        "text": "",
        "tool_calls": [{"id": "call-1", "name": "list_tasks", "args": {}}],
        "usage": {"input": 500, "output": 20, "cache_read": 0},
        "latency_ms": planned["latency_ms"],
        "stop_reason": "tool_use",
    }
    assert observed == {
        "role": "tool",
        "name": "list_tasks",
        "call_id": "call-1",
        "text": "Open tasks:\n[1] Buy milk",
        "outcome": "ok",
        "status_code": None,
        "verdict": "done",
        "latency_ms": observed["latency_ms"],
    }
    assert answered["text"] == "One task: Buy milk." and answered["tool_calls"] == []
    assert answered["usage"] == {"input": 0, "output": 0, "cache_read": 0}


def test_run_records_the_observers_verdicts(ask, memory, todoist_api):
    todoist_api(tasks=[], fail="HTTP 503 - down", fail_status=503, fail_times=1)
    model = ScriptedModel(
        tool_call("list_tasks", {}),
        tool_call("list_tasks", {}, "call-2"),
        AIMessage("Nothing on the list."),
    )

    ask(model, "list")

    run = latest_run(memory)
    observations = [step for step in run.trace["steps"] if step["role"] == "tool"]
    assert [(o["outcome"], o["status_code"], o["verdict"]) for o in observations] == [
        ("api_error", 503, "retry"),
        ("ok", None, "done"),
    ]
    assert (run.retries, run.outcome) == (1, "ok"), "a retry that worked is still a clean run"


@pytest.mark.parametrize(
    ("fake", "expected"),
    [
        ({"tasks": []}, "asked"),  # nothing to complete: only the user can resolve it
        ({"fail": "HTTP 401 - bad token", "fail_status": 401}, "failed"),
    ],
)
def test_run_outcome_follows_the_verdict(ask, memory, todoist_api, fake, expected):
    todoist_api(**fake)
    model = ScriptedModel(tool_call("complete_task", {"task": "x"}), AIMessage("Sorry."))

    ask(model, "mark x done")

    assert latest_run(memory).outcome == expected


def test_run_records_a_stuck_loop(ask, memory, todoist_api, monkeypatch):
    todoist_api(tasks=[])
    monkeypatch.setattr(config, "MAX_TOOL_LOOPS", 1)
    model = ScriptedModel(tool_call("list_tasks", {}), tool_call("list_tasks", {}, "call-2"))

    reply = ask(model, "list")

    assert reply == agent.STUCK_REPLY
    run = latest_run(memory)
    assert (run.outcome, run.reply, run.steps) == ("stuck", agent.STUCK_REPLY, 1)


def test_run_records_a_crash_then_re_raises(ask, memory):
    with pytest.raises(RuntimeError, match="boom"):
        ask(ExplodingModel(), "hello")

    run = latest_run(memory)
    assert (run.outcome, run.reply, run.error) == ("crashed", "", "RuntimeError: boom")
    assert (run.steps, run.calls, run.tool_calls) == (0, 0, 0)
    assert run.trace["user"] == "hello" and run.trace["steps"] == []
    assert [(row["role"], row["run_id"]) for row in log_rows(memory)] == [("user", run.id)], (
        "the message is logged with the run; no reply was made"
    )


def test_run_tags_every_log_row_with_the_run_id(ask, memory, todoist_api):
    todoist_api(tasks=[{"id": "1", "content": "Buy milk"}])
    model = ScriptedModel(tool_call("complete_task", {"task": "milk"}), AIMessage("Done."))

    ask(model, "milk is done")

    run = latest_run(memory)
    assert len(run.id) == 12
    assert [(row["role"], row["run_id"]) for row in log_rows(memory)] == [
        ("user", run.id),
        ("tool", run.id),
        ("assistant", run.id),
    ]


def test_prompt_version_is_a_short_hash_of_the_prompt_and_the_tools(monkeypatch):
    assert len(agent.PROMPT_VERSION) == 12 and int(agent.PROMPT_VERSION, 16) >= 0
    assert agent._prompt_version() == agent.PROMPT_VERSION, "deterministic"

    monkeypatch.setattr(agent, "SYSTEM_PROMPT", agent.SYSTEM_PROMPT + " Be brief.")
    assert agent._prompt_version() != agent.PROMPT_VERSION, "a prompt edit is a new version"


def test_observe_keeps_the_verdict_with_the_observation():
    observation = ToolMessage(
        content="boom",
        tool_call_id="call-1",
        name="create_task",
        id="m1",
        status="error",
        artifact={"outcome": "clarify"},
        response_metadata={"latency_ms": 5},
    )

    out = agent._observe_node({"messages": [HumanMessage("x"), observation], "retries": 0})

    [rewritten] = out["messages"]
    assert rewritten.response_metadata == {"latency_ms": 5, "verdict": "ask"}
    assert agent._verdict(rewritten) == "ask"
    assert agent._verdict(observation) == "fail", "an unrewritten failure reads as fail"

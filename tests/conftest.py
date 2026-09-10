"""Shared pytest fixtures. The doubles themselves live in tests/support.py."""

import pytest

import agent
import config
import todoist
from memory import Memory
from tests.support import FakeTodoist, HTTPRecorder


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """A transient-error retry pauses for RETRY_BACKOFF_SECONDS. Not in tests."""
    monkeypatch.setattr(config, "RETRY_BACKOFF_SECONDS", 0)


@pytest.fixture
def todoist_api(monkeypatch):
    """Factory: install an in-memory Todoist and hand it back for assertions."""

    def install(**kwargs):
        fake = FakeTodoist(**kwargs)
        monkeypatch.setattr(todoist, "_request", fake)
        return fake

    return install


@pytest.fixture
def http(monkeypatch):
    """Record the HTTP calls todoist._request actually makes."""
    recorder = HTTPRecorder()
    monkeypatch.setattr(todoist.httpx, "request", recorder)
    return recorder


@pytest.fixture
def memory():
    """A fresh in-RAM memory, gone when the test ends."""
    return Memory(":memory:")


@pytest.fixture
def ask(memory):
    """Run one user message through a graph built around a scripted model, with
    the test's memory around it (request the `memory` fixture to inspect it)."""

    def run_one(model, text, chat_id=7):
        return agent.run(agent.build_graph(model), text, memory, chat_id)

    return run_one

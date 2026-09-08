"""Shared pytest fixtures. The doubles themselves live in tests/support.py."""

import pytest

import agent
import config
import todoist
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
def ask():
    """Run one user message through a graph built around a scripted model."""

    def run_one(model, text):
        return agent.run(agent.build_graph(model), text)

    return run_one

"""Shared fixtures for the in-memory MCP tests (spec §26)."""

import pytest

from annealbridge.interfaces.mcp import server
from annealbridge.orchestration import ExecutionPolicy


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _inject_default_state():
    # Explicitly inject the default policy so stray ANNEALBRIDGE_* environment
    # variables on a developer machine cannot change test behaviour.
    server.reset_state(server.build_state_from_policy(ExecutionPolicy()))
    yield
    server.reset_state(None)

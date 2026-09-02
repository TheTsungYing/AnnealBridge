"""Shared fixtures for the in-memory MCP tests (spec §26)."""

import json
from pathlib import Path

import pytest

from annealbridge.interfaces.mcp import server
from annealbridge.orchestration import ExecutionPolicy

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


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


def load_example(name: str, **solver_overrides) -> dict:
    """Load an example problem as a call_tool payload dict."""
    data = json.loads((EXAMPLES_DIR / name).read_text())
    data["solver"] = {**data.get("solver", {}), **solver_overrides}
    return data

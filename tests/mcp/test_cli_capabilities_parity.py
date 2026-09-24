"""``annealbridge capabilities --json`` is what the MCP tool returns.

A core-only user reads the CLI; an agent reads ``get_optimization_capabilities``
over MCP. Both come from ``build_capabilities`` without the schema, so the
two payloads must be equal field for field and value for value.
"""

import json
import os

import pytest
from mcp import Client
from typer.testing import CliRunner

from annealbridge.interfaces.cli.main import app
from annealbridge.interfaces.mcp import mcp

pytestmark = pytest.mark.anyio

runner = CliRunner()


async def test_capabilities_json_equals_the_mcp_tool_default(monkeypatch):
    # The MCP fixture injects the default policy; give the CLI's
    # environment-built state the same one.
    for key in list(os.environ):
        if key.startswith("ANNEALBRIDGE_"):
            monkeypatch.delenv(key)

    async with Client(mcp) as client:
        result = await client.call_tool("get_optimization_capabilities", {})
    assert result.is_error is False
    served = result.structured_content

    printed = runner.invoke(app, ["capabilities", "--json"])
    assert printed.exit_code == 0
    payload = json.loads(printed.stdout)

    assert set(payload) == set(served)
    assert payload == served
    assert payload["problem_json_schema"] is None
    # Batch 6 (J): the new per-backend flag is on both sides.
    for backend in payload["backends"]:
        assert isinstance(backend["supports_interrupt"], bool), backend["name"]

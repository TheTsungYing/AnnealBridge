"""A settings error while a tool call builds the process-wide state.

``get_state()`` builds the state lazily, inside the first tool call that needs
it, while the server is already serving: a ``SettingsError`` there cannot end
the process the way ``main()`` does. It must reach the client as a tool error
carrying the operator-formatted message — not the SDK's generic crash result —
without being logged as a crash, and the state must stay unbuilt.
"""

import logging

import pytest
from mcp import Client

from annealbridge.config import SettingsError
from annealbridge.interfaces.mcp import mcp, server

pytestmark = pytest.mark.anyio

# Shaped like the message load_settings() raises for an out-of-range value.
SETTINGS_ERROR = (
    "Invalid server settings (1 error(s)); values are not echoed:\n"
    "  ANNEALBRIDGE_MAX_CONCURRENT_SOLVES: Input should be greater than or equal to 1"
)

TOOL_CALLS = [
    ("get_optimization_capabilities", {}),
    ("validate_optimization_problem", {"problem": server.EXAMPLE_PROBLEM}),
    ("recommend_backend", {"problem": server.EXAMPLE_PROBLEM}),
    ("solve_optimization", {"problem": server.EXAMPLE_PROBLEM}),
]


@pytest.mark.parametrize(
    ("tool", "arguments"), TOOL_CALLS, ids=[tool for tool, _ in TOOL_CALLS]
)
async def test_settings_error_while_building_the_state_is_a_tool_error_naming_the_variable(
    tool, arguments, monkeypatch, caplog
):
    """With no state injected, the tool's ``get_state()`` call builds it and
    ``build_state()`` refuses the settings. The client gets an ``is_error``
    result whose text keeps the settings message after the SDK's
    ``Error executing tool <name>:`` prefix, nothing is logged at ERROR or
    with a traceback (the SDK's crash path), and ``_state`` stays ``None``."""

    def refuse_the_settings(settings=None):
        raise SettingsError(SETTINGS_ERROR)

    # monkeypatch, not reset_state(): the injected state comes back afterwards.
    monkeypatch.setattr(server, "_state", None)
    monkeypatch.setattr(server, "build_state", refuse_the_settings)
    caplog.set_level(logging.INFO)

    async with Client(mcp) as client:
        result = await client.call_tool(tool, arguments)

    assert result.is_error is True
    message = result.content[0].text
    assert f"Error executing tool {tool}: Invalid server settings" in message
    assert "ANNEALBRIDGE_MAX_CONCURRENT_SOLVES" in message
    assert server._state is None
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    assert "Traceback" not in caplog.text

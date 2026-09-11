"""What the server tells a host at initialize: instructions and version.

The per-tool descriptions say *when* to call each tool; the server-level
instructions carry what only the whole server can say — the call order, the
rules a first document most often breaks, and one complete example problem —
so an agent learns the document shape before its first call instead of from
its first error.
"""

from importlib.metadata import version

import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp, server
from annealbridge.models import OptimizationProblem

pytestmark = pytest.mark.anyio


async def test_instructions_reach_the_client():
    async with Client(mcp) as client:
        instructions = client.instructions

    assert instructions == server.SERVER_INSTRUCTIONS
    for tool in (
        "get_optimization_capabilities",
        "validate_optimization_problem",
        "recommend_backend",
        "solve_optimization",
    ):
        assert tool in instructions
    # The three rules that cost a round trip when learned from an error.
    assert '"version": "1.1"' in instructions
    assert "integer coefficients" in instructions
    assert "never ignored" in instructions


async def test_version_is_the_installed_distribution_version():
    async with Client(mcp) as client:
        info = client.server_info

    assert info is not None
    assert info.name == "AnnealBridge"
    assert info.version == version("annealbridge")


def test_example_problem_is_a_valid_problem_and_is_embedded_verbatim():
    problem = OptimizationProblem.model_validate(server.EXAMPLE_PROBLEM)

    assert problem.solver.backend == "exact"
    assert [variable.name for variable in problem.variables] == [
        "item_a",
        "item_b",
        "item_c",
    ]
    # Every field name of the example appears in the instructions text, so
    # the example the agent reads is the one the test validated.
    for key in ("linear_terms", "constraints", "operator", "rhs", "solver"):
        assert f'"{key}"' in server.SERVER_INSTRUCTIONS


def test_instructions_carry_no_configuration_values():
    # Same rule as recommended_action: categorical guidance only; limits
    # and settings come from get_optimization_capabilities.
    text = server.SERVER_INSTRUCTIONS.split("A minimal complete problem")[0]
    assert "ANNEALBRIDGE_" not in text
    assert "max_" not in text
    assert "limit of" not in text


def test_package_version_falls_back_outside_a_distribution(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    def missing(_name: str) -> str:
        raise PackageNotFoundError("annealbridge")

    from annealbridge import version as version_module

    monkeypatch.setattr(version_module, "version", missing)
    assert version_module.package_version() == "unknown"
    assert server._package_version() == "unknown"

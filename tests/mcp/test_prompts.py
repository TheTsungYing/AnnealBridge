"""The three MCP prompts, through a real client.

A host lists prompts in its input menu, so these are the one place a user
sees what the server can do without reading a tool description. What the
tests below hold is that each prompt is listed with the metadata a menu
needs, that it carries the user's own words through when they gave any, that
the guidance it returns names the same tools the server instructions do and
no configuration value, and that the example document it ends on is a valid,
feasible problem rather than prose that only looks like one.
"""

import json

import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp, prompts, server
from annealbridge.models import OptimizationProblem

pytestmark = pytest.mark.anyio

PROMPT_NAMES = ("pick_subset", "assign", "schedule_shifts")

# The marker the example document follows; the prompt text is split on it.
EXAMPLE_HEADING = "The smallest complete document of this shape:\n"

# Every tool the guidance points at. Each must also be in the server
# instructions, so a prompt never sends an agent at a tool the server does
# not describe.
NAMED_TOOLS = (
    "solve_optimization",
    "validate_optimization_problem",
    "recommend_backend",
    "get_optimization_capabilities",
)


async def _prompt_text(name: str, arguments: dict | None = None) -> str:
    """The single user message of ``name``, rendered with ``arguments``."""
    async with Client(mcp) as client:
        if arguments is None:
            result = await client.get_prompt(name)
        else:
            result = await client.get_prompt(name, arguments)

    assert len(result.messages) == 1
    message = result.messages[0]
    assert message.role == "user"
    return message.content.text


def _example_json(text: str) -> dict:
    """The document the prompt ends on, parsed."""
    return json.loads(text.split(EXAMPLE_HEADING, 1)[1])


async def test_the_three_prompts_are_listed_with_menu_metadata():
    async with Client(mcp) as client:
        listed = (await client.list_prompts()).prompts

    assert {prompt.name for prompt in listed} == set(PROMPT_NAMES)
    for prompt in listed:
        # A host shows the title in its menu and the description under it.
        assert prompt.title
        assert prompt.description
        # One optional argument, so the host can offer the prompt before the
        # user has typed anything.
        assert len(prompt.arguments) == 1
        argument = prompt.arguments[0]
        assert argument.name == "request"
        assert argument.required is False
        assert argument.description


@pytest.mark.parametrize("name", PROMPT_NAMES)
async def test_the_request_is_carried_through_verbatim(name):
    request = "Ann and Bo, two jobs, keep the total cost down"

    with_request = await _prompt_text(name, {"request": request})
    without_request = await _prompt_text(name)

    assert request in with_request
    assert "Translate it into a problem document" in with_request
    # Nothing is invented when the user gave no words of their own.
    assert "The user's request" not in without_request
    assert EXAMPLE_HEADING in with_request
    assert EXAMPLE_HEADING in without_request


@pytest.mark.parametrize("name", PROMPT_NAMES)
async def test_the_embedded_example_is_the_one_the_module_holds(name):
    text = await _prompt_text(name)

    assert _example_json(text) == prompts.EXAMPLES[name]


@pytest.mark.parametrize("name", PROMPT_NAMES)
async def test_the_embedded_example_solves_to_a_proven_optimum(name):
    # The example an agent copies must parse against the schema and actually
    # solve, not merely look like a document.
    text = await _prompt_text(name)
    problem = OptimizationProblem.model_validate(_example_json(text))

    assert problem.solver.backend == "exact"

    result = server.get_state().service.solve(problem)

    assert result.status == "success"
    assert result.optimality_proven is True
    assert result.solutions
    assert result.solutions[0].hard_constraints_satisfied is True


@pytest.mark.parametrize("name", PROMPT_NAMES)
async def test_the_guidance_names_the_tools_and_no_configuration_value(name):
    text = await _prompt_text(name)

    for tool in NAMED_TOOLS:
        assert tool in text
        # The server instructions describe every tool a prompt points at.
        assert tool in server.SERVER_INSTRUCTIONS
    assert "annealbridge://schema" in text
    # The one rule that costs a round trip when learned from an error.
    assert '"version": "1.1"' in text
    # Same rule as the server instructions and recommended_action: no
    # setting, no limit, nothing that goes stale when a value changes.
    assert "ANNEALBRIDGE_" not in text

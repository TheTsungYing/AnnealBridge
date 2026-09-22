"""The five MCP resources, through a real client.

An agent reads a resource instead of guessing what a complete document looks
like, so the four examples must be the repository's own ``examples/*.json``
— shipped inside the package because the repository directory never reaches
an installed wheel — and the schema must be the one the CLI's
``export-schema`` prints. The drift guard below holds the two copies of each
example byte-for-byte equal; without it the packaged copy could quietly fall
behind the file the README links to.
"""

import json
from pathlib import Path

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError
from typer.testing import CliRunner

from annealbridge.interfaces.cli.main import app
from annealbridge.interfaces.mcp import mcp, resources
from annealbridge.models import OptimizationProblem
from tests.conftest import EXAMPLES_DIR

pytestmark = pytest.mark.anyio

EXAMPLE_NAMES = ("knapsack", "integer_knapsack", "assignment", "tsp")
EXAMPLE_URIS = tuple(f"annealbridge://examples/{name}" for name in EXAMPLE_NAMES)
ALL_URIS = EXAMPLE_URIS + ("annealbridge://schema",)

# The directory the resources read from, inside the installed package.
PACKAGED_EXAMPLES = Path(resources.__file__).parent / "examples"

runner = CliRunner()


async def _read(uri: str):
    async with Client(mcp) as client:
        result = await client.read_resource(uri)

    assert len(result.contents) == 1
    return result.contents[0]


async def test_the_five_resources_are_listed_with_their_metadata():
    async with Client(mcp) as client:
        listed = (await client.list_resources()).resources

    assert {str(resource.uri) for resource in listed} == set(ALL_URIS)
    for resource in listed:
        assert resource.mime_type == "application/json"
        assert resource.name
        assert resource.title
        assert resource.description


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
async def test_an_example_resource_is_the_shipped_file_verbatim(name):
    content = await _read(f"annealbridge://examples/{name}")

    assert content.mime_type == "application/json"
    assert content.text == (EXAMPLES_DIR / f"{name}.json").read_text(encoding="utf-8")

    payload = json.loads(content.text)
    problem = OptimizationProblem.model_validate(payload)
    assert problem.name == payload["name"]


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_the_packaged_example_has_not_drifted_from_the_repository_one(name):
    assert (PACKAGED_EXAMPLES / f"{name}.json").read_bytes() == (
        EXAMPLES_DIR / f"{name}.json"
    ).read_bytes()


def test_the_package_ships_exactly_the_four_examples():
    assert sorted(path.name for path in PACKAGED_EXAMPLES.glob("*.json")) == sorted(
        f"{name}.json" for name in EXAMPLE_NAMES
    )


async def test_the_schema_resource_is_what_export_schema_prints():
    content = await _read("annealbridge://schema")
    schema = OptimizationProblem.model_json_schema()

    assert content.mime_type == "application/json"
    assert json.loads(content.text) == schema
    # Same text, not merely the same object: the CLI's indentation too.
    assert content.text == json.dumps(schema, indent=2)

    exported = runner.invoke(app, ["export-schema"])
    assert exported.exit_code == 0
    assert json.loads(exported.output) == schema


async def test_an_unknown_resource_uri_is_refused():
    async with Client(mcp) as client:
        with pytest.raises(MCPError):
            await client.read_resource("annealbridge://examples/nope")

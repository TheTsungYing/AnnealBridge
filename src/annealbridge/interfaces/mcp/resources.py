"""The MCP resources: the shipped example problems and the problem JSON schema.

A host lists these under ``resources/list`` and an agent reads one instead of
guessing what a complete document looks like. The examples, and each one's
name, title and description, come from
:mod:`annealbridge.interfaces.bundled_examples` — the same registry the CLI's
``example`` command prints — so both interfaces serve the same files.
The schema is the same ``OptimizationProblem.model_json_schema()`` the CLI's
``export-schema`` prints, with the same indentation.
"""

import json
from collections.abc import Callable

from annealbridge.interfaces.bundled_examples import EXAMPLES, example_text
from annealbridge.interfaces.mcp.server import mcp
from annealbridge.models import OptimizationProblem

JSON = "application/json"


def schema_text() -> str:
    """The problem JSON schema, formatted as ``annealbridge export-schema``
    prints it."""
    return json.dumps(OptimizationProblem.model_json_schema(), indent=2)


def _example_reader(name: str) -> Callable[[], str]:
    """A resource function returning the shipped ``<name>.json`` verbatim."""

    def read() -> str:
        return example_text(name)

    read.__name__ = f"{name}_example"
    return read


for _example in EXAMPLES:
    mcp.resource(
        f"annealbridge://examples/{_example.name}",
        name=_example.name,
        title=_example.title,
        description=_example.description,
        mime_type=JSON,
    )(_example_reader(_example.name))


@mcp.resource(
    "annealbridge://schema",
    name="problem_json_schema",
    title="OptimizationProblem JSON schema",
    description=(
        "The full JSON schema of the problem document this server accepts, "
        "with a description on every field. Identical to what "
        "get_optimization_capabilities returns as problem_json_schema when "
        "called with include_schema: true, and to what the CLI's "
        "export-schema prints."
    ),
    mime_type=JSON,
)
def problem_schema() -> str:
    return schema_text()

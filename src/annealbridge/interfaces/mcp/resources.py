"""The MCP resources: the four example problems and the problem JSON schema.

A host lists these under ``resources/list`` and an agent reads one instead of
guessing what a complete document looks like. The four examples are the
repository's ``examples/*.json``, shipped inside this package as data files
(``pyproject.toml`` ``package-data``) because the repository directory never
reaches an installed wheel; a test holds the two copies byte-for-byte equal.
The schema is the same ``OptimizationProblem.model_json_schema()`` the CLI's
``export-schema`` prints, with the same indentation.
"""

import json
from importlib.resources import files

from annealbridge.interfaces.mcp.server import mcp
from annealbridge.models import OptimizationProblem

# Package-relative directory the example files live in.
_EXAMPLES = files("annealbridge.interfaces.mcp") / "examples"

JSON = "application/json"


def example_text(name: str) -> str:
    """The shipped ``examples/<name>.json`` exactly as the file holds it."""
    return (_EXAMPLES / f"{name}.json").read_text(encoding="utf-8")


def schema_text() -> str:
    """The problem JSON schema, formatted as ``annealbridge export-schema``
    prints it."""
    return json.dumps(OptimizationProblem.model_json_schema(), indent=2)


@mcp.resource(
    "annealbridge://examples/knapsack",
    name="knapsack",
    title="Example: 0/1 knapsack",
    description=(
        "A complete problem document: four binary variables, a maximized "
        "linear objective and one hard <= constraint (capacity 10). Schema "
        "version 1.0. The optimum is items A and C with value 17."
    ),
    mime_type=JSON,
)
def knapsack_example() -> str:
    return example_text("knapsack")


@mcp.resource(
    "annealbridge://examples/integer_knapsack",
    name="integer_knapsack",
    title="Example: bounded integer knapsack with a soft constraint",
    description=(
        "A complete problem document with four bounded integer variables "
        "(0..3 copies each), a hard capacity constraint and one soft "
        "constraint with a weight. Schema version 1.1, which integer "
        "variables require. The optimum has value 34."
    ),
    mime_type=JSON,
)
def integer_knapsack_example() -> str:
    return example_text("integer_knapsack")


@mcp.resource(
    "annealbridge://examples/assignment",
    name="assignment",
    title="Example: assignment (workers to tasks)",
    description=(
        "A complete problem document assigning three workers to three tasks: "
        "one binary variable per worker/task pair, a minimized cost objective "
        "and six hard == 1 constraints (each worker one task, each task one "
        "worker). The optimum has total cost 8."
    ),
    mime_type=JSON,
)
def assignment_example() -> str:
    return example_text("assignment")


@mcp.resource(
    "annealbridge://examples/tsp",
    name="tsp",
    title="Example: travelling salesman over four cities",
    description=(
        "A complete problem document with a quadratic objective: one binary "
        "variable per city/position pair, quadratic distance terms between "
        "consecutive positions, and hard == 1 constraints per city and per "
        "position. The optimal tour has length 8."
    ),
    mime_type=JSON,
)
def tsp_example() -> str:
    return example_text("tsp")


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

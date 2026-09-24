"""The example problems shipped inside the package, shared by CLI and MCP.

The repository's ``examples/*.json`` never reach an installed wheel, so the
same files are shipped as package data (``pyproject.toml`` ``package-data``)
in the ``examples`` directory next to this module; a test holds the two
copies byte-for-byte equal. This registry is the one source of each
example's name, title and description: the MCP server lists them as
resources and the CLI ``example`` command prints them.
Lives outside the ``mcp`` subpackage so the CLI can use it without the
``[mcp]`` extra installed. Pure data access; no optimization logic lives here.
"""

from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True)
class BundledExample:
    """One shipped example problem: its file stem and how it is described."""

    name: str
    title: str
    description: str


class UnknownExampleError(KeyError):
    """Raised for a name that is not one of the shipped examples."""


# Package-relative directory the example files live in. ``annealbridge.
# interfaces`` imports nothing optional, so reading from it is safe on a
# core-only install.
_EXAMPLES = files("annealbridge.interfaces") / "examples"

EXAMPLES: tuple[BundledExample, ...] = (
    BundledExample(
        name="knapsack",
        title="Example: 0/1 knapsack",
        description=(
            "A complete problem document: four binary variables, a maximized "
            "linear objective and one hard <= constraint (capacity 10). Schema "
            "version 1.0. The optimum is items A and C with value 17."
        ),
    ),
    BundledExample(
        name="integer_knapsack",
        title="Example: bounded integer knapsack with a soft constraint",
        description=(
            "A complete problem document with four bounded integer variables "
            "(0..3 copies each), a hard capacity constraint and one soft "
            "constraint with a weight. Schema version 1.1, which integer "
            "variables require. The optimum has value 34."
        ),
    ),
    BundledExample(
        name="assignment",
        title="Example: assignment (workers to tasks)",
        description=(
            "A complete problem document assigning three workers to three tasks: "
            "one binary variable per worker/task pair, a minimized cost objective "
            "and six hard == 1 constraints (each worker one task, each task one "
            "worker). The optimum has total cost 8."
        ),
    ),
    BundledExample(
        name="tsp",
        title="Example: travelling salesman over four cities",
        description=(
            "A complete problem document with a quadratic objective: one binary "
            "variable per city/position pair, quadratic distance terms between "
            "consecutive positions, and hard == 1 constraints per city and per "
            "position. The optimal tour has length 8."
        ),
    ),
    BundledExample(
        name="shift_scheduling",
        title="Example: shift scheduling (people to shifts)",
        description=(
            "A complete problem document rostering three people onto four "
            "shifts: one binary variable per person/shift pair, a minimized "
            "dislike objective, hard == 1 coverage per shift, hard <= 2 shifts "
            "per person, a hard <= 1 rest rule (no Monday night followed by "
            "Tuesday day) and one soft preference with a weight. Schema "
            "version 1.0. The optimum has total dislike 7 with the preference "
            "honoured."
        ),
    ),
)

_BY_NAME = {example.name: example for example in EXAMPLES}


def example_names() -> tuple[str, ...]:
    """The shipped example names, in registry order."""
    return tuple(_BY_NAME)


def get_example(name: str) -> BundledExample:
    """The registry entry for ``name``; :class:`UnknownExampleError` if none."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise UnknownExampleError(name) from None


def example_text(name: str) -> str:
    """The shipped ``examples/<name>.json`` exactly as the file holds it.

    Decoded from the file's bytes rather than read in text mode, so no
    newline translation can make the text differ from the file.
    """
    get_example(name)
    return (_EXAMPLES / f"{name}.json").read_bytes().decode("utf-8")

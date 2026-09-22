"""The MCP prompts: one per everyday request shape the server is for.

A host (Claude Desktop, for one) lists prompts in its input menu, so these are
the one place a user sees what the server can do without reading a tool
description. Each prompt takes the user's request in their own words, if
they gave one, and returns a guide for turning that shape of request into a
problem document — which decisions become variables, what the objective and
the constraints are, which schema version to declare and which tools to call
— followed by the smallest complete document of that shape.

The guidance repeats what the server instructions already say and adds no
rule of its own; like them it names no configuration value and no limit. The
example documents are kept as data so a test can prove each one is a valid,
feasible problem.
"""

import json
from typing import Annotated

from pydantic import Field

from annealbridge.interfaces.mcp.server import mcp

# Request shape -> the smallest complete document of that shape. Every value
# is a plain number so the story is readable next to the JSON.
EXAMPLES: dict[str, dict] = {
    "pick_subset": {
        "version": "1.0",
        "name": "pick_items",
        "description": (
            "Pick items worth the most within a weight limit of 10 "
            "(A: value 10, weight 6; B: value 8, weight 5; C: value 7, "
            "weight 4)."
        ),
        "variables": [
            {"name": "take_a", "type": "binary"},
            {"name": "take_b", "type": "binary"},
            {"name": "take_c", "type": "binary"},
        ],
        "objective": {
            "direction": "maximize",
            "linear_terms": [
                {"variable": "take_a", "coefficient": 10},
                {"variable": "take_b", "coefficient": 8},
                {"variable": "take_c", "coefficient": 7},
            ],
        },
        "constraints": [
            {
                "id": "weight_limit",
                "type": "hard",
                "terms": [
                    {"variable": "take_a", "coefficient": 6},
                    {"variable": "take_b", "coefficient": 5},
                    {"variable": "take_c", "coefficient": 4},
                ],
                "operator": "<=",
                "rhs": 10,
            }
        ],
        "solver": {"backend": "exact"},
    },
    "assign": {
        "version": "1.0",
        "name": "assign_workers",
        "description": (
            "Assign two workers to two tasks, one task each and one worker "
            "per task, at the lowest total cost (ann: cook 2, drive 8; "
            "bo: cook 7, drive 5)."
        ),
        "variables": [
            {"name": "ann_cook", "type": "binary"},
            {"name": "ann_drive", "type": "binary"},
            {"name": "bo_cook", "type": "binary"},
            {"name": "bo_drive", "type": "binary"},
        ],
        "objective": {
            "direction": "minimize",
            "linear_terms": [
                {"variable": "ann_cook", "coefficient": 2},
                {"variable": "ann_drive", "coefficient": 8},
                {"variable": "bo_cook", "coefficient": 7},
                {"variable": "bo_drive", "coefficient": 5},
            ],
        },
        "constraints": [
            {
                "id": "ann_one_task",
                "type": "hard",
                "terms": [
                    {"variable": "ann_cook", "coefficient": 1},
                    {"variable": "ann_drive", "coefficient": 1},
                ],
                "operator": "==",
                "rhs": 1,
            },
            {
                "id": "bo_one_task",
                "type": "hard",
                "terms": [
                    {"variable": "bo_cook", "coefficient": 1},
                    {"variable": "bo_drive", "coefficient": 1},
                ],
                "operator": "==",
                "rhs": 1,
            },
            {
                "id": "cook_one_worker",
                "type": "hard",
                "terms": [
                    {"variable": "ann_cook", "coefficient": 1},
                    {"variable": "bo_cook", "coefficient": 1},
                ],
                "operator": "==",
                "rhs": 1,
            },
            {
                "id": "drive_one_worker",
                "type": "hard",
                "terms": [
                    {"variable": "ann_drive", "coefficient": 1},
                    {"variable": "bo_drive", "coefficient": 1},
                ],
                "operator": "==",
                "rhs": 1,
            },
        ],
        "solver": {"backend": "exact"},
    },
    "schedule_shifts": {
        "version": "1.0",
        "name": "schedule_shifts",
        "description": (
            "Cover a morning and an evening shift with exactly one person "
            "each, nobody working more than one shift; ann prefers mornings "
            "(cost 1 morning, 3 evening), bo prefers evenings (3 morning, "
            "1 evening), and bo would rather not work mornings at all."
        ),
        "variables": [
            {"name": "ann_morning", "type": "binary"},
            {"name": "ann_evening", "type": "binary"},
            {"name": "bo_morning", "type": "binary"},
            {"name": "bo_evening", "type": "binary"},
        ],
        "objective": {
            "direction": "minimize",
            "linear_terms": [
                {"variable": "ann_morning", "coefficient": 1},
                {"variable": "ann_evening", "coefficient": 3},
                {"variable": "bo_morning", "coefficient": 3},
                {"variable": "bo_evening", "coefficient": 1},
            ],
        },
        "constraints": [
            {
                "id": "morning_covered",
                "type": "hard",
                "terms": [
                    {"variable": "ann_morning", "coefficient": 1},
                    {"variable": "bo_morning", "coefficient": 1},
                ],
                "operator": "==",
                "rhs": 1,
            },
            {
                "id": "evening_covered",
                "type": "hard",
                "terms": [
                    {"variable": "ann_evening", "coefficient": 1},
                    {"variable": "bo_evening", "coefficient": 1},
                ],
                "operator": "==",
                "rhs": 1,
            },
            {
                "id": "ann_at_most_one_shift",
                "type": "hard",
                "terms": [
                    {"variable": "ann_morning", "coefficient": 1},
                    {"variable": "ann_evening", "coefficient": 1},
                ],
                "operator": "<=",
                "rhs": 1,
            },
            {
                "id": "bo_at_most_one_shift",
                "type": "hard",
                "terms": [
                    {"variable": "bo_morning", "coefficient": 1},
                    {"variable": "bo_evening", "coefficient": 1},
                ],
                "operator": "<=",
                "rhs": 1,
            },
            {
                "id": "bo_prefers_not_morning",
                "type": "soft",
                "terms": [{"variable": "bo_morning", "coefficient": 1}],
                "operator": "<=",
                "rhs": 0,
                "weight": 2,
            },
        ],
        "solver": {"backend": "exact"},
    },
}

# The one argument every prompt takes; optional, so a host can offer the
# prompt before the user has typed anything.
Request = Annotated[
    str | None,
    Field(description="The user's request in their own words (optional)."),
]

# Shared by the three prompts: how a document is finished and which tool
# runs it. Mirrors the server instructions; it adds no rule of its own.
_COMMON_TAIL = """\
Schema version: declare "version": "1.0" when every variable is binary. If any
decision is a bounded count rather than yes/no, declare it with "type":
"integer" plus integer lower_bound and upper_bound, and then the document must
carry "version": "1.1" at its top level.

Rules: a hard constraint must hold; a soft constraint is only penalized and
needs a positive weight in objective-value units (a hard one must not carry a
weight). Inequality constraints (<=, >=) need integer coefficients and an
integer rhs. Leave solver.penalty_multiplier at its default. Use only the
fields the problem JSON schema declares — an unknown field is rejected, never
ignored.

Tools: if the user named a backend, write it into solver.backend; otherwise
call recommend_backend and follow the server instructions on when to present
the top entries and ask which to run. On a local backend you may call
solve_optimization directly — an invalid document comes back as
invalid_problem with every error and a recommended_action, so fix it and solve
again. Before a remote backend, or when the problem is large, call
validate_optimization_problem first. When the document needs more than the
example below shows, read the resource annealbridge://schema or call
get_optimization_capabilities with include_schema: true; complete example
documents are under annealbridge://examples/. In the answer, report the
solution in the user's terms, say which backend ran and, when the result says
so, that optimality was proven.
"""


def _render(intro: str, body: str, example_key: str, request: str | None) -> str:
    """Assemble one prompt: the request (if any), the guide, and the example."""
    parts: list[str] = []
    if request:
        parts.append(
            "The user's request, in their own words:\n\n"
            + request.strip()
            + "\n\nTranslate it into a problem document as follows.\n"
        )
    parts.append(intro + "\n")
    parts.append(body + "\n")
    parts.append(_COMMON_TAIL)
    parts.append(
        "\nThe smallest complete document of this shape:\n"
        + json.dumps(EXAMPLES[example_key], indent=2)
        + "\n"
    )
    return "\n".join(parts)


@mcp.prompt(
    name="pick_subset",
    title="Pick a subset within a budget or capacity",
    description=(
        "Guide for a request of the form 'which items should I take within a "
        "budget, weight or capacity' (a knapsack): how the decisions, the "
        "score and the limits become a problem document, and which tools to "
        "call. Optionally takes the user's request in their own words."
    ),
)
def pick_subset(request: Request = None) -> str:
    intro = (
        "Shape: choose a subset of candidate items so that a total score is as "
        "high (or as low) as possible while one or more totals stay within a "
        "limit."
    )
    body = """\
Variables: one binary variable per candidate item, 1 when it is taken. If an
item can be taken several times up to a count, use one bounded integer
variable per item instead.

Objective: linear_terms with one term per item whose coefficient is the item's
value (or cost); direction "maximize" for value, "minimize" for cost.

Constraints: one hard constraint per limit (budget, weight, volume, slots)
with one term per item whose coefficient is that item's contribution, operator
"<=" and the limit as rhs. A requirement such as "at least two of these" is a
hard constraint with operator ">=". A preference ("ideally not both A and B")
is a soft constraint with a weight."""
    return _render(intro, body, "pick_subset", request)


@mcp.prompt(
    name="assign",
    title="Assign people or jobs to seats, shifts or machines",
    description=(
        "Guide for a request of the form 'who does what' or 'what goes where' "
        "(an assignment): one binary per pairing, exactly-one constraints per "
        "side, a cost or preference objective, and which tools to call. "
        "Optionally takes the user's request in their own words."
    ),
)
def assign(request: Request = None) -> str:
    intro = (
        "Shape: match each member of one group (people, jobs) to a member of "
        "another (tasks, seats, machines) so that the total cost, time or "
        "dissatisfaction is as small as possible."
    )
    body = """\
Variables: one binary variable per allowed pairing, named after both sides
(for example ann_cook), 1 when that pairing is chosen. Leave out pairings that
are impossible instead of forbidding them with a constraint.

Objective: linear_terms with one term per pairing whose coefficient is that
pairing's cost (direction "minimize") or its benefit (direction "maximize").
Where the cost of a pairing depends on another pairing being chosen too, add
an entry to the objective's quadratic_terms (variable1, variable2,
coefficient) over the two variables.

Constraints: for each member of the first group, one hard constraint summing
its pairings with operator "==" and rhs 1 (everyone gets exactly one task);
for each member of the second group, one hard constraint summing its pairings
with "==" 1 (every task gets exactly one person), or "<=" 1 when a task may
stay unfilled, or "<=" with a larger rhs when it takes several people.
Preferences that may be broken are soft constraints with a weight."""
    return _render(intro, body, "assign", request)


@mcp.prompt(
    name="schedule_shifts",
    title="Schedule people to shifts",
    description=(
        "Guide for a request of the form 'who works which shift' (a roster): "
        "one binary per person/shift, coverage and workload limits as hard "
        "constraints, preferences as soft ones, and which tools to call. "
        "Optionally takes the user's request in their own words."
    ),
)
def schedule_shifts(request: Request = None) -> str:
    intro = (
        "Shape: decide which people work which shifts so that every shift is "
        "covered, nobody works more than allowed, and preferences are honoured "
        "as far as possible."
    )
    body = """\
Variables: one binary variable per person/shift pair (for example
ann_morning), 1 when that person works that shift. Only create pairs the
person could actually work.

Objective: direction "minimize" with linear_terms whose coefficients express
how unwelcome each pair is (0 or a small number for a preferred shift, larger
for a disliked one), or "maximize" with a preference score; if every
assignment is equally fine, keep every coefficient 0 and let the constraints
decide.

Constraints, all hard unless the user says a rule may bend: for each shift,
one constraint summing that shift's pairs with operator "==" and rhs equal to
the number of people it needs (or "<=" the number it can hold, or ">=" the
minimum it must have); for each person, one constraint summing their pairs
with "<=" and rhs equal to the most shifts they may work (and ">=" for a
minimum). Two shifts the same person must not work back to back are one
constraint over those two pairs with "<=" 1. A preference (someone would
rather not work evenings) is a soft constraint over those pairs with operator
"<=", rhs 0 and a weight that says how much it matters relative to the
objective."""
    return _render(intro, body, "schedule_shifts", request)

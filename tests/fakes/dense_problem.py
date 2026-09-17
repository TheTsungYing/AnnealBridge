"""Large fully dense binary problems for the structure-fit tests (2026-09-17).

A fully dense problem on ``n`` variables carries ``n * (n - 1) / 2`` quadratic
terms, so building one is the expensive part of a routing test (about 0.2 s
at 500 variables) and every ``recommend`` over it validates that term list
once per backend. Each shape is therefore built once per test session and
shared; the models are never mutated by the code under test.
"""

from functools import lru_cache

from annealbridge.models import OptimizationProblem


@lru_cache(maxsize=None)
def dense_problem(num_variables: int, *, hard_constraint: bool = False) -> OptimizationProblem:
    """Every pairwise product with coefficient 1 (a fully dense QUBO).

    With ``hard_constraint`` one hard ``==`` constraint over the first three
    variables is added: effective on the bqm path (a non-redundant equality),
    so it turns the large dense shape into the penalty-dominated one.
    """
    names = [f"d{index}" for index in range(num_variables)]
    constraints = []
    if hard_constraint:
        constraints.append(
            {
                "id": "pick_one",
                "type": "hard",
                "terms": [{"variable": name, "coefficient": 1} for name in names[:3]],
                "operator": "==",
                "rhs": 1,
            }
        )
    return OptimizationProblem.model_validate(
        {
            "name": "dense",
            "variables": [{"name": name} for name in names],
            "objective": {
                "direction": "minimize",
                "linear_terms": [{"variable": name, "coefficient": -1} for name in names],
                "quadratic_terms": [
                    {"variable1": first, "variable2": second, "coefficient": 1}
                    for index, first in enumerate(names)
                    for second in names[index + 1 :]
                ],
            },
            "constraints": constraints,
        }
    )

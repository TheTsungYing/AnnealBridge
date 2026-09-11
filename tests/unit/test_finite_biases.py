"""``has_finite_biases`` and ``bqm_has_finite_biases`` are one predicate, twice.

The CQM path runs the generic ``has_finite_biases`` (in dimod 0.12 neither
the objective nor a constraint's lhs view has ``to_numpy_vectors``); the
BQM path runs the vectorised ``bqm_has_finite_biases``. This module pins
the two to the same answer for the same BQM: if either one stops seeing a
non-finite offset, linear or quadratic bias, it fails here.

The BQM compiler's ``_check_finite`` is tested directly as well, so the
switch to the vectorised form still raises a ``NonFiniteModelError`` that
names the ``hard_penalty``.
"""

import math

import dimod
import pytest

from annealbridge.compiler.bqm import _check_finite
from annealbridge.compiler.objective import bqm_has_finite_biases, has_finite_biases
from annealbridge.exceptions import NonFiniteModelError
from annealbridge.models import OptimizationProblem


def make_problem() -> OptimizationProblem:
    """The smallest valid problem; only its name reaches the error message."""
    return OptimizationProblem.model_validate(
        {
            "version": "1.0",
            "name": "finite biases test",
            "variables": [{"name": "x", "type": "binary"}],
            "objective": {
                "direction": "minimize",
                "linear_terms": [{"variable": "x", "coefficient": 1.0}],
            },
            "constraints": [],
        }
    )


def make_bqm(
    linear: dict[str, float], quadratic: dict[tuple[str, str], float], offset: float
) -> dimod.BinaryQuadraticModel:
    return dimod.BinaryQuadraticModel(linear, quadratic, offset, "BINARY")


FINITE_CASES = [
    pytest.param({"a": 1.5, "b": -2.0}, {("a", "b"): 0.25}, 3.0, True, id="all-finite"),
    pytest.param({"a": 1.5, "b": -2.0}, {("a", "b"): 0.25}, math.inf, False, id="offset-inf"),
    pytest.param({"a": 1.5, "b": math.nan}, {("a", "b"): 0.25}, 3.0, False, id="linear-nan"),
    pytest.param({"a": 1.5, "b": -2.0}, {("a", "b"): -math.inf}, 3.0, False, id="quadratic-neg-inf"),
]


@pytest.mark.parametrize("linear, quadratic, offset, expected", FINITE_CASES)
def test_both_predicates_agree(
    linear: dict[str, float],
    quadratic: dict[tuple[str, str], float],
    offset: float,
    expected: bool,
) -> None:
    bqm = make_bqm(linear, quadratic, offset)
    assert bqm_has_finite_biases(bqm) == has_finite_biases(bqm)
    assert bqm_has_finite_biases(bqm) is expected


def test_both_predicates_agree_on_the_empty_bqm() -> None:
    """No biases at all: both must answer True (empty arrays must not trip the vectorised form)."""
    bqm = dimod.BinaryQuadraticModel("BINARY")
    assert bqm_has_finite_biases(bqm) == has_finite_biases(bqm)
    assert bqm_has_finite_biases(bqm) is True


def test_check_finite_raises_on_a_non_finite_bqm() -> None:
    problem = make_problem()
    bqm = make_bqm({"a": math.inf}, {}, 0.0)
    with pytest.raises(NonFiniteModelError) as excinfo:
        _check_finite(bqm, problem, 100.0)
    assert "hard_penalty=100.0" in str(excinfo.value)


def test_check_finite_accepts_a_finite_bqm() -> None:
    problem = make_problem()
    bqm = make_bqm({"a": 1.0, "b": -1.0}, {("a", "b"): 2.0}, 0.5)
    _check_finite(bqm, problem, 100.0)

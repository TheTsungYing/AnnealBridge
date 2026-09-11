"""The one floating-point tolerance every constraint comparison uses (spec §23.1).

Constraint satisfaction is decided against the *original* problem with
float arithmetic, so a small tolerance is unavoidable. Since review F-05
(2026-09-09) it is a *hybrid* of an absolute floor and a relative part::

    tol(actual, rhs) = max(ABSOLUTE_TOLERANCE,
                           RELATIVE_TOLERANCE * max(|actual|, |rhs|))

    ==  : abs(actual - rhs) <= tol
    <=  : actual <= rhs + tol
    >=  : actual >= rhs - tol

Up to magnitude ``1e4`` the relative part is below the floor and the rule is
exactly the old absolute ``EPSILON = 1e-8``; at magnitude ``1e9`` the
tolerance is ``1e-3``, which absorbs the accumulation error of summing
coefficients of that size (``1e9 + 0.1 + 0.2`` lands 1.19e-7 away from
``1e9 + 0.3``) without accepting a real violation of one unit.

Four callers share this rule and must never drift apart:

- :mod:`annealbridge.validation.solution_validator` judges every candidate's
  ``satisfied`` flag with it, in both its scalar and its numpy kernel. A
  *soft* constraint's score deliberately does **not** use the tolerance: see
  that module's docstring (review F-05, 2026-09-11).
- :mod:`annealbridge.validation.problem_validator` uses the same rule to
  decide ``TRIVIALLY_INFEASIBLE`` (review F-25) and, for soft constraints,
  ``SOFT_ALWAYS_VIOLATED`` (review F-04 / F-12), so a constraint that is
  rejected or flagged before compilation is exactly one that no assignment
  could have passed afterwards.
- :mod:`annealbridge.compiler.cqm` decides its constant-constraint branch
  (``0 <op> rhs``) with :func:`satisfies`, so the compiler never refuses a
  constraint the validator accepted (review F-04).
- :func:`annealbridge.validation.estimates.analyze_inequality` reports a
  slack range of ``0`` instead of a negative one when :func:`satisfies`
  accepts ``lhs_min`` (review F-04, 2026-09-11), so a hard inequality that
  is only satisfiable *within the tolerance* is estimated and encoded rather
  than raising past the validator.

:func:`tolerance` is pure Python and :func:`tolerance_array` is the numpy
version; they perform the same IEEE operations in the same order, so their
results are bit-identical (``tests/unit/test_tolerance.py`` asserts it).
This module depends on nothing but numpy so that any layer — including the
compilers — can import it.
"""

import numpy as np

ABSOLUTE_TOLERANCE = 1e-8
"""Floor of the tolerance; the whole tolerance at small magnitudes."""

RELATIVE_TOLERANCE = 1e-12
"""Fraction of ``max(|actual|, |rhs|)`` added at large magnitudes."""

EPSILON = ABSOLUTE_TOLERANCE
"""Spec §23.1's original name for the absolute floor, kept for callers."""


def tolerance(actual: float, rhs: float) -> float:
    """Return the tolerance for comparing ``actual`` with ``rhs`` (scalar)."""
    return max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * max(abs(actual), abs(rhs)))


def satisfies(operator: str, actual: float, rhs: float) -> bool:
    """Whether ``actual <operator> rhs`` holds under the §23.1 tolerance.

    The three expressions are the ones ``solution_validator._evaluate``
    uses, kept literally identical (``tests/unit/test_tolerance.py`` checks
    them against ``validate_solution`` verdict by verdict): ``==`` is
    ``abs(actual - rhs) <= tol``, ``<=`` is ``actual <= rhs + tol`` and
    ``>=`` is ``actual >= rhs - tol``.
    """
    tol = tolerance(actual, rhs)
    if operator == "==":
        return abs(actual - rhs) <= tol
    if operator == "<=":
        return actual <= rhs + tol
    if operator == ">=":
        return actual >= rhs - tol
    raise ValueError(f"unknown constraint operator {operator!r}")


def tolerance_array(actual: np.ndarray, rhs: float) -> np.ndarray:
    """Vectorised :func:`tolerance`: one value per element of ``actual``.

    Same operations in the same order as the scalar version, so each element
    equals ``tolerance(actual[i], rhs)`` bit for bit.
    """
    return np.maximum(
        ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * np.maximum(np.abs(actual), abs(rhs))
    )

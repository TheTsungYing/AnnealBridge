"""Shared pure arithmetic for validation, penalty and compilation.

Single source of truth (Phase 2 spec §20) for the formulas that both the
problem validator (estimates, warnings) and the compiler / penalty strategy
rely on. Everything here is deterministic arithmetic over the raw problem
models: no BQM is ever built, nothing is logged, and no solver is known.

``compiler/slack.py`` and ``penalty/strategy.py`` import these functions so
the validator's estimates can never drift from what compilation actually
produces.

Bounds (3b spec §8)
-------------------
Every formula takes an optional ``bounds`` mapping ``{name: (lower, upper)}``
from :func:`variable_bounds`. ``None`` (the default) means "every variable is
binary", which is exactly the Phase 1/2/3a behaviour, so every existing
single-argument caller is unchanged — and for an all-binary problem each
function returns the *same float, bit for bit* whether ``bounds`` is passed
or not (``tests/unit/test_golden_phase3a.py`` and
``test_estimates_bounds.py`` prove it). Names absent from a given mapping are
binary too: that covers the slack bits the compiler generates and the
internal ``__slack_total`` pseudo-variable below.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from annealbridge.models import (
    Constraint,
    LinearTerm,
    Objective,
    OptimizationProblem,
)

Bounds = Mapping[str, tuple[int, int]]

_BINARY_BOUNDS: tuple[int, int] = (0, 1)


def variable_bounds(problem: OptimizationProblem) -> dict[str, tuple[int, int]]:
    """Return ``{name: (lower, upper)}`` for every declared variable.

    Binary variables map to ``(0, 1)``. Only meaningful after the validator's
    error pass (``Variable.bounds()`` raises for incomplete integer bounds).
    """
    return {variable.name: variable.bounds() for variable in problem.variables}


def integer_encoding_bits(lower: int, upper: int) -> int:
    """Number of bits the BQM path needs for an integer in ``lower..upper``.

    ``(upper - lower).bit_length()``: the same count
    :func:`compute_slack_coefficients` uses for a slack in ``0..S`` — integer
    variables and slacks share one binary expansion (3b §14). A binary
    variable (``0..1``) is one bit, so this also counts business binaries.
    """
    if upper < lower:
        raise ValueError(f"upper bound {upper} is below lower bound {lower}")
    return (upper - lower).bit_length()


def _bounds_of(bounds: Bounds | None, name: str) -> tuple[int, int]:
    if bounds is None:
        return _BINARY_BOUNDS
    return bounds.get(name, _BINARY_BOUNDS)


def _magnitude(bounds: Bounds | None, name: str) -> int:
    """``M = max(|lower|, |upper|)``: the largest absolute value a variable takes."""
    lower, upper = _bounds_of(bounds, name)
    return max(abs(lower), abs(upper))


def _term_range(value: float, lower: int, upper: int) -> tuple[float, float]:
    """``(min, max)`` of ``value * x`` over ``x in lower..upper``."""
    if lower == 0 and upper == 1:
        # The literal Phase 1 expression, kept so all-binary results stay
        # bit-identical (including the sign of a zero coefficient).
        return min(value, 0.0), max(value, 0.0)
    at_lower, at_upper = value * lower, value * upper
    if at_lower <= at_upper:
        return at_lower, at_upper
    return at_upper, at_lower


def accumulate_terms(terms: Iterable[LinearTerm]) -> dict[str, float]:
    """Sum coefficients per variable; duplicate variables accumulate (spec §15)."""
    coefficients: dict[str, float] = {}
    for term in terms:
        coefficients[term.variable] = coefficients.get(term.variable, 0.0) + term.coefficient
    return coefficients


def lhs_bounds(
    coefficients: dict[str, float], bounds: Bounds | None = None
) -> tuple[float, float]:
    """Return ``(lhs_min, lhs_max)`` of ``sum(a_i * x_i)`` over the variables' ranges.

    ``lhs_min = sum(min(a_i * lo_i, a_i * hi_i))`` and symmetrically for the
    maximum; with binary bounds (the default) this is ``sum(min(a_i, 0))`` /
    ``sum(max(a_i, 0))`` exactly as before.

    Must be called on *accumulated* coefficients (one entry per variable):
    per-term bounds over-estimate the range when a variable repeats with
    mixed signs, which is exactly how validator and compiler used to
    disagree on trivial infeasibility.
    """
    ranges = {
        variable: _term_range(value, *_bounds_of(bounds, variable))
        for variable, value in coefficients.items()
    }
    lhs_min = sum(low for low, _ in ranges.values())
    lhs_max = sum(high for _, high in ranges.values())
    return lhs_min, lhs_max


def compute_slack_coefficients(slack_range: int) -> list[int]:
    """Return binary-expansion coefficients for a slack in ``0..slack_range``.

    Uses ``m = slack_range.bit_length()`` bits (== ceil(log2(S+1)) without
    floating point): the first ``m - 1`` coefficients are ``1, 2, ..., 2^(m-2)``
    and the last is the remainder ``S - (2^(m-1) - 1)``, so every integer in
    ``0..S`` is representable and no bit combination exceeds ``S``.
    """
    if slack_range < 0:
        raise ValueError(f"slack range must be >= 0, got {slack_range}")
    if slack_range == 0:
        return []
    num_bits = slack_range.bit_length()
    coefficients = [1 << k for k in range(num_bits - 1)]
    coefficients.append(slack_range - ((1 << (num_bits - 1)) - 1))
    return coefficients


def compute_objective_scale(objective: Objective, bounds: Bounds | None = None) -> float:
    """Return ``max(1.0, sum(|c_i| * M_i) + sum(|c_ij| * M_i * M_j))``.

    ``M = max(|lower|, |upper|)`` is the largest absolute value a variable
    can take (``1`` for binary, so the all-binary result is the Phase 1
    ``sum(|coefficients|)`` unchanged; ``x*x`` contributes ``M_i**2``). This
    is an upper bound on the objective's absolute value, which penalty
    strategies use to scale hard penalties (spec §18). The term lists are
    used verbatim (duplicates are not merged first): the result is an upper
    bound either way and the spec formula reads over the raw coefficient
    lists.
    """
    total = sum(
        abs(term.coefficient) * _magnitude(bounds, term.variable)
        for term in objective.linear_terms
    )
    total += sum(
        abs(term.coefficient)
        * _magnitude(bounds, term.variable1)
        * _magnitude(bounds, term.variable2)
        for term in objective.quadratic_terms
    )
    return max(1.0, total)


def _max_abs_affine(
    coefficients: dict[str, float], constant: float, bounds: Bounds | None = None
) -> float:
    """Return ``max |sum(c_i * y_i) + constant|`` over the variables' ranges.

    An affine function of independent box-bounded variables attains its
    extremes at the vertices where every variable sits at the bound that
    minimises / maximises its own term, so the maximum absolute value is
    exactly ``max(|lhs_min + constant|, |lhs_max + constant|)``.
    """
    lhs_min, lhs_max = lhs_bounds(coefficients, bounds)
    return max(abs(lhs_min + constant), abs(lhs_max + constant))


def compute_soft_energy_bound(constraint: Constraint, bounds: Bounds | None = None) -> float:
    """Upper bound on the energy one soft constraint can add to the BQM.

    Mirrors the compiler's squared form (spec §10.2): ``weight * (sum(a_i x_i)
    [+ slack] - rhs)^2``. The maximum of the squared affine term over every
    assignment (declared variables *and* the slack bits the compiler will
    generate) is ``weight * D^2`` with ``D`` from :func:`_max_abs_affine`;
    for a single constraint this is the exact maximum, not just a bound.

    Case analysis follows ``encode_slack`` so the two can never drift:

    - ``==``: no slack, ``D = max(|lhs_min - rhs|, |lhs_max - rhs|)``.
    - redundant inequality: the compiler emits nothing -> ``0``.
    - inequality with slack range ``S >= 0``: the slack bits add exactly
      ``S`` to the normalized lhs maximum.
    - trivially infeasible *soft* inequality: the compiler clamps to zero
      slack bits, so the bound uses no slack either.

    Returns ``0.0`` for a hard constraint: hard penalties are chosen by the
    penalty strategy, never bounded by a weight (spec §10.4).
    """
    if constraint.type != "soft" or constraint.weight is None:
        return 0.0
    weight = constraint.weight

    if constraint.operator == "==":
        coefficients = {
            variable: value
            for variable, value in accumulate_terms(constraint.terms).items()
            if value != 0.0
        }
        deviation = _max_abs_affine(coefficients, -constraint.rhs, bounds)
        return weight * deviation * deviation

    analysis = analyze_inequality(constraint, bounds)
    if analysis.redundant:
        return 0.0
    slack_range = analysis.slack_range
    assert slack_range is not None  # non-redundant analysis always sets it
    slack_total = float(sum(compute_slack_coefficients(max(slack_range, 0))))
    coefficients = dict(analysis.coefficients)
    if slack_total:
        # A binary pseudo-variable standing for "all slack bits set"; it is
        # never in ``bounds`` and therefore ranges over 0..1 as intended.
        coefficients["__slack_total"] = slack_total
    deviation = _max_abs_affine(coefficients, -analysis.rhs, bounds)
    return weight * deviation * deviation


def compute_penalty_scale(problem: OptimizationProblem) -> float:
    """Return ``objective_scale + sum(soft energy bounds)`` (spec §18).

    The hard penalty must dominate the *whole* non-penalty energy landscape,
    which is the objective plus every soft term the compiler emits. With
    ``lambda > penalty_scale`` any assignment violating a hard constraint by
    at least one unit costs more than the best feasible assignment can gain.

    ``objective_scale`` itself stays objective-only (it is what the agent
    compares soft weights against); this function only *adds* the soft
    contribution, it never feeds a weight into the penalty as a value. For a
    problem without soft constraints the result equals ``objective_scale``
    exactly, so Phase 1 behaviour is unchanged. Both parts use the problem's
    variable bounds (3b §8).
    """
    bounds = variable_bounds(problem)
    soft_bound = sum(
        compute_soft_energy_bound(constraint, bounds) for constraint in problem.constraints
    )
    return compute_objective_scale(problem.objective, bounds) + soft_bound


@dataclass(frozen=True)
class InequalityAnalysis:
    """Pure-arithmetic view of one inequality constraint.

    ``coefficients`` and ``rhs`` are normalized to the ``<=`` form (a ``>=``
    constraint has both sides multiplied by -1) with zero coefficients
    dropped. ``slack_range`` keeps the raw ``int(round(rhs - lhs_min))``
    value — it is negative for a trivially infeasible constraint (a hard one
    is rejected by the validator before compilation; the compiler clamps a
    soft one) and is ``None`` when ``redundant``.
    """

    coefficients: dict[str, float]
    rhs: float
    lhs_min: float
    lhs_max: float
    redundant: bool
    slack_range: int | None


def analyze_inequality(
    constraint: Constraint, bounds: Bounds | None = None
) -> InequalityAnalysis:
    """Analyze a ``<=`` or ``>=`` constraint without building anything."""
    if constraint.operator not in ("<=", ">="):
        raise ValueError(
            f"analyze_inequality only handles inequalities, got "
            f"{constraint.operator!r} for constraint {constraint.id}"
        )

    coefficients = {
        variable: value
        for variable, value in accumulate_terms(constraint.terms).items()
        if value != 0.0
    }
    rhs = constraint.rhs
    if constraint.operator == ">=":
        coefficients = {variable: -value for variable, value in coefficients.items()}
        rhs = -rhs

    lhs_min, lhs_max = lhs_bounds(coefficients, bounds)
    redundant = lhs_max <= rhs
    slack_range = None if redundant else int(round(rhs - lhs_min))
    return InequalityAnalysis(
        coefficients=coefficients,
        rhs=rhs,
        lhs_min=lhs_min,
        lhs_max=lhs_max,
        redundant=redundant,
        slack_range=slack_range,
    )


def count_slack_bits(constraint: Constraint, bounds: Bounds | None = None) -> int:
    """Number of slack variables compilation will generate for ``constraint``.

    Equality constraints and redundant inequalities need none. A trivially
    infeasible *soft* inequality counts as zero bits, matching the
    compiler's explicit clamp in ``encode_slack``. A trivially infeasible
    *hard* inequality is never silently clamped: ``validate_problem`` rejects
    it with TRIVIALLY_INFEASIBLE before any estimate is made, so reaching it
    here is a programming error and raises. The count goes through
    :func:`compute_slack_coefficients` — the same function ``encode_slack``
    uses — so the two can never drift.
    """
    if constraint.operator not in ("<=", ">="):
        return 0
    analysis = analyze_inequality(constraint, bounds)
    if analysis.redundant:
        return 0
    slack_range = analysis.slack_range
    assert slack_range is not None  # non-redundant analysis always sets it
    if slack_range < 0:
        if constraint.type == "hard":
            raise ValueError(
                f"Hard constraint {constraint.id} is trivially infeasible; "
                "validate_problem must reject it before estimating"
            )
        return 0
    return len(compute_slack_coefficients(slack_range))


def constraint_bit_count(constraint: Constraint, bounds: Bounds | None = None) -> int:
    """Number of compiled bits one constraint's penalty couples together.

    Every distinct variable the constraint mentions contributes its encoding
    bits (one for a binary variable, :func:`integer_encoding_bits` for an
    integer one) plus the slack bits the constraint generates. On the BQM
    path a squared penalty forms a clique over exactly these bits, which is
    what the density warning and :func:`estimate_encoded_interactions` need.
    """
    names = {term.variable for term in constraint.terms}
    business_bits = sum(integer_encoding_bits(*_bounds_of(bounds, name)) for name in names)
    return business_bits + count_slack_bits(constraint, bounds)


def estimate_compiled_variables(problem: OptimizationProblem) -> int:
    """Estimate the BQM-path compiled variable count (3b §8).

    ``binary variables + sum(integer encoding bits) + sum(slack bits)`` —
    pure arithmetic (spec §20), no BQM is built. For a problem that passes
    ``validate_problem`` this equals ``BQMCompiler.compile(...).num_variables``
    exactly: the compiler adds one bit per binary variable, the encoding
    bits of every integer variable, and one variable per slack bit.
    """
    bounds = variable_bounds(problem)
    declared = sum(
        integer_encoding_bits(*bounds[name])
        for name in {variable.name for variable in problem.variables}
    )
    slack_bits = sum(
        count_slack_bits(constraint, bounds) for constraint in problem.constraints
    )
    return declared + slack_bits


def estimate_cqm_variables(problem: OptimizationProblem) -> int:
    """Estimate the CQM-path compiled variable count (3b §9.4, §15.3).

    Every declared variable is native, so the count is ``len(variables)``
    plus one integer slack per *soft* inequality that the CQM compiler
    writes in objective form: it mentions at least one integer variable (in
    its accumulated, non-zero coefficients), is not redundant, and has a
    positive slack range. Hard constraints, equalities, binary-only soft
    constraints and clamped (``slack_range <= 0``) ones add nothing. For an
    all-binary problem this is the plain variable count, as in 3a.
    """
    bounds = variable_bounds(problem)
    integer_names = {
        variable.name for variable in problem.variables if variable.type == "integer"
    }
    slack_variables = 0
    for constraint in problem.constraints:
        if constraint.type != "soft" or constraint.operator not in ("<=", ">="):
            continue
        analysis = analyze_inequality(constraint, bounds)
        if analysis.redundant or analysis.slack_range is None:
            continue
        if analysis.slack_range <= 0:
            continue
        if integer_names.isdisjoint(analysis.coefficients):
            continue
        slack_variables += 1
    return len(problem.variables) + slack_variables


def estimate_encoded_interactions(problem: OptimizationProblem) -> int:
    """Upper bound on the quadratic interactions the BQM path produces (3b §8).

    Binary-encoding an integer variable turns every product it takes part
    in into a product of bit vectors: an objective term ``c * x * y``
    contributes ``bits(x) * bits(y)`` interactions (``x * x`` contributes
    ``bits(x) * (bits(x) - 1) / 2``), and each non-redundant constraint's
    squared penalty couples all of its :func:`constraint_bit_count` bits
    ``B`` pairwise, ``B * (B - 1) / 2``. Drives the INTEGER_QUADRATIC_BLOWUP
    warning; it deliberately ignores that different terms may share a pair.
    """
    bounds = variable_bounds(problem)
    bits = {name: integer_encoding_bits(lower, upper) for name, (lower, upper) in bounds.items()}

    total = 0
    for term in problem.objective.quadratic_terms:
        bits_1 = bits.get(term.variable1, 1)
        bits_2 = bits.get(term.variable2, 1)
        if term.variable1 == term.variable2:
            total += bits_1 * (bits_1 - 1) // 2
        else:
            total += bits_1 * bits_2
    for constraint in problem.constraints:
        if constraint.operator in ("<=", ">=") and analyze_inequality(
            constraint, bounds
        ).redundant:
            continue
        clique = constraint_bit_count(constraint, bounds)
        total += clique * (clique - 1) // 2
    return total

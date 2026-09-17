"""Problem-shape helpers behind solver routing (2026-09-17).

``effective_hard_constraints`` / ``coupled_variable_pairs`` /
``estimate_interaction_density`` / ``is_large_dense`` /
``is_penalty_dominated`` are the pure-arithmetic shapes that a backend's
``strong_on_large_dense`` and ``weak_on_penalty_dominated`` declarations are
matched against, so they decide which backend ``recommend`` names first. The
thresholds are a judgement call; what must not drift silently is *where* the
boundary sits and what counts as being on either side of it, so both are
pinned here by enumeration rather than by one happy-path example.

The 500-variable instances are expensive to validate (a fully dense one is
~125k quadratic terms), so every test in this module shares the single
module-scoped ``dense_instances`` build.
"""

import itertools

import pytest

from annealbridge.models import Constraint, LinearTerm, OptimizationProblem
from annealbridge.validation.estimates import (
    DENSE_INTERACTION_RATIO_THRESHOLD,
    LARGE_DENSE_VARIABLES_THRESHOLD,
    coupled_variable_pairs,
    effective_hard_constraints,
    estimate_compiled_variables,
    estimate_interaction_density,
    is_large_dense,
    is_penalty_dominated,
)

# --------------------------------------------------------------------------
# Problem construction
# --------------------------------------------------------------------------


def make_problem(
    variable_count: int,
    *,
    quadratic_pairs=(),
    quadratic_terms=(),
    linear_variables=(),
    constraints=(),
) -> OptimizationProblem:
    """A binary problem with ``variable_count`` variables named ``x0..``.

    ``quadratic_pairs`` are index pairs with coefficient 1.0;
    ``quadratic_terms`` are ``(first, second, coefficient)`` triples for the
    tests that need a coefficient other than 1.
    """
    return OptimizationProblem.model_validate(
        {
            "name": f"shape-{variable_count}",
            "variables": [
                {"name": f"x{index}", "type": "binary"} for index in range(variable_count)
            ],
            "objective": {
                "direction": "minimize",
                "linear_terms": [
                    {"variable": f"x{index}", "coefficient": 1.0} for index in linear_variables
                ],
                "quadratic_terms": [
                    {"variable1": f"x{first}", "variable2": f"x{second}", "coefficient": 1.0}
                    for first, second in quadratic_pairs
                ]
                + [
                    {
                        "variable1": f"x{first}",
                        "variable2": f"x{second}",
                        "coefficient": coefficient,
                    }
                    for first, second, coefficient in quadratic_terms
                ],
            },
            "constraints": list(constraints),
        }
    )


def equality_constraint(
    indices, *, constraint_type: str = "hard", identifier: str = "c1", coefficient: float = 1.0
) -> Constraint:
    """An ``==`` constraint over ``indices``.

    Equality generates no slack bit and is never redundant, so its compiled
    clique is exactly the business variables it mentions -- which keeps the
    expected pair counts below readable.
    """
    return Constraint(
        id=identifier,
        type=constraint_type,
        terms=[LinearTerm(variable=f"x{index}", coefficient=coefficient) for index in indices],
        operator="==",
        rhs=1,
        weight=None if constraint_type == "hard" else 1.0,
    )


def inequality_constraint(
    indices, operator: str, rhs: int, *, constraint_type: str = "hard", identifier: str = "c1"
) -> Constraint:
    """A ``<=`` / ``>=`` constraint over ``indices`` with unit coefficients."""
    return Constraint(
        id=identifier,
        type=constraint_type,
        terms=[LinearTerm(variable=f"x{index}", coefficient=1) for index in indices],
        operator=operator,
        rhs=rhs,
        weight=None if constraint_type == "hard" else 1.0,
    )


# Redundant under binary bounds: one variable can never exceed 1, so the
# compiler emits no penalty and no slack for it.
REDUNDANT_HARD = inequality_constraint([0], "<=", 1, identifier="redundant")


def pair_count(variable_count: int) -> int:
    return variable_count * (variable_count - 1) // 2


def with_constraints(problem: OptimizationProblem, *constraints: Constraint):
    """A copy of ``problem`` carrying ``constraints``.

    ``model_copy``, not ``model_validate``, so the shared dense instance's
    ~125k quadratic terms are not re-validated once per test.
    """
    return problem.model_copy(update={"constraints": list(constraints)})


@pytest.fixture(scope="module")
def dense_instances():
    """The 500-variable instances, built once for the whole module."""
    pairs = list(itertools.combinations(range(500), 2))
    full = len(pairs)
    assert full == pair_count(500) == 124_750
    half = full // 2
    assert half * 2 == full  # so ``half`` pairs is an exact density of 0.5
    return {
        # Fully dense: density 1.0.
        "dense_500": make_problem(500, quadratic_pairs=pairs),
        # One variable short of the threshold, also fully dense.
        "dense_499": make_problem(499, quadratic_pairs=itertools.combinations(range(499), 2)),
        # Exactly at the density threshold, and one pair below it.
        "half_500": make_problem(500, quadratic_pairs=pairs[:half]),
        "below_half_500": make_problem(500, quadratic_pairs=pairs[: half - 1]),
        # Clearly sparse: 40%.
        "sparse_500": make_problem(500, quadratic_pairs=pairs[: int(full * 0.4)]),
    }


# --------------------------------------------------------------------------
# effective_hard_constraints
# --------------------------------------------------------------------------


class TestEffectiveHardConstraints:
    def test_no_constraints(self):
        assert effective_hard_constraints(make_problem(3)) == []

    def test_soft_only(self):
        problem = with_constraints(
            make_problem(3), equality_constraint([0, 1], constraint_type="soft")
        )
        assert effective_hard_constraints(problem) == []

    @pytest.mark.parametrize(
        "types",
        [("hard",), ("soft", "hard"), ("hard", "soft")],
        ids=["hard-only", "soft-then-hard", "hard-then-soft"],
    )
    def test_any_effective_hard_constraint_counts(self, types):
        problem = with_constraints(
            make_problem(3),
            *[
                equality_constraint([0, 1], constraint_type=kind, identifier=f"c{index}")
                for index, kind in enumerate(types)
            ],
        )
        assert [c.id for c in effective_hard_constraints(problem)] == [
            f"c{index}" for index, kind in enumerate(types) if kind == "hard"
        ]

    @pytest.mark.parametrize(
        "constraint",
        [
            REDUNDANT_HARD,
            inequality_constraint([0, 1, 2], "<=", 3, identifier="all-may-be-one"),
            inequality_constraint([0, 1], ">=", 0, identifier="never-below-zero"),
        ],
        ids=["x<=1", "sum<=count", "sum>=0"],
    )
    def test_a_redundant_inequality_is_not_effective(self, constraint):
        # The compiler emits nothing for it (2026-09-17 review): declared, but
        # no penalty term reaches the model.
        problem = with_constraints(make_problem(3), constraint)
        assert effective_hard_constraints(problem) == []

    def test_a_binding_inequality_is_effective(self):
        problem = with_constraints(make_problem(3), inequality_constraint([0, 1], "<=", 1))
        assert [c.id for c in effective_hard_constraints(problem)] == ["c1"]

    def test_all_zero_coefficients_are_not_effective(self):
        problem = with_constraints(make_problem(3), equality_constraint([0, 1], coefficient=0.0))
        assert effective_hard_constraints(problem) == []

    def test_the_redundant_one_is_dropped_and_the_binding_one_kept(self):
        problem = with_constraints(
            make_problem(3),
            REDUNDANT_HARD,
            inequality_constraint([0, 1], "<=", 1, identifier="binding"),
        )
        assert [c.id for c in effective_hard_constraints(problem)] == ["binding"]


# --------------------------------------------------------------------------
# coupled_variable_pairs / estimate_interaction_density
# --------------------------------------------------------------------------


class TestInteractionDensity:
    @pytest.mark.parametrize("variable_count", [0, 1])
    def test_fewer_than_two_variables_have_no_pair(self, variable_count):
        # No pair exists, so there is nothing to divide by: 0.0, neither a
        # ZeroDivisionError nor a vacuous 1.0.
        problem = make_problem(variable_count)
        assert coupled_variable_pairs(problem) == set()
        assert estimate_interaction_density(problem) == 0.0

    @pytest.mark.parametrize("variable_count", [2, 3, 5, 10, 33])
    def test_fully_dense_is_exactly_one(self, variable_count):
        problem = make_problem(
            variable_count, quadratic_pairs=itertools.combinations(range(variable_count), 2)
        )
        assert len(coupled_variable_pairs(problem)) == pair_count(variable_count)
        assert estimate_interaction_density(problem) == 1.0

    @pytest.mark.parametrize("variable_count", [2, 5, 10])
    def test_only_linear_terms_is_zero(self, variable_count):
        problem = make_problem(variable_count, linear_variables=range(variable_count))
        assert estimate_interaction_density(problem) == 0.0

    def test_a_pair_is_counted_once_however_often_it_is_mentioned(self):
        # Every pair is both a quadratic term and part of the constraint's
        # clique (2026-09-17 review: the term count would be twice the pair
        # count, and repeating one term could fake a dense model).
        problem = with_constraints(
            make_problem(
                4,
                quadratic_pairs=list(itertools.combinations(range(4), 2)) * 3,
            ),
            equality_constraint(range(4)),
        )
        assert len(coupled_variable_pairs(problem)) == pair_count(4)
        assert estimate_interaction_density(problem) == 1.0

    def test_one_pair_repeated_many_times_is_still_one_pair(self):
        problem = make_problem(10, quadratic_pairs=[(0, 1)] * 44)
        assert coupled_variable_pairs(problem) == {frozenset({"x0", "x1"})}
        assert estimate_interaction_density(problem) == pytest.approx(1 / 45)

    def test_coefficients_that_cancel_leave_no_edge(self):
        problem = make_problem(3, quadratic_terms=[(0, 1, 2.0), (1, 0, -2.0), (1, 2, 0.0)])
        assert coupled_variable_pairs(problem) == set()
        assert estimate_interaction_density(problem) == 0.0

    def test_an_integer_square_couples_no_pair(self):
        problem = OptimizationProblem.model_validate(
            {
                "version": "1.1",
                "name": "square",
                "variables": [
                    {"name": "a", "type": "integer", "lower_bound": 0, "upper_bound": 7},
                    {"name": "b", "type": "integer", "lower_bound": 0, "upper_bound": 7},
                ],
                "objective": {
                    "direction": "minimize",
                    "linear_terms": [{"variable": "a", "coefficient": 1}],
                    "quadratic_terms": [{"variable1": "a", "variable2": "a", "coefficient": 1}],
                },
                "constraints": [],
            }
        )
        assert coupled_variable_pairs(problem) == set()
        assert estimate_interaction_density(problem) == 0.0

    def test_density_is_over_declared_variables_not_compiled_bits(self):
        # Two 3-bit integers coupled once: one pair of one, whatever the
        # encoding does to the compiled size.
        problem = OptimizationProblem.model_validate(
            {
                "version": "1.1",
                "name": "integers",
                "variables": [
                    {"name": "a", "type": "integer", "lower_bound": 0, "upper_bound": 7},
                    {"name": "b", "type": "integer", "lower_bound": 0, "upper_bound": 7},
                ],
                "objective": {
                    "direction": "minimize",
                    "linear_terms": [{"variable": "a", "coefficient": 1}],
                    "quadratic_terms": [{"variable1": "a", "variable2": "b", "coefficient": 1}],
                },
                "constraints": [],
            }
        )
        assert estimate_compiled_variables(problem) == 6
        assert estimate_interaction_density(problem) == 1.0

    @pytest.mark.parametrize("clique_size", [2, 3, 5, 10])
    @pytest.mark.parametrize("constraint_type", ["hard", "soft"])
    def test_one_equality_constraint_alone_sets_the_density(self, clique_size, constraint_type):
        variable_count = 10
        problem = with_constraints(
            make_problem(variable_count),
            equality_constraint(range(clique_size), constraint_type=constraint_type),
        )
        assert len(coupled_variable_pairs(problem)) == pair_count(clique_size)
        assert estimate_interaction_density(problem) == pytest.approx(
            pair_count(clique_size) / pair_count(variable_count)
        )

    def test_a_redundant_constraint_adds_no_pair(self):
        problem = with_constraints(
            make_problem(4), inequality_constraint([0, 1, 2, 3], "<=", 4)
        )
        assert coupled_variable_pairs(problem) == set()

    def test_a_zero_coefficient_variable_is_outside_the_clique(self):
        constraint = Constraint(
            id="c1",
            type="hard",
            terms=[
                LinearTerm(variable="x0", coefficient=1.0),
                LinearTerm(variable="x1", coefficient=1.0),
                LinearTerm(variable="x2", coefficient=0.0),
            ],
            operator="==",
            rhs=1,
        )
        problem = with_constraints(make_problem(3), constraint)
        assert coupled_variable_pairs(problem) == {frozenset({"x0", "x1"})}


# --------------------------------------------------------------------------
# is_large_dense
# --------------------------------------------------------------------------


class TestIsLargeDense:
    def test_thresholds_are_the_ones_the_boundary_tests_assume(self):
        assert LARGE_DENSE_VARIABLES_THRESHOLD == 500
        assert DENSE_INTERACTION_RATIO_THRESHOLD == 0.5

    def test_one_variable_below_the_threshold_is_not_large(self, dense_instances):
        problem = dense_instances["dense_499"]
        assert estimate_compiled_variables(problem) == 499
        assert estimate_interaction_density(problem) == 1.0
        assert is_large_dense(problem, 499, "bqm") is False

    def test_at_the_threshold_it_is(self, dense_instances):
        problem = dense_instances["dense_500"]
        assert estimate_compiled_variables(problem) == 500
        assert estimate_interaction_density(problem) == 1.0
        assert is_large_dense(problem, 500, "bqm") is True

    @pytest.mark.parametrize(
        ("key", "expected_density", "expected"),
        [
            ("sparse_500", 0.4, False),
            ("below_half_500", None, False),
            ("half_500", 0.5, True),
        ],
        ids=["forty-percent", "just-below-half", "exactly-half"],
    )
    def test_the_density_boundary(self, dense_instances, key, expected_density, expected):
        problem = dense_instances[key]
        density = estimate_interaction_density(problem)
        if expected_density is None:
            # One pair short of the threshold: below it, but only just.
            assert density < DENSE_INTERACTION_RATIO_THRESHOLD
            assert density == pytest.approx(0.5, abs=1e-4)
        else:
            assert density == pytest.approx(expected_density)
        assert is_large_dense(problem, 500, "bqm") is expected

    def test_the_size_is_the_callers_compiled_estimate(self, dense_instances):
        # Slack and integer bits count towards the size; the density does
        # not change with the estimate.
        problem = dense_instances["dense_499"]
        assert is_large_dense(problem, 499, "bqm") is False
        assert is_large_dense(problem, 500, "bqm") is True

    @pytest.mark.parametrize("model_type", ["cqm", None])
    def test_only_the_bqm_path_has_the_shape(self, dense_instances, model_type):
        # 2026-09-17 review: the density and the size estimate are bqm-path
        # quantities, and the strength was measured there.
        assert is_large_dense(dense_instances["dense_500"], 500, model_type) is False

    def test_a_hard_constraint_disqualifies_it(self, dense_instances):
        # The measured strength was on unconstrained dense models; a hard
        # constraint makes it the penalty-dominated shape instead.
        problem = with_constraints(dense_instances["dense_500"], equality_constraint([0, 1]))
        assert estimate_interaction_density(problem) == 1.0
        assert is_large_dense(problem, 500, "bqm") is False

    def test_a_redundant_hard_constraint_does_not(self, dense_instances):
        problem = with_constraints(dense_instances["dense_500"], REDUNDANT_HARD)
        assert is_large_dense(problem, 500, "bqm") is True

    def test_a_soft_constraint_does_not(self, dense_instances):
        problem = with_constraints(
            dense_instances["dense_500"],
            equality_constraint([0, 1], constraint_type="soft"),
        )
        assert effective_hard_constraints(problem) == []
        assert is_large_dense(problem, 500, "bqm") is True

    def test_one_hard_constraint_among_soft_ones_still_disqualifies(self, dense_instances):
        problem = with_constraints(
            dense_instances["dense_500"],
            equality_constraint([0, 1], constraint_type="soft", identifier="c1"),
            equality_constraint([2, 3], constraint_type="hard", identifier="c2"),
        )
        assert is_large_dense(problem, 500, "bqm") is False

    def test_a_small_dense_problem_is_not_large(self):
        problem = make_problem(10, quadratic_pairs=itertools.combinations(range(10), 2))
        assert estimate_interaction_density(problem) == 1.0
        assert is_large_dense(problem, 10, "bqm") is False


# --------------------------------------------------------------------------
# is_penalty_dominated
# --------------------------------------------------------------------------


class TestIsPenaltyDominated:
    @pytest.mark.parametrize(
        ("model_type", "constraint_types", "expected"),
        [
            ("bqm", ("hard",), True),
            ("bqm", ("soft", "hard"), True),
            ("bqm", ("soft",), False),
            ("bqm", (), False),
            # The cqm path keeps a hard constraint native: no penalty term,
            # so never this shape however many hard constraints there are.
            ("cqm", ("hard",), False),
            ("cqm", ("soft",), False),
            ("cqm", (), False),
            # No model type decided yet: not the shape either.
            (None, ("hard",), False),
            (None, (), False),
        ],
        ids=[
            "bqm-hard",
            "bqm-soft-and-hard",
            "bqm-soft-only",
            "bqm-unconstrained",
            "cqm-hard",
            "cqm-soft-only",
            "cqm-unconstrained",
            "none-hard",
            "none-unconstrained",
        ],
    )
    def test_shape_is_bqm_plus_a_hard_constraint(self, model_type, constraint_types, expected):
        problem = with_constraints(
            make_problem(4),
            *[
                equality_constraint([0, 1], constraint_type=kind, identifier=f"c{index}")
                for index, kind in enumerate(constraint_types)
            ],
        )
        assert is_penalty_dominated(problem, model_type) is expected

    def test_a_redundant_hard_constraint_alone_is_not_the_shape(self):
        # 2026-09-17 review: no penalty term is emitted for it.
        problem = with_constraints(make_problem(4), REDUNDANT_HARD)
        assert is_penalty_dominated(problem, "bqm") is False

    def test_it_does_not_look_at_size_or_density(self, dense_instances):
        # Only the effective hard constraint matters, so a large dense
        # problem with one hard constraint is this shape and not the other.
        problem = with_constraints(dense_instances["dense_500"], equality_constraint([0, 1]))
        assert is_penalty_dominated(problem, "bqm") is True
        assert is_large_dense(problem, 500, "bqm") is False

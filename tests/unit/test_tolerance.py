"""Tests for the shared hybrid tolerance (spec §23.1, review F-05).

``tol(actual, rhs) = max(1e-8, 1e-12 * max(|actual|, |rhs|))`` is the single
rule every constraint comparison uses. The scalar and the numpy version must
stay bit-identical, because the batch validator ranks candidates with one and
the full validator reports them with the other.
"""

import numpy as np

from annealbridge.validation import (
    ABSOLUTE_TOLERANCE,
    EPSILON,
    RELATIVE_TOLERANCE,
    tolerance,
    tolerance_array,
)


class TestConstants:
    def test_absolute_floor_is_the_original_epsilon(self):
        assert EPSILON == ABSOLUTE_TOLERANCE
        assert ABSOLUTE_TOLERANCE == 1e-8

    def test_relative_part(self):
        assert RELATIVE_TOLERANCE == 1e-12


class TestSmallMagnitudes:
    """Below the crossover the rule is exactly the old absolute 1e-8."""

    def test_magnitude_one(self):
        assert tolerance(1.0, 1.0) == 1e-8

    def test_fractional_magnitude(self):
        assert tolerance(0.3, 0.3) == 1e-8

    def test_zero(self):
        assert tolerance(0.0, 0.0) == 1e-8

    def test_at_the_crossover_magnitude(self):
        assert tolerance(1e4, 1e4) == 1e-8

    def test_negative_small_values(self):
        assert tolerance(-1.0, -0.5) == 1e-8


class TestLargeMagnitudes:
    """Above the crossover the relative part takes over exactly."""

    def test_relative_part_at_1e9(self):
        # The formula multiplies the larger magnitude, so this is exact.
        assert tolerance(1e9, 1e9 + 0.3) == 1e-12 * (1e9 + 0.3)

    def test_relative_part_at_1e12(self):
        assert tolerance(1e12, 0.0) == 1e-12 * 1e12

    def test_takes_the_larger_magnitude(self):
        assert tolerance(1e9, 0.0) == 1e-12 * 1e9
        assert tolerance(1e9, 1.0) == tolerance(1e9, 0.0)

    def test_is_symmetric_in_its_arguments(self):
        assert tolerance(1e9, 0.0) == tolerance(0.0, 1e9)

    def test_uses_absolute_values(self):
        assert tolerance(-1e9, 0.0) == tolerance(1e9, 0.0)
        assert tolerance(0.0, -1e9) == tolerance(1e9, 0.0)


class TestCrossover:
    """``1e-12 * m == 1e-8`` at ``m = 1e4``: the floor holds up to there and
    the relative part grows from there on."""

    def test_floor_still_wins_at_1e4(self):
        assert tolerance(1e4, 0.0) == 1e-8

    def test_relative_part_wins_just_above(self):
        assert tolerance(2e4, 0.0) == 2e-8
        assert tolerance(0.0, -2e4) == 2e-8

    def test_below_the_crossover_stays_at_the_floor(self):
        assert tolerance(9e3, 0.0) == 1e-8


class TestReturnTypes:
    def test_scalar_returns_a_python_float(self):
        assert type(tolerance(1.0, 1.0)) is float
        assert type(tolerance(1e12, 0.0)) is float

    def test_array_returns_float64_ndarray(self):
        result = tolerance_array(np.array([1.0, 1e12]), 0.0)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.shape == (2,)


# --------------------------------------------------------------------------
# the two kernels must agree bit for bit
# --------------------------------------------------------------------------

FIXED_VALUES = [
    0.0,
    -0.0,
    1e-8,
    -1e-8,
    5e-9,
    2e-8,
    0.1,
    -0.3,
    1.0,
    -1.0,
    1e4,
    -1e4,
    2e4,
    1e8,
    1e9,
    -1e9,
    1e12,
    -1e12,
]


def _mixed_values(rng: np.random.Generator, count: int) -> list[float]:
    """Values spanning every side of the crossover, plus the exact edges."""
    return (
        rng.uniform(-1.0, 1.0, count).tolist()
        + rng.uniform(-1e8, 1e8, count).tolist()
        + rng.uniform(-1e12, 1e12, count).tolist()
        + list(FIXED_VALUES)
    )


class TestScalarAndArrayAgree:
    def test_every_pair_matches_element_wise(self):
        rng = np.random.default_rng(20260909)
        actuals = _mixed_values(rng, 700)
        rhs_values = _mixed_values(rng, 700)
        assert len(actuals) >= 2000
        for actual, rhs in zip(actuals, rhs_values):
            expected = tolerance(actual, rhs)
            got = float(tolerance_array(np.array([actual]), rhs)[0])
            assert got == expected, (actual, rhs)

    def test_vectorised_call_matches_the_scalar_loop(self):
        rng = np.random.default_rng(20260909)
        actuals = _mixed_values(rng, 700)
        matrix = np.array(actuals, dtype=np.float64)
        for rhs in [0.0, 1.0, -1e-8, 1e4, -1e9, 1e12, 0.3]:
            vectorised = tolerance_array(matrix, rhs)
            scalar = np.array([tolerance(actual, rhs) for actual in actuals])
            assert np.array_equal(vectorised, scalar), rhs

    def test_multi_dimensional_input_keeps_the_same_values(self):
        rng = np.random.default_rng(20260909)
        actuals = _mixed_values(rng, 100)
        matrix = np.array(actuals, dtype=np.float64).reshape(-1, 1)
        result = tolerance_array(matrix, 1e9)
        assert result.shape == matrix.shape
        assert np.array_equal(
            result.ravel(), np.array([tolerance(actual, 1e9) for actual in actuals])
        )

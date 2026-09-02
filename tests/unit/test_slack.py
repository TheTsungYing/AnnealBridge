"""Unit tests for binary slack encoding (spec §14, §33)."""

import itertools
import logging

import pytest

from annealbridge.compiler import compute_slack_coefficients, encode_slack
from annealbridge.exceptions import CompilationError
from annealbridge.models import Constraint


def make_constraint(
    operator: str,
    rhs: float,
    coefficients: list[tuple[str, float]],
    *,
    constraint_type: str = "hard",
    weight: float | None = None,
    constraint_id: str = "c1",
) -> Constraint:
    """Build a constraint from (variable, coefficient) pairs (duplicates allowed)."""
    payload: dict = {
        "id": constraint_id,
        "type": constraint_type,
        "terms": [
            {"variable": name, "coefficient": value} for name, value in coefficients
        ],
        "operator": operator,
        "rhs": rhs,
    }
    if weight is not None:
        payload["weight"] = weight
    return Constraint.model_validate(payload)


class TestComputeSlackCoefficients:
    def test_zero_range_has_no_bits(self):
        assert compute_slack_coefficients(0) == []

    def test_negative_range_rejected(self):
        with pytest.raises(ValueError):
            compute_slack_coefficients(-1)

    @pytest.mark.parametrize(
        ("slack_range", "expected"),
        [
            (1, [1]),
            (2, [1, 1]),
            (3, [1, 2]),
            (4, [1, 2, 1]),
            (5, [1, 2, 2]),
            (6, [1, 2, 3]),
            (7, [1, 2, 4]),
        ],
    )
    def test_hand_computed_coefficients(self, slack_range, expected):
        assert compute_slack_coefficients(slack_range) == expected

    @pytest.mark.parametrize("slack_range", list(range(1, 65)))
    def test_covers_exactly_zero_to_s(self, slack_range):
        # Every integer 0..S must be reachable and no bit combination may
        # exceed S (spec §14).
        coefficients = compute_slack_coefficients(slack_range)
        reachable = {
            sum(bit * value for bit, value in zip(bits, coefficients))
            for bits in itertools.product((0, 1), repeat=len(coefficients))
        }
        assert reachable == set(range(slack_range + 1))

    @pytest.mark.parametrize("slack_range", [1, 2, 3, 4, 7, 8, 15, 16, 100])
    def test_bit_count_is_ceil_log2(self, slack_range):
        assert len(compute_slack_coefficients(slack_range)) == slack_range.bit_length()


class TestEncodeSlackLessEqual:
    def test_basic_encoding(self):
        # 2*x1 + 3*x2 <= 4: lhs range [0, 5], S = 4 -> bits [1, 2, 1]
        encoding = encode_slack(
            make_constraint("<=", 4, [("x1", 2), ("x2", 3)], constraint_id="cap")
        )
        assert not encoding.redundant
        assert encoding.slack_range == 4
        assert encoding.coefficients == {"x1": 2.0, "x2": 3.0}
        assert encoding.constant == -4.0
        assert encoding.slack_coefficients == {
            "__slack_cap_0": 1,
            "__slack_cap_1": 2,
            "__slack_cap_2": 1,
        }

    def test_negative_coefficient_extends_slack_range(self):
        # -x1 + 2*x2 <= 1: lhs range [-1, 2], S = 1 - (-1) = 2
        encoding = encode_slack(make_constraint("<=", 1, [("x1", -1), ("x2", 2)]))
        assert encoding.slack_range == 2
        assert list(encoding.slack_coefficients.values()) == [1, 1]

    def test_zero_slack_range_forces_equality_at_bound(self):
        # x1 <= 0: lhs range [0, 1], S = 0 -> no bits, but not redundant
        encoding = encode_slack(make_constraint("<=", 0, [("x1", 1)]))
        assert not encoding.redundant
        assert encoding.slack_range == 0
        assert encoding.slack_coefficients == {}


class TestEncodeSlackGreaterEqual:
    def test_conversion_negates_both_sides(self):
        # 2*x1 + 3*x2 >= 2 becomes -2*x1 - 3*x2 <= -2; S = lhs_max - rhs = 3
        encoding = encode_slack(make_constraint(">=", 2, [("x1", 2), ("x2", 3)]))
        assert not encoding.redundant
        assert encoding.coefficients == {"x1": -2.0, "x2": -3.0}
        assert encoding.constant == 2.0
        assert encoding.slack_range == 3
        assert list(encoding.slack_coefficients.values()) == [1, 2]


class TestRedundantConstraints:
    def test_less_equal_always_satisfied(self):
        # x1 + x2 <= 5 with binary variables can never exceed 2
        encoding = encode_slack(make_constraint("<=", 5, [("x1", 1), ("x2", 1)]))
        assert encoding.redundant
        assert encoding.slack_range is None
        assert encoding.slack_coefficients == {}

    def test_greater_equal_always_satisfied(self):
        encoding = encode_slack(make_constraint(">=", 0, [("x1", 1), ("x2", 1)]))
        assert encoding.redundant
        assert encoding.slack_range is None
        assert encoding.slack_coefficients == {}


class TestDuplicateTermAccumulation:
    def test_range_uses_accumulated_coefficients(self):
        # [x: +2, x: -1] accumulates to x: +1; per-term ranges would give
        # [-1, 2] but the true range is [0, 1] -> redundant against rhs 1.
        encoding = encode_slack(make_constraint("<=", 1, [("x1", 2), ("x1", -1)]))
        assert encoding.redundant

    def test_accumulated_coefficient_drives_slack_range(self):
        encoding = encode_slack(make_constraint("<=", 0, [("x1", 2), ("x1", -1)]))
        assert not encoding.redundant
        assert encoding.coefficients == {"x1": 1.0}
        assert encoding.slack_range == 0


class TestTriviallyInfeasible:
    def test_hard_raises_compilation_error(self):
        # Accumulated coefficient is -1, so lhs range is [-1, 0] and rhs -2
        # can never be reached. The validator now judges this the same way
        # (it also sums repeated variables before taking the lhs range), so
        # this assertion is the compiler's own defence for callers that skip
        # validate_problem and call the encoder directly.
        with pytest.raises(CompilationError):
            encode_slack(make_constraint("<=", -2, [("x1", 1), ("x1", -2)]))

    def test_soft_clamps_to_zero_bits_with_warning(self, caplog):
        constraint = make_constraint(
            "<=",
            -2,
            [("x1", 1), ("x1", -2)],
            constraint_type="soft",
            weight=1.0,
        )
        with caplog.at_level(logging.WARNING, logger="annealbridge.compiler"):
            encoding = encode_slack(constraint)
        assert not encoding.redundant
        assert encoding.slack_range == 0
        assert encoding.slack_coefficients == {}
        assert any("can never be satisfied" in record.message for record in caplog.records)

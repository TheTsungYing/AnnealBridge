"""Tests keeping the error catalog in sync with Phase 2 spec §13.2."""

import pytest

from annealbridge.models import RECOMMENDED_ACTIONS, RETRYABLE_CODES, catalog_error

# The full set of error codes defined by Phase 2 spec §13.2.
EXPECTED_CODES = [
    "UNKNOWN_BACKEND",
    "BACKEND_NOT_INSTALLED",
    "REMOTE_DISABLED",
    "REMOTE_CREDENTIALS_MISSING",
    "BACKEND_DISABLED_BY_POLICY",
    "EXACT_VARIABLE_LIMIT",
    "QPU_READS_LIMIT",
    "QPU_ANNEALING_TIME_LIMIT",
    "REMOTE_TIME_LIMIT",
    "CONCURRENCY_LIMIT",
    "EMBEDDING_FAILED",
    "REMOTE_AUTH_FAILED",
    "REMOTE_TIMEOUT",
    "REMOTE_SOLVER_ERROR",
    "REMOTE_RETRIES_DISABLED",
    "SOLVER_ERROR",
    "DWAVE_CONFIG_INVALID",
    "BACKEND_UNAVAILABLE",
    # Phase 3a spec §20
    "BACKEND_CONFIG_INVALID",
    "NO_COMPILER_FOR_MODEL_TYPE",
    # Problem validator codes (Phase 1 spec §12)
    "UNKNOWN_VARIABLE",
    "DUPLICATE_VARIABLE",
    "RESERVED_VARIABLE_NAME",
    "DUPLICATE_CONSTRAINT_ID",
    "SELF_QUADRATIC_TERM",
    "NON_FINITE_COEFFICIENT",
    "EMPTY_CONSTRAINT",
    "NON_INTEGER_INEQUALITY",
    "HARD_CONSTRAINT_HAS_WEIGHT",
    "SOFT_CONSTRAINT_MISSING_WEIGHT",
    "INVALID_SOLVER_PREFERENCE",
    "TRIVIALLY_INFEASIBLE",
    "NO_VARIABLES",
    # Compilation
    "COMPILATION_FAILED",
]

# Codes whose failure is transient: the same request may succeed later.
EXPECTED_RETRYABLE_CODES = {
    "CONCURRENCY_LIMIT",
    "REMOTE_TIMEOUT",
    "REMOTE_SOLVER_ERROR",
}


class TestRecommendedActions:
    @pytest.mark.parametrize("code", EXPECTED_CODES)
    def test_code_present_with_non_empty_text(self, code):
        assert code in RECOMMENDED_ACTIONS
        action = RECOMMENDED_ACTIONS[code]
        assert isinstance(action, str)
        assert action.strip()

    def test_no_extra_codes(self):
        assert set(RECOMMENDED_ACTIONS) == set(EXPECTED_CODES)

    def test_expected_codes_are_unique(self):
        assert len(EXPECTED_CODES) == len(set(EXPECTED_CODES)) == 34


class TestRetryableCodes:
    def test_retryable_codes_are_exactly_the_transient_ones(self):
        assert set(RETRYABLE_CODES) == EXPECTED_RETRYABLE_CODES

    def test_retryable_codes_are_all_catalog_keys(self):
        assert EXPECTED_RETRYABLE_CODES <= set(RECOMMENDED_ACTIONS)
        for code in RETRYABLE_CODES:
            assert code in RECOMMENDED_ACTIONS


# The codes the problem validator is allowed to emit (Phase 1 spec §12).
EXPECTED_VALIDATOR_CODES = {
    "UNKNOWN_VARIABLE",
    "DUPLICATE_VARIABLE",
    "RESERVED_VARIABLE_NAME",
    "DUPLICATE_CONSTRAINT_ID",
    "SELF_QUADRATIC_TERM",
    "NON_FINITE_COEFFICIENT",
    "EMPTY_CONSTRAINT",
    "NON_INTEGER_INEQUALITY",
    "HARD_CONSTRAINT_HAS_WEIGHT",
    "SOFT_CONSTRAINT_MISSING_WEIGHT",
    "INVALID_SOLVER_PREFERENCE",
    "TRIVIALLY_INFEASIBLE",
    "NO_VARIABLES",
}


class TestValidatorCodesCovered:
    """Every code the validator emits must have catalog guidance."""

    @pytest.mark.parametrize("code", sorted(EXPECTED_VALIDATOR_CODES))
    def test_validator_code_has_non_empty_action(self, code):
        from annealbridge.validation.problem_validator import VALIDATOR_ERROR_CODES

        assert code in VALIDATOR_ERROR_CODES
        assert code in RECOMMENDED_ACTIONS
        action = RECOMMENDED_ACTIONS[code]
        assert isinstance(action, str)
        assert action.strip()

    def test_validator_error_codes_match_expected_set(self):
        from annealbridge.validation.problem_validator import VALIDATOR_ERROR_CODES

        assert set(VALIDATOR_ERROR_CODES) == EXPECTED_VALIDATOR_CODES


class TestCatalogError:
    def test_retryable_code_gets_retryable_and_action_and_path(self):
        error = catalog_error("REMOTE_TIMEOUT", "m", path="p")
        assert error.code == "REMOTE_TIMEOUT"
        assert error.message == "m"
        assert error.path == "p"
        assert error.retryable is True
        assert error.recommended_action == RECOMMENDED_ACTIONS["REMOTE_TIMEOUT"]

    def test_non_retryable_code_without_path(self):
        error = catalog_error("UNKNOWN_VARIABLE", "m")
        assert error.retryable is False
        assert error.path is None
        assert error.recommended_action == RECOMMENDED_ACTIONS["UNKNOWN_VARIABLE"]

    def test_unknown_code_has_no_action(self):
        error = catalog_error("NOT_A_CODE", "m")
        assert error.recommended_action is None
        assert error.retryable is False

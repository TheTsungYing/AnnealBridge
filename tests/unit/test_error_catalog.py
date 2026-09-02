"""Tests keeping the error catalog in sync with Phase 2 spec §13.2."""

import pytest

from annealbridge.models import RECOMMENDED_ACTIONS, RETRYABLE_CODES

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
        assert len(EXPECTED_CODES) == len(set(EXPECTED_CODES)) == 18


class TestRetryableCodes:
    def test_retryable_codes_are_exactly_the_transient_ones(self):
        assert set(RETRYABLE_CODES) == EXPECTED_RETRYABLE_CODES

    def test_retryable_codes_are_all_catalog_keys(self):
        assert EXPECTED_RETRYABLE_CODES <= set(RECOMMENDED_ACTIONS)
        for code in RETRYABLE_CODES:
            assert code in RECOMMENDED_ACTIONS

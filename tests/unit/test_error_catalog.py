"""Tests keeping the error catalog in sync with Phase 2 spec §13.2 and 3a §20."""

import ast
import re
from pathlib import Path

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
    # Phase 3b spec §19 (Fujitsu Digital Annealer remote codes)
    "REMOTE_QUOTA_EXCEEDED",
    "REMOTE_BUSY",
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
    # Problem validator codes for integer variables (Phase 3b spec §9.1)
    "INTEGER_BOUNDS_MISSING",
    "INTEGER_BOUNDS_INVALID",
    "BOUNDS_ON_BINARY",
    "INTEGER_RANGE_TOO_LARGE",
    "INTEGER_REQUIRES_VERSION_1_1",
    # Compilation
    "COMPILATION_FAILED",
]

# Codes whose guidance quotes an IR *contract constant* verbatim, as 3b spec
# §9.1 words them: the ±(2^31-1) integer range the IR itself fixes, the schema
# version string "1.1", and the 0/1 domain that *defines* a binary variable.
# None is a configurable threshold an operator can retune, so quoting one
# cannot leak a deployment's settings — which is what the no-numbers rule
# below exists to prevent. Everything else stays categorical.
CONTRACT_CONSTANT_CODES = frozenset(
    {"INTEGER_RANGE_TOO_LARGE", "INTEGER_REQUIRES_VERSION_1_1", "BOUNDS_ON_BINARY"}
)

# Codes whose failure is transient: the same request may succeed later.
EXPECTED_RETRYABLE_CODES = {
    "CONCURRENCY_LIMIT",
    "REMOTE_TIMEOUT",
    "REMOTE_SOLVER_ERROR",
    "REMOTE_BUSY",
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
        assert len(EXPECTED_CODES) == len(set(EXPECTED_CODES)) == 41


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
    # Phase 3b spec §9.1: integer variables.
    "INTEGER_BOUNDS_MISSING",
    "INTEGER_BOUNDS_INVALID",
    "BOUNDS_ON_BINARY",
    "INTEGER_RANGE_TOO_LARGE",
    "INTEGER_REQUIRES_VERSION_1_1",
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


# ---------------------------------------------------------------------------
# Phase 3a §20 / §30 step 10: every code the service, the validator and the
# solver layer can actually emit must have catalog guidance. The emitted set
# is collected from the source (AST) and from the shipped declarations, so a
# new ``catalog_error("NEW_CODE", ...)`` without a catalog entry fails here.
# ---------------------------------------------------------------------------

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "annealbridge"

# Where the service and the validator build errors and warnings.
EMITTING_PACKAGES = ["orchestration", "validation", "interfaces"]

# Functions whose first positional argument (or ``code=`` keyword) is a code.
ERROR_BUILDERS = frozenset({"catalog_error", "_error", "SolverExecutionError"})
WARNING_BUILDERS = frozenset({"_warning"})


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _literal_codes(files: list[Path], builders: frozenset[str]) -> dict[str, list[str]]:
    """``code -> [file:line, ...]`` for every literal code passed to ``builders``."""
    found: dict[str, list[str]] = {}
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in builders:
                continue
            code = None
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                code = node.args[0].value
            for keyword in node.keywords:
                if (
                    keyword.arg == "code"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    code = keyword.value.value
            if code is not None:
                where = f"{path.relative_to(SRC_ROOT).as_posix()}:{node.lineno}"
                found.setdefault(code, []).append(where)
    return found


def _package_files(*packages: str) -> list[Path]:
    files: list[Path] = []
    for package in packages:
        directory = SRC_ROOT / package
        assert directory.is_dir(), f"missing package directory: {directory}"
        files.extend(sorted(directory.rglob("*.py")))
    return files


def literal_error_codes() -> dict[str, list[str]]:
    return _literal_codes(_package_files(*EMITTING_PACKAGES, "solvers"), ERROR_BUILDERS)


def literal_warning_codes() -> dict[str, list[str]]:
    return _literal_codes(_package_files(*EMITTING_PACKAGES), WARNING_BUILDERS)


def declared_error_codes() -> dict[str, list[str]]:
    """Codes the service emits from *declarations* rather than literals.

    ``limit_error`` uses each backend's declared ``ParameterLimit.error_code``;
    the availability gate uses ``AVAILABILITY_MAP``'s defaults; solver
    exceptions are classified through the Ocean code tables, the Fujitsu DA
    HTTP tables (3b §20.8) and the fallback code.
    """
    from annealbridge.orchestration.limits import AVAILABILITY_MAP
    from annealbridge.solvers import SolverRegistry
    from annealbridge.solvers.fujitsu_da import (
        _BAD_REQUEST_MESSAGE_CODES,
        _HTTP_STATUS_CODES,
    )
    from annealbridge.solvers.metadata import REMOTE_ERROR_FALLBACK_CODE
    from annealbridge.solvers.ocean import (
        HYBRID_SAMPLE_EXCEPTION_CODES,
        SAMPLER_INIT_EXCEPTION_CODES,
    )

    found: dict[str, list[str]] = {}
    for category, (_status, code) in AVAILABILITY_MAP.items():
        found.setdefault(code, []).append(f"AVAILABILITY_MAP[{category!r}]")
    registry = SolverRegistry.default()
    for name in registry.names():
        for declaration in registry.get(name).capabilities.parameter_limits:
            found.setdefault(declaration.error_code, []).append(
                f"{name}.parameter_limits[{declaration.preference!r}]"
            )
    for table_name, table in (
        ("SAMPLER_INIT_EXCEPTION_CODES", SAMPLER_INIT_EXCEPTION_CODES),
        ("HYBRID_SAMPLE_EXCEPTION_CODES", HYBRID_SAMPLE_EXCEPTION_CODES),
        ("fujitsu_da._HTTP_STATUS_CODES", _HTTP_STATUS_CODES),
        ("fujitsu_da._BAD_REQUEST_MESSAGE_CODES", _BAD_REQUEST_MESSAGE_CODES),
    ):
        for key, code in table.items():
            found.setdefault(code, []).append(f"{table_name}[{key!r}]")
    found.setdefault(REMOTE_ERROR_FALLBACK_CODE, []).append("REMOTE_ERROR_FALLBACK_CODE")
    return found


class TestEmittedErrorCodesAreCatalogued:
    """3a §20: the coverage check spans every code the code base can emit."""

    def test_collector_sees_the_3a_codes(self):
        literal = literal_error_codes()
        declared = declared_error_codes()
        # Literal emitters in the service (3a §20 rows).
        assert "NO_COMPILER_FOR_MODEL_TYPE" in literal
        assert "UNKNOWN_BACKEND" in literal
        # Declaration-driven emitters.
        assert "BACKEND_CONFIG_INVALID" in declared
        assert "REMOTE_TIME_LIMIT" in declared
        assert "REMOTE_SOLVER_ERROR" in declared

    def test_collector_sees_the_fujitsu_da_http_codes(self):
        """3b §20.8: the DA status / message tables feed the coverage check."""
        declared = declared_error_codes()
        assert "REMOTE_QUOTA_EXCEEDED" in declared
        assert "REMOTE_BUSY" in declared
        assert "REMOTE_AUTH_FAILED" in declared

    def test_every_literal_code_has_a_recommended_action(self):
        missing = {
            code: where
            for code, where in literal_error_codes().items()
            if code not in RECOMMENDED_ACTIONS
        }
        assert missing == {}, f"codes emitted without catalog guidance: {missing}"

    def test_every_declared_code_has_a_recommended_action(self):
        missing = {
            code: where
            for code, where in declared_error_codes().items()
            if code not in RECOMMENDED_ACTIONS
        }
        assert missing == {}, f"codes emitted without catalog guidance: {missing}"

    def test_unknown_backend_is_shared_by_solve_and_validate(self):
        """§20: validate()'s UNKNOWN_BACKEND warning reuses the existing entry."""
        where = literal_error_codes()["UNKNOWN_BACKEND"]
        assert len(where) >= 2
        assert all(site.startswith("orchestration/optimizer.py") for site in where)

    def test_validator_error_codes_are_exactly_its_literal_codes(self):
        """``VALIDATOR_ERROR_CODES`` (the documented set) matches the source."""
        from annealbridge.validation.problem_validator import VALIDATOR_ERROR_CODES

        literal = _literal_codes(_package_files("validation"), ERROR_BUILDERS)
        assert set(literal) == set(VALIDATOR_ERROR_CODES)


class TestEmittedWarningCodesAreCatalogued:
    """Every ``_warning("CODE", ...)`` in the validator has fixed guidance."""

    def test_every_emitted_warning_has_guidance_and_none_is_stale(self):
        from annealbridge.validation.problem_validator import (
            _WARNING_RECOMMENDED_ACTIONS,
        )

        emitted = literal_warning_codes()
        assert emitted, "no warning emitters were found; the collector is broken"
        assert set(emitted) == set(_WARNING_RECOMMENDED_ACTIONS)
        for code, text in _WARNING_RECOMMENDED_ACTIONS.items():
            assert isinstance(text, str) and text.strip(), code

    def test_service_warnings_reuse_the_error_catalog(self):
        """§20: warnings the *service* adds (UNKNOWN_BACKEND, ...) are catalog codes."""
        for code in _literal_codes(_package_files("orchestration"), ERROR_BUILDERS):
            assert code in RECOMMENDED_ACTIONS, code


class TestGuidanceTextCarriesNoConfigValues:
    """§20 / Phase 2 §13.2: guidance is categorical and never embeds a limit."""

    @pytest.mark.parametrize(
        "code", [c for c in EXPECTED_CODES if c not in CONTRACT_CONSTANT_CODES]
    )
    def test_recommended_action_has_no_numbers(self, code):
        assert not re.search(r"\d", RECOMMENDED_ACTIONS[code]), code

    def test_contract_constant_texts_match_the_spec_wording(self):
        """The exempted texts quote the contract constant, and only that."""
        assert "±(2^31-1)" in RECOMMENDED_ACTIONS["INTEGER_RANGE_TOO_LARGE"]
        assert '"1.1"' in RECOMMENDED_ACTIONS["INTEGER_REQUIRES_VERSION_1_1"]
        assert "0/1" in RECOMMENDED_ACTIONS["BOUNDS_ON_BINARY"]

    def test_warning_guidance_has_no_numbers(self):
        from annealbridge.validation.problem_validator import (
            _WARNING_RECOMMENDED_ACTIONS,
        )

        for code, text in _WARNING_RECOMMENDED_ACTIONS.items():
            assert not re.search(r"\d", text), code

    def test_3a_texts_match_the_spec_wording(self):
        assert RECOMMENDED_ACTIONS["BACKEND_CONFIG_INVALID"].startswith(
            "The backend's configuration is present but invalid"
        )
        assert RECOMMENDED_ACTIONS["NO_COMPILER_FOR_MODEL_TYPE"].startswith(
            "The server has no compiler for the model types this backend accepts"
        )

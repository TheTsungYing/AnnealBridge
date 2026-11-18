"""``_prepare_attempts`` estimates the compiled size once (2026-11-18 review R-19).

An exhaustive backend that also declares a ``compiled_variable_limit`` has
two pre-compile ceilings: the policy's ``exact_max_variables`` and its own
declared maximum. Both are checked against one call of
``estimate_model_variables`` in ``annealbridge.orchestration.optimizer``
(it used to be called once per ceiling). The validation stage runs its own
estimate inside ``annealbridge.validation``; only the optimizer module's
name is counted here. Each ceiling still refuses an oversized problem with
``resource_limit_exceeded`` before the backend runs.
"""

import json

import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.models.capabilities import CompiledVariableLimit
from annealbridge.orchestration import ExecutionPolicy, OptimizationService
from annealbridge.orchestration import optimizer as optimizer_module
from annealbridge.solvers import SolverRegistry
from annealbridge.solvers.base import SolverCapabilities
from annealbridge.validation.estimates import estimate_model_variables
from tests.conftest import EXAMPLES_DIR
from tests.fakes.interruptible_backend import ScriptedInterruptBackend, route

NAME = "scripted_exhaustive_declared"
DECLARED_CODE = "SB_VARIABLE_LIMIT"


def knapsack_problem() -> OptimizationProblem:
    """``examples/knapsack.json``: optimum {item_a, item_c}, value 17."""
    payload = json.loads((EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8"))
    return OptimizationProblem.model_validate(payload)


FEASIBLE = [{"item_a": 1, "item_b": 0, "item_c": 1, "item_d": 0}]
COMPILED = estimate_model_variables(knapsack_problem(), "bqm")


class ExhaustiveDeclaredBackend(ScriptedInterruptBackend):
    """Exhaustive, not interruptible, with a declared compiled-size ceiling."""

    def __init__(self, maximum: int) -> None:
        super().__init__(FEASIBLE, supports_interrupt=False, name=NAME)
        self.capabilities = SolverCapabilities(
            name=NAME,
            remote=False,
            heuristic=False,
            exhaustive=True,
            supports_seed=False,
            supports_num_reads=False,
            supports_time_limit=False,
            supported_model_types=["bqm"],
            returns_multiple_samples=True,
            description="test-only: exhaustive backend with its own size ceiling",
            compiled_variable_limit=CompiledVariableLimit(
                maximum=maximum, error_code=DECLARED_CODE
            ),
        )

    @property
    def is_exhaustive(self) -> bool:
        return True


@pytest.fixture
def estimate_calls(monkeypatch) -> list[tuple[OptimizationProblem, str]]:
    """Every call of the optimizer module's ``estimate_model_variables``."""
    calls: list[tuple[OptimizationProblem, str]] = []
    original = optimizer_module.estimate_model_variables

    def counting(problem, model_type):
        calls.append((problem, model_type))
        return original(problem, model_type)

    monkeypatch.setattr(optimizer_module, "estimate_model_variables", counting)
    return calls


def service_for(
    maximum: int, exact_max_variables: int
) -> tuple[ExhaustiveDeclaredBackend, OptimizationService]:
    backend = ExhaustiveDeclaredBackend(maximum)
    service = OptimizationService(
        registry=SolverRegistry({backend.name: backend}),
        policy=ExecutionPolicy(exact_max_variables=exact_max_variables),
    )
    return backend, service


def test_the_problem_compiles_to_more_than_its_business_variables():
    # Slack bits included, so both ceilings below can sit between the two.
    assert COMPILED > 4


def test_both_ceilings_share_one_estimate(estimate_calls):
    backend, service = service_for(maximum=COMPILED, exact_max_variables=COMPILED)

    result = service.solve(route(knapsack_problem(), NAME))

    assert result.status == "success"
    assert result.optimality_proven is True
    assert len(backend.calls) == 1
    assert [model_type for _, model_type in estimate_calls] == ["bqm"]


def test_the_policy_ceiling_still_refuses(estimate_calls):
    backend, service = service_for(maximum=COMPILED, exact_max_variables=COMPILED - 1)

    result = service.solve(route(knapsack_problem(), NAME))

    assert result.status == "resource_limit_exceeded"
    assert [error.code for error in result.errors] == ["EXACT_VARIABLE_LIMIT"]
    assert backend.calls == []
    assert len(estimate_calls) == 1


def test_the_declared_ceiling_still_refuses(estimate_calls):
    backend, service = service_for(maximum=COMPILED - 1, exact_max_variables=COMPILED)

    result = service.solve(route(knapsack_problem(), NAME))

    assert result.status == "resource_limit_exceeded"
    assert [error.code for error in result.errors] == [DECLARED_CODE]
    assert f"limit of {COMPILED - 1}" in result.errors[0].message
    assert backend.calls == []
    assert len(estimate_calls) == 1

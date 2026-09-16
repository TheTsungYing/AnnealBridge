"""``OptimizationService.validate()`` (Phase 3a spec §10, §26.1).

The service is the only place that wires registry declaration and policy
into the validator, so the contract is: its answer equals a direct
``validate_problem_full`` call with the registry's capabilities and the
policy's variable ceiling; nothing is compiled, solved or gated; and an
unknown backend yields an UNKNOWN_BACKEND *warning*, never an error.
"""

import pytest

from annealbridge.models import OptimizationProblem, SolverPreferences
from annealbridge.orchestration import ExecutionPolicy, OptimizationService
from annealbridge.solvers import SolverRegistry
from annealbridge.validation import validate_problem_full


def make_problem(num_variables: int = 3, **solver) -> OptimizationProblem:
    names = [f"x{index}" for index in range(1, num_variables + 1)]
    return OptimizationProblem.model_validate(
        {
            "name": "service validate",
            "variables": [{"name": name} for name in names],
            "objective": {
                "direction": "minimize",
                "linear_terms": [{"variable": names[0], "coefficient": 1}],
            },
            "constraints": [
                {
                    "id": "cap",
                    "type": "hard",
                    "terms": [{"variable": name, "coefficient": 1} for name in names[:2]],
                    "operator": "<=",
                    "rhs": 1,
                }
            ],
            "solver": solver,
        }
    )


class TestMatchesDirectValidatorCall:
    @pytest.mark.parametrize(
        "solver",
        [
            {"backend": "exact", "seed": 42, "num_reads": 200},
            {"backend": "simulated_annealing", "seed": 42},
            {"backend": "dwave_qpu", "seed": 1},
            {"backend": "leap_hybrid_bqm", "num_sweeps": 5},
        ],
    )
    def test_same_result_as_validate_problem_full(self, solver):
        registry = SolverRegistry.default()
        policy = ExecutionPolicy(exact_max_variables=8)
        service = OptimizationService(registry=registry, policy=policy)
        problem = make_problem(**solver)

        via_service = service.validate(problem)
        caps = registry.get(problem.solver.backend).capabilities
        direct = validate_problem_full(
            problem,
            capabilities=caps,
            max_compiled_variables=int(policy.limit("variables")),
            model_type=caps.preferred_model_type,
        )
        assert via_service == direct

    def test_policy_variable_limit_reaches_the_validator(self):
        # 3 variables + 1 slack bit = 4 compiled; limit 4 -> near limit.
        service = OptimizationService(policy=ExecutionPolicy(exact_max_variables=4))
        result = service.validate(make_problem(backend="exact"))
        assert result.valid is True
        assert result.estimated_compiled_variables == 4
        assert "EXACT_NEAR_LIMIT" in {w.code for w in result.warnings}

    def test_exact_with_seed_reports_seed_ignored(self):
        # Drift 2 (3a §0): exact declares supports_seed=False.
        result = OptimizationService().validate(make_problem(backend="exact", seed=42))
        assert "SEED_IGNORED" in {w.code for w in result.warnings}

    def test_model_type_follows_the_declaration(self):
        result = OptimizationService().validate(make_problem(backend="exact"))
        assert result.model_type == "bqm"

    def test_remote_backend_is_not_gated(self):
        # Remote solving is off by default, yet validate() never gates:
        # the declaration is all it reads.
        result = OptimizationService().validate(make_problem(backend="dwave_qpu", seed=1))
        assert result.valid is True
        assert "SEED_IGNORED" in {w.code for w in result.warnings}
        assert "REMOTE_DISABLED" not in {w.code for w in result.warnings}

    def test_invalid_problem_reports_errors(self):
        problem = make_problem()
        broken = problem.model_copy(
            update={
                "solver": SolverPreferences.model_validate(
                    {"backend": "exact", "top_k": 0}
                )
            }
        )
        result = OptimizationService().validate(broken)
        assert result.valid is False
        assert [e.code for e in result.errors] == ["INVALID_SOLVER_PREFERENCE"]


class TestDoesNotSolve:
    def test_no_backend_is_called_and_no_slot_is_taken(self, monkeypatch):
        registry = SolverRegistry.default()
        calls: list[str] = []
        for name in registry.names():
            backend = registry.get(name)
            monkeypatch.setattr(
                backend,
                "solve",
                lambda *args, _name=name, **kwargs: calls.append(_name),
            )
            monkeypatch.setattr(
                backend,
                "is_available",
                lambda *args, _name=name, **kwargs: calls.append(f"avail:{_name}"),
            )
        service = OptimizationService(
            registry=registry, policy=ExecutionPolicy(max_concurrent_solves=1)
        )

        # Occupy the only slot: validate() must still answer.
        assert service._solve_slots.acquire(blocking=False)
        try:
            result = service.validate(make_problem(backend="exact"))
        finally:
            service._solve_slots.release()

        assert result.valid is True
        assert calls == []


class TestUnknownBackend:
    def test_custom_registry_without_backend_warns(self):
        sa = SolverRegistry.default().get("simulated_annealing")
        service = OptimizationService(registry=SolverRegistry({"simulated_annealing": sa}))

        result = service.validate(make_problem(backend="exact", seed=42))

        assert result.valid is True
        codes = [w.code for w in result.warnings]
        assert codes == ["UNKNOWN_BACKEND"]
        (warning,) = result.warnings
        assert warning.path == "solver.backend"
        assert "exact" in warning.message
        assert "simulated_annealing" in warning.message
        assert warning.recommended_action  # shared catalog entry
        # No declaration -> no backend-fit / ignored-parameter advice, and
        # the estimate falls back to the BQM path.
        assert result.model_type == "bqm"
        assert result.estimated_compiled_variables == 4

    def test_registered_backend_does_not_warn(self):
        result = OptimizationService().validate(make_problem(backend="exact"))
        assert "UNKNOWN_BACKEND" not in {w.code for w in result.warnings}


SA_SEED_MAX = 2**31 - 1


def seed_findings(result) -> list:
    """Every error and warning the result reports at ``solver.seed``."""
    return [
        finding
        for finding in [*result.errors, *result.warnings]
        if finding.path == "solver.seed"
    ]


class TestSeedRange:
    """The registry declaration reaches the validator, so a seed outside
    ``simulated_annealing``'s declared ``seed_min``..``seed_max`` is an error
    from validate() rather than a vendor failure that only solve() finds.
    """

    @pytest.mark.parametrize("seed", [-1, 2**31, 2**40])
    def test_out_of_range_seed_is_an_error(self, seed):
        result = OptimizationService().validate(
            make_problem(backend="simulated_annealing", seed=seed)
        )
        assert result.valid is False
        (error,) = result.errors
        assert error.code == "INVALID_SOLVER_PREFERENCE"
        assert error.path == "solver.seed"
        assert error.message == (
            f"solver.seed must be an integer between 0 and {SA_SEED_MAX} "
            f"on backend simulated_annealing, got {seed}"
        )
        assert result.warnings == []

    @pytest.mark.parametrize("seed", [0, SA_SEED_MAX])
    def test_both_ends_of_the_range_are_valid(self, seed):
        result = OptimizationService().validate(
            make_problem(backend="simulated_annealing", seed=seed)
        )
        assert result.valid is True
        assert result.errors == []
        assert seed_findings(result) == []

    @pytest.mark.parametrize("seed", [-1, 2**40])
    def test_backend_without_seed_support_only_warns_seed_ignored(self, seed):
        result = OptimizationService().validate(make_problem(backend="exact", seed=seed))
        assert result.valid is True
        assert result.errors == []
        assert [(f.code, f.path) for f in seed_findings(result)] == [
            ("SEED_IGNORED", "solver.seed")
        ]

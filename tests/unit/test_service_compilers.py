"""Service compiler selection and the no-hard-penalty path (3a spec §14, §16, §22).

The service owns a list of compilers keyed by ``model_type`` and picks one
per solve from the backend's *declaration* (``supported_model_types``),
never from its name. These tests use fakes for both sides so the contract
can be exercised before any real constraint-model compiler or backend
exists (steps 6–7 of the 3a plan add those):

* construction rejects two compilers for the same model type;
* a backend whose declared model types have no compiler is a
  ``configuration_error`` / ``NO_COMPILER_FOR_MODEL_TYPE`` — reported, never
  silently routed to another compiler (overview principle 5);
* a compiler with ``uses_hard_penalty=False`` walks the whole pipeline
  with ``hard_penalty=None``, exactly one attempt, ``penalty=None`` on the
  attempt record and no ``REMOTE_RETRIES_DISABLED`` warning;
* ``validate()`` answers ``model_type=None`` for such a backend, agreeing
  with what ``solve`` would do;
* the service stamps ``model_type`` into metadata a backend returned, and
  creates none when the backend returned none.
"""

import pytest

from annealbridge.compiler import BQMCompiler, build_objective_bqm
from annealbridge.compiler.base import select_business_columns
from annealbridge.exceptions import CompilationError
from annealbridge.models import (
    CompiledProblem,
    Constraint,
    ConstraintTrace,
    LinearTerm,
    Objective,
    OptimizationProblem,
    SolverExecutionMetadata,
    SolverPreferences,
    Variable,
)
from annealbridge.orchestration import OptimizationService
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.solvers import RawSolverResult, SolverRegistry
from annealbridge.solvers.base import AvailabilityStatus, SolverCapabilities
from tests.fakes.declared_backend import FakeDeclaredBackend

FAKE_CQM_NAME = "fake_cqm_only"


def make_problem(backend: str) -> OptimizationProblem:
    """max 3 x1 + 2 x2 s.t. x1 + x2 <= 1 (optimum x1=1, objective 3)."""
    return OptimizationProblem(
        name="compiler-selection",
        variables=[
            Variable(name="x1", type="binary"),
            Variable(name="x2", type="binary"),
        ],
        objective=Objective(
            direction="maximize",
            linear_terms=[
                LinearTerm(variable="x1", coefficient=3),
                LinearTerm(variable="x2", coefficient=2),
            ],
        ),
        constraints=[
            Constraint(
                id="capacity",
                type="hard",
                terms=[
                    LinearTerm(variable="x1", coefficient=1),
                    LinearTerm(variable="x2", coefficient=1),
                ],
                operator="<=",
                rhs=1,
            )
        ],
        solver=SolverPreferences(backend="simulated_annealing", max_retries=2)
        .model_copy(update={"backend": backend}),
    )


class FakeNativeCompiler:
    """A ``uses_hard_penalty=False`` compiler that hands back a plain BQM.

    It compiles only the objective (constraints are "native", i.e. left to
    the backend), which is enough to prove the service's penalty/retry
    handling; the sample the backend returns is validated against the
    original problem regardless of what the model says.
    """

    model_type = "cqm"
    uses_hard_penalty = False

    def __init__(self) -> None:
        self.received_penalties: list[float | None] = []

    def compile(self, problem, hard_penalty):
        self.received_penalties.append(hard_penalty)
        if hard_penalty is not None:
            raise CompilationError("native compiler must receive hard_penalty=None")
        bqm = build_objective_bqm(problem.objective)
        for variable in problem.variables:
            bqm.add_variable(variable.name)
        return CompiledProblem(
            model_type=self.model_type,
            model=bqm,
            original_problem=problem,
            internal_variables=set(),
            constraint_trace=[
                ConstraintTrace(
                    constraint_id=constraint.id,
                    constraint_type=constraint.type,
                    operator=constraint.operator,
                    source_description=constraint.description,
                    generated_variables=[],
                    penalty=None if constraint.type == "hard" else constraint.weight,
                    slack_range=None,
                    native=True,
                    compiler="FakeNativeCompiler",
                )
                for constraint in problem.constraints
            ],
            hard_penalty=None,
            objective_scale=1.0,
            num_variables=bqm.num_variables,
        )

    def decode(self, compiled, raw):
        return select_business_columns(compiled, raw)


class FakeBQMOnlyCompiler:
    """Second compiler claiming the BQM slot, for the duplicate check."""

    model_type = "bqm"
    uses_hard_penalty = True

    def compile(self, problem, hard_penalty):  # pragma: no cover - never reached
        raise AssertionError("not meant to compile")

    def decode(self, compiled, raw):  # pragma: no cover - never reached
        raise AssertionError("not meant to decode")


def _cqm_only_capabilities(remote: bool) -> SolverCapabilities:
    return SolverCapabilities(
        name=FAKE_CQM_NAME,
        remote=remote,
        heuristic=True,
        exhaustive=False,
        supports_seed=False,
        supports_num_reads=False,
        supports_time_limit=False,
        supported_model_types=["cqm"],
        returns_multiple_samples=True,
        description="Test-only backend that accepts constraint models only.",
    )


class FakeCQMOnlyBackend:
    """Declares ``["cqm"]`` only and returns a fixed assignment.

    ``assignment`` is widened over every compiled variable; ``metadata``
    (if given) is returned as-is so the test can see what the service
    adds to it.
    """

    def __init__(
        self,
        assignment: dict[str, int] | None = None,
        *,
        remote: bool = False,
        metadata: SolverExecutionMetadata | None = None,
    ) -> None:
        self._assignment = dict(assignment or {})
        self._caps = _cqm_only_capabilities(remote)
        self._metadata = metadata
        self.solve_calls = 0
        self.received_hard_penalties: list[float | None] = []

    @property
    def capabilities(self) -> SolverCapabilities:
        return self._caps

    def is_available(self) -> AvailabilityStatus:
        return AvailabilityStatus(category="available")

    @property
    def name(self) -> str:
        return self._caps.name

    @property
    def is_exhaustive(self) -> bool:
        return False

    def resolve_time_limit(self, compiled_problem, preferences) -> float | None:
        return None

    def solve(self, compiled_problem, preferences) -> RawSolverResult:
        self.solve_calls += 1
        self.received_hard_penalties.append(compiled_problem.hard_penalty)
        bqm = compiled_problem.model
        variables = [str(v) for v in bqm.variables]
        row = {v: int(self._assignment.get(v, 0)) for v in variables}
        return RawSolverResult.from_dicts(
            [row],
            [float(bqm.energy(row))],
            backend=self.name,
            variables=variables,
            metadata=self._metadata,
        )


def _service(backend, compilers=None, policy=None) -> OptimizationService:
    kwargs = {"registry": SolverRegistry({backend.name: backend})}
    if compilers is not None:
        kwargs["compilers"] = compilers
    if policy is not None:
        kwargs["policy"] = policy
    return OptimizationService(**kwargs)


class TestConstruction:
    def test_duplicate_model_type_is_rejected(self):
        with pytest.raises(ValueError, match="duplicate compiler.*bqm"):
            OptimizationService(compilers=[BQMCompiler(), FakeBQMOnlyCompiler()])

    def test_default_compiler_list_serves_bqm_backends(self):
        # The shipped backends declare bqm; the default service must still
        # solve them (Phase 1/2 behaviour, bit for bit).
        result = OptimizationService().solve(make_problem("exact"))

        assert result.status == "success"
        assert result.solutions[0].objective_value == 3.0
        assert result.attempts[0].penalty is not None

    def test_empty_compiler_list_is_allowed_but_solves_nothing(self):
        backend = FakeDeclaredBackend()
        service = OptimizationService(
            compilers=[],
            registry=SolverRegistry({backend.name: backend}),
            policy=ExecutionPolicy(allow_remote=True, limits={"iterations": 1000}),
        )

        result = service.solve(make_problem(backend.name))

        assert result.status == "configuration_error"
        assert [e.code for e in result.errors] == ["NO_COMPILER_FOR_MODEL_TYPE"]
        assert backend.solve_calls == 0


class TestNoCompilerForModelType:
    def test_cqm_only_backend_with_bqm_only_service(self):
        backend = FakeCQMOnlyBackend()
        # The default list now ships CQMCompiler too (3a step 6), so the
        # "no compiler" case needs an explicit bqm-only list.
        service = _service(backend, compilers=[BQMCompiler()])

        result = service.solve(make_problem(backend.name))

        assert result.status == "configuration_error"
        assert len(result.errors) == 1
        error = result.errors[0]
        assert error.code == "NO_COMPILER_FOR_MODEL_TYPE"
        assert error.recommended_action
        assert "cqm" in error.message
        assert result.backend == backend.name
        assert result.objective_direction == "maximize"
        assert result.attempts == []
        assert result.solutions == []
        assert result.message == error.message
        # Declared model types are honoured: nothing is compiled or solved
        # through a compiler the backend never asked for.
        assert backend.solve_calls == 0

    def test_validate_reports_model_type_none_without_compiler(self):
        backend = FakeCQMOnlyBackend()
        service = _service(backend, compilers=[BQMCompiler()])

        result = service.validate(make_problem(backend.name))

        assert result.model_type is None
        # Still a full validation of the backend-independent parts, plus a
        # heads-up (warning, like UNKNOWN_BACKEND) that solve would fail.
        assert result.errors == []
        assert "NO_COMPILER_FOR_MODEL_TYPE" in [w.code for w in result.warnings]
        warning = next(w for w in result.warnings if w.code == "NO_COMPILER_FOR_MODEL_TYPE")
        assert warning.path == "solver.backend"
        assert warning.recommended_action

    def test_validate_agrees_with_solve_once_a_compiler_exists(self):
        backend = FakeCQMOnlyBackend()
        service = _service(backend, compilers=[BQMCompiler(), FakeNativeCompiler()])

        assert service.validate(make_problem(backend.name)).model_type == "cqm"

    def test_first_declared_type_with_a_compiler_wins(self):
        # The service follows the backend's own preference order.
        caps = _cqm_only_capabilities(remote=False).model_copy(
            update={"supported_model_types": ["cqm", "bqm"]}
        )
        backend = FakeCQMOnlyBackend({"x1": 1})
        backend._caps = caps
        service = _service(backend, compilers=[BQMCompiler(), FakeNativeCompiler()])

        assert service.validate(make_problem(backend.name)).model_type == "cqm"
        result = service.solve(make_problem(backend.name))
        assert result.status == "success"
        assert result.attempts[0].penalty is None

        bqm_only = _service(backend, compilers=[BQMCompiler()])
        assert bqm_only.validate(make_problem(backend.name)).model_type == "bqm"
        assert bqm_only.solve(make_problem(backend.name)).attempts[0].penalty is not None


class TestNoHardPenaltyPath:
    def test_feasible_answer_walks_the_full_pipeline(self):
        compiler = FakeNativeCompiler()
        backend = FakeCQMOnlyBackend({"x1": 1})
        service = _service(backend, compilers=[compiler])

        result = service.solve(make_problem(backend.name))

        assert result.status == "success"
        assert compiler.received_penalties == [None]
        assert backend.received_hard_penalties == [None]
        assert len(result.attempts) == 1
        attempt = result.attempts[0]
        assert attempt.attempt == 1
        assert attempt.penalty is None
        assert attempt.samples_received == 1
        assert attempt.feasible_samples == 1
        # Objective and feasibility come from the original problem, not the
        # model the fake compiler built.
        solution = result.solutions[0]
        assert solution.variables == {"x1": 1, "x2": 0}
        assert solution.objective_value == 3.0
        assert solution.hard_constraints_satisfied is True
        # A success carries no infeasibility diagnosis, even though the
        # attempt ran the same validation pass.
        assert result.infeasibility is None
        # make_problem asks for retries, which the CQM path cannot honour;
        # solve reports the validator's PARAMETER_IGNORED for it, and nothing
        # else (no REMOTE_RETRIES_DISABLED: nothing was cut).
        assert [w.code for w in result.warnings] == ["PARAMETER_IGNORED"]
        assert result.warnings[0].path == "solver.max_retries"

    def test_infeasible_answer_is_one_attempt_and_not_proven(self):
        # x1 = x2 = 1 violates x1 + x2 <= 1; the independent validator
        # rejects it even though the model itself carries no penalty.
        backend = FakeCQMOnlyBackend({"x1": 1, "x2": 1})
        service = _service(backend, compilers=[FakeNativeCompiler()])
        problem = make_problem(backend.name)
        assert problem.solver.max_retries == 2

        result = service.solve(problem)

        assert result.status == "infeasible"
        assert result.infeasibility_proven is False
        assert len(result.attempts) == 1
        assert result.attempts[0].penalty is None
        assert result.attempts[0].feasible_samples == 0
        assert backend.solve_calls == 1
        assert "constraint-model backend" in result.message
        assert "not proven" in result.message
        assert [w.code for w in result.warnings] == ["PARAMETER_IGNORED"]

    def test_remote_backend_gets_no_retries_disabled_warning(self):
        # A remote no-penalty backend with retries disabled by policy: the
        # single attempt is the model's nature, not a policy cut, so no
        # REMOTE_RETRIES_DISABLED warning (3a §16.2 step 16/17).
        backend = FakeCQMOnlyBackend({"x1": 1, "x2": 1}, remote=True)
        service = _service(
            backend,
            compilers=[FakeNativeCompiler()],
            policy=ExecutionPolicy(allow_remote=True, allow_remote_retries=False),
        )

        result = service.solve(make_problem(backend.name))

        assert result.status == "infeasible"
        assert len(result.attempts) == 1
        # Only the validator's advice about max_retries on the CQM path.
        assert [w.code for w in result.warnings] == ["PARAMETER_IGNORED"]
        assert "constraint-model backend" in result.message

    def test_bqm_path_still_retries(self):
        # Control case: the same backend on the hard-penalty path keeps the
        # Phase 2 retry behaviour (1 + max_retries attempts).
        caps = _cqm_only_capabilities(remote=False).model_copy(
            update={"supported_model_types": ["bqm"]}
        )
        backend = FakeCQMOnlyBackend({"x1": 1, "x2": 1})
        backend._caps = caps
        service = _service(backend, compilers=[BQMCompiler()])

        result = service.solve(make_problem(backend.name))

        assert result.status == "infeasible"
        assert len(result.attempts) == 3
        assert all(a.penalty is not None for a in result.attempts)
        # The diagnosis describes the *last* attempt, the one that ran at
        # the highest penalty: its candidate count is the rates' denominator.
        assert result.infeasibility is not None
        rates = result.infeasibility.hard_violation_rates
        assert [rate.constraint_id for rate in rates] == ["capacity"]
        assert rates[0].candidates == result.attempts[-1].unique_samples
        assert result.infeasibility.closest_candidate.variables == {"x1": 1, "x2": 1}


class TestMetadataModelType:
    def test_service_stamps_model_type_into_returned_metadata(self):
        backend = FakeCQMOnlyBackend(
            {"x1": 1},
            metadata=SolverExecutionMetadata(backend=FAKE_CQM_NAME, remote=False),
        )
        service = _service(backend, compilers=[FakeNativeCompiler()])

        result = service.solve(make_problem(backend.name))

        assert result.status == "success"
        assert result.metadata is not None
        assert result.metadata.model_type == "cqm"
        assert result.metadata.sampler_reported_feasible is None

    def test_bqm_path_metadata_says_bqm(self):
        backend = FakeDeclaredBackend()
        backend_metadata = SolverExecutionMetadata(backend=backend.name, remote=True)
        original_solve = backend.solve

        def solve_with_metadata(compiled_problem, preferences):
            raw = original_solve(compiled_problem, preferences)
            raw.metadata = backend_metadata
            return raw

        backend.solve = solve_with_metadata
        service = OptimizationService(
            registry=SolverRegistry({backend.name: backend}),
            policy=ExecutionPolicy(
                allow_remote=True, limits={"iterations": 1000}
            ),
        )
        problem = make_problem(backend.name).model_copy(
            update={"solver": SolverPreferences(backend="simulated_annealing").model_copy(
                update={"backend": backend.name}
            )}
        )
        backend._assignment = {"x1": 1}

        result = service.solve(problem)

        assert result.status == "success"
        assert result.metadata is not None
        assert result.metadata.model_type == "bqm"
        # The backend's own object is left untouched (the service copies).
        assert backend_metadata.model_type is None

    def test_no_metadata_is_not_invented(self):
        backend = FakeCQMOnlyBackend({"x1": 1})
        service = _service(backend, compilers=[FakeNativeCompiler()])

        result = service.solve(make_problem(backend.name))

        assert result.status == "success"
        assert result.metadata is None


class TestBQMCompilerContract:
    def test_declares_bqm_with_hard_penalty(self):
        compiler = BQMCompiler()

        assert compiler.model_type == "bqm"
        assert compiler.uses_hard_penalty is True

    def test_none_hard_penalty_is_a_caller_error(self):
        with pytest.raises(CompilationError, match="hard_penalty"):
            BQMCompiler().compile(make_problem("exact"), None)

    def test_compiled_problem_records_model_type(self):
        compiled = BQMCompiler().compile(make_problem("exact"), 5.0)

        assert compiled.model_type == "bqm"
        assert compiled.hard_penalty == 5.0
        assert all(trace.native is False for trace in compiled.constraint_trace)
        assert all(trace.penalty is not None for trace in compiled.constraint_trace)

"""End-to-end consistency between validator, compiler and service (spec §14, §27).

Everything here goes through ``OptimizationService.solve`` — the same entry
point the MCP and CLI adapters use — rather than calling ``validate_problem``
or a backend directly, so the tests cover the *contract* callers see:

* a problem the validator rejects gets ``invalid_problem`` from every
  backend, with the user's own operator/rhs echoed back;
* a compiler that still refuses a validated problem is reported as
  ``invalid_problem`` / ``COMPILATION_FAILED``, never as a solver error;
* an exhaustive backend only claims ``infeasibility_proven`` when it
  actually enumerated something.
"""

import pytest

from annealbridge.exceptions import CompilationError
from annealbridge.models import (
    Constraint,
    LinearTerm,
    Objective,
    OptimizationProblem,
    SolverPreferences,
    Variable,
)
from annealbridge.orchestration import ExecutionPolicy, OptimizationService
from annealbridge.solvers import RawSolverResult, SolverRegistry
from annealbridge.solvers.base import AvailabilityStatus, SolverCapabilities
from tests.fakes.declared_backend import (
    FAKE_DECLARED_NAME,
    FAKE_LIMIT_KEY,
    FakeDeclaredBackend,
)

BACKENDS = ["simulated_annealing", "exact"]


def duplicate_term_problem() -> OptimizationProblem:
    """x1 appears twice with +1 and -1: the summed coefficient is 0.

    Per-term ranges would suggest lhs in [-1, 1] (so ">= 1" looks
    reachable); the accumulated range is [0, 0] and the constraint is
    trivially infeasible.
    """
    return OptimizationProblem(
        name="duplicate-term-infeasible",
        variables=[Variable(name="x1", type="binary")],
        objective=Objective(
            direction="maximize",
            linear_terms=[LinearTerm(variable="x1", coefficient=1)],
        ),
        constraints=[
            Constraint(
                id="cancelling",
                type="hard",
                terms=[
                    LinearTerm(variable="x1", coefficient=1),
                    LinearTerm(variable="x1", coefficient=-1),
                ],
                operator=">=",
                rhs=1,
            )
        ],
    )


def feasible_problem(backend: str = "simulated_annealing") -> OptimizationProblem:
    """A small, valid, satisfiable problem."""
    return OptimizationProblem(
        name="valid-problem",
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
        solver=SolverPreferences(backend=backend),
    )


class TestDuplicateTermInequality:
    """The validator judges repeated variables by their summed coefficient."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_rejected_identically_by_every_backend(self, backend):
        problem = duplicate_term_problem().model_copy(
            update={"solver": SolverPreferences(backend=backend)}
        )

        result = OptimizationService().solve(problem)

        assert result.status == "invalid_problem"
        assert [error.code for error in result.errors] == ["TRIVIALLY_INFEASIBLE"]
        # No backend was ever reached, so nothing was attempted.
        assert result.attempts == []

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_message_echoes_the_user_operator_and_rhs(self, backend):
        problem = duplicate_term_problem().model_copy(
            update={"solver": SolverPreferences(backend=backend)}
        )

        result = OptimizationService().solve(problem)

        message = result.errors[0].message
        # The user wrote ">= 1"; the compiler's normalized form ("<= -1")
        # must never surface in a user-facing message.
        assert ">= 1" in message
        assert "-1" not in message

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_error_carries_recommended_action(self, backend):
        problem = duplicate_term_problem().model_copy(
            update={"solver": SolverPreferences(backend=backend)}
        )

        result = OptimizationService().solve(problem)

        assert result.errors[0].recommended_action


class TestNoVariables:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_empty_variable_list_is_invalid_problem(self, backend):
        problem = OptimizationProblem(
            name="no-variables",
            variables=[],
            objective=Objective(direction="minimize", linear_terms=[]),
            constraints=[],
            solver=SolverPreferences(backend=backend),
        )

        result = OptimizationService().solve(problem)

        assert result.status == "invalid_problem"
        assert [error.code for error in result.errors] == ["NO_VARIABLES"]
        assert result.errors[0].recommended_action
        assert result.attempts == []

    def test_message_mirrors_the_first_error(self):
        # invalid_problem is built like every other failure: message is the
        # first error's text, never None.
        problem = OptimizationProblem(
            name="no-variables",
            variables=[],
            objective=Objective(direction="minimize", linear_terms=[]),
            constraints=[],
            solver=SolverPreferences(backend="exact"),
        )

        result = OptimizationService().solve(problem)

        assert result.message is not None
        assert result.message == result.errors[0].message
        assert result.backend is None
        assert result.objective_direction is None


class FakeFailingCompiler:
    """A compiler that refuses everything, to exercise the §27 fallback.

    Declares the BQM slot (3a §16.1) so the service selects it for the
    shipped backends and walks the hard-penalty path up to ``compile``.
    """

    @property
    def model_type(self):
        return "bqm"

    @property
    def uses_hard_penalty(self):
        return True

    def compile(self, problem, hard_penalty):
        raise CompilationError("boom")

    def decode(self, compiled, raw):  # pragma: no cover - never reached
        raise AssertionError("decode must not be reached: compile always fails")


def declared_service(backend) -> OptimizationService:
    """A service whose only backend is the fifth backend of 3a §13.1."""
    return OptimizationService(
        registry=SolverRegistry({FAKE_DECLARED_NAME: backend}),
        policy=ExecutionPolicy(allow_remote=True, limits={FAKE_LIMIT_KEY: 1000}),
    )


def problem_on_declared_backend() -> OptimizationProblem:
    """``feasible_problem`` routed at the fifth backend.

    Its ``<=`` constraint is what takes compilation through
    ``encode_slack``. ``SolverPreferences.backend`` is a Literal of the
    shipped names, so a test-only backend is set with ``model_construct``.
    """
    solver = SolverPreferences.model_construct(backend=FAKE_DECLARED_NAME)
    return feasible_problem().model_copy(update={"solver": solver})


class TestCompilationErrorFallback:
    def test_bare_exception_inside_compile_is_a_compilation_failure(self, monkeypatch):
        """2026-09-09 review (addition 3): anything a compiler raises before
        a backend is touched is a problem verdict, not a solver error.

        A bare ``ValueError`` escaping ``encode_slack`` used to surface as
        ``solver_error`` / ``SOLVER_ERROR`` even though no backend had run.
        """

        def raiser(*args, **kwargs):
            raise ValueError("bare slack failure")

        monkeypatch.setattr("annealbridge.compiler.bqm.encode_slack", raiser)
        backend = FakeDeclaredBackend()

        result = declared_service(backend).solve(problem_on_declared_backend())

        assert result.status == "invalid_problem"
        assert [error.code for error in result.errors] == ["COMPILATION_FAILED"]
        message = result.errors[0].message
        assert "ValueError" in message
        assert "bare slack failure" in message
        assert backend.solve_calls == 0
        assert result.solutions == []
        assert result.backend == FAKE_DECLARED_NAME

    def test_compilation_error_is_reported_as_invalid_problem(self):
        service = OptimizationService(compilers=[FakeFailingCompiler()])

        result = service.solve(feasible_problem())

        assert result.status == "invalid_problem"
        assert len(result.errors) == 1
        error = result.errors[0]
        assert error.code == "COMPILATION_FAILED"
        assert error.message == "boom"
        assert error.recommended_action
        assert error.retryable is False
        # The backend was resolved but never invoked; it is still reported.
        assert result.backend == "simulated_annealing"
        assert result.solutions == []


_FAKE_EXHAUSTIVE_CAPABILITIES = SolverCapabilities(
    name="fake_exhaustive",
    remote=False,
    heuristic=False,
    exhaustive=True,
    supports_seed=False,
    supports_num_reads=False,
    supports_time_limit=False,
    supported_model_types=["bqm"],
    returns_multiple_samples=True,
    description="Exhaustive backend that returns nothing, for defensive tests.",
)


class EmptyExhaustiveBackend:
    """Mirrors ``ExactSolverBackend`` but always returns zero samples."""

    @property
    def capabilities(self) -> SolverCapabilities:
        return _FAKE_EXHAUSTIVE_CAPABILITIES

    def is_available(self) -> AvailabilityStatus:
        return AvailabilityStatus(category="available")

    @property
    def name(self) -> str:
        return self.capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        return self.capabilities.exhaustive

    def resolve_time_limit(self, compiled_problem, preferences) -> float | None:
        return None

    def solve(self, compiled_problem, preferences) -> RawSolverResult:
        return RawSolverResult(
            variables=[str(v) for v in compiled_problem.model.variables],
            samples=[],
            energies=[],
            backend=self.name,
        )


def _with_fake_backend(problem: OptimizationProblem) -> OptimizationProblem:
    """Point ``solver.backend`` at the fake backend.

    ``SolverPreferences.backend`` is a ``Literal`` over the shipped backend
    names, so "fake_exhaustive" cannot be constructed normally; ``model_copy``
    assigns without re-validating, which is exactly what a test-only backend
    needs.
    """
    preferences = problem.solver.model_copy(update={"backend": "fake_exhaustive"})
    return problem.model_copy(update={"solver": preferences})


class TestExhaustiveZeroSamples:
    def test_zero_samples_does_not_prove_infeasibility(self):
        backend = EmptyExhaustiveBackend()
        service = OptimizationService(
            registry=SolverRegistry({backend.name: backend})
        )

        result = service.solve(_with_fake_backend(feasible_problem()))

        assert result.status == "infeasible"
        assert result.infeasibility_proven is False
        assert "returned no samples" in result.message
        assert len(result.attempts) == 1
        assert result.attempts[0].samples_received == 0

    def test_real_exact_backend_still_proves_infeasibility(self):
        # Neither constraint is trivially infeasible on its own (each has a
        # reachable rhs), so the validator passes and only enumeration can
        # show the pair is unsatisfiable.
        problem = OptimizationProblem(
            name="jointly-infeasible",
            variables=[
                Variable(name="x1", type="binary"),
                Variable(name="x2", type="binary"),
            ],
            objective=Objective(
                direction="maximize",
                linear_terms=[
                    LinearTerm(variable="x1", coefficient=1),
                    LinearTerm(variable="x2", coefficient=1),
                ],
            ),
            constraints=[
                Constraint(
                    id="at-least-two",
                    type="hard",
                    terms=[
                        LinearTerm(variable="x1", coefficient=1),
                        LinearTerm(variable="x2", coefficient=1),
                    ],
                    operator=">=",
                    rhs=2,
                ),
                Constraint(
                    id="at-most-one",
                    type="hard",
                    terms=[
                        LinearTerm(variable="x1", coefficient=1),
                        LinearTerm(variable="x2", coefficient=1),
                    ],
                    operator="<=",
                    rhs=1,
                ),
            ],
            solver=SolverPreferences(backend="exact"),
        )

        result = OptimizationService().solve(problem)

        assert result.status == "infeasible"
        assert result.infeasibility_proven is True
        assert len(result.attempts) == 1
        assert result.attempts[0].samples_received > 0

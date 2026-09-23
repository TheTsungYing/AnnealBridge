"""Compiler interface (spec §13; 3a spec §14; 3b spec §13)."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from annealbridge.models import CompiledProblem, ModelType, OptimizationProblem

if TYPE_CHECKING:
    # Type-only: ``solvers`` sits above ``compiler`` in the §4 dependency
    # direction, so the runtime never imports it from here.
    from annealbridge.solvers.base import RawSolverResult


class ModelCompiler(Protocol):
    """Compiles an :class:`OptimizationProblem` into a solver-specific model.

    ``model_type`` is what the service matches against a backend's
    ``supported_model_types``; ``uses_hard_penalty`` tells the service
    whether to compute a hard-constraint penalty (and retry with a larger
    one) or to pass ``None`` because the model expresses hard constraints
    natively. ``decode`` is the inverse direction: it turns the backend's
    model-variable samples back into business variables.
    """

    @property
    def model_type(self) -> ModelType:
        """The model type this compiler produces (``"bqm"``, ``"cqm"``, ...)."""
        ...

    @property
    def uses_hard_penalty(self) -> bool:
        """True when ``compile`` needs a hard penalty (penalty-method models)."""
        ...

    def compile(
        self,
        problem: OptimizationProblem,
        hard_penalty: float | None,
    ) -> CompiledProblem:
        """Compile ``problem``; ``hard_penalty`` is the lambda for hard constraints.

        Compilers with ``uses_hard_penalty`` True require a float and treat
        ``None`` as a caller error; the others require ``None``.
        """
        ...

    def decode(
        self,
        compiled: CompiledProblem,
        raw: "RawSolverResult",
    ) -> "RawSolverResult":
        """Turn the backend's model-variable matrix into business variables.

        Drops the internal columns (slack, encoding bits) and combines
        encoding bits into integer values; ``energies`` and the row order
        are unchanged and ``metadata`` is carried over as is. The returned
        ``variables`` follow ``compiled.original_problem.variables`` on
        every path (a backend's own order is not reused: ``ExactCQMSolver``
        puts binary variables before integer ones, for instance). dtype:
        a problem without integer variables keeps the input dtype (the BQM
        backends' ``int8`` bit path is untouched); with integer variables
        the output is ``int64``. Every value of the result is an integer
        within the business variable's bounds.
        """
        ...


class PreparedModel(Protocol):
    """The hard-penalty-independent part of one problem's compile.

    ``compile(hard_penalty)`` must return exactly what the compiler's own
    ``compile(problem, hard_penalty)`` returns for the problem it was
    prepared from, bit for bit, however many times and in whatever order it
    is called: the prepared state is read, never changed.
    """

    def compile(self, hard_penalty: float | None) -> CompiledProblem:
        """Finish the compile with ``hard_penalty`` as the hard-constraint lambda."""
        ...


@runtime_checkable
class SupportsPrepare(Protocol):
    """Optional compiler capability: compile in two stages.

    ``prepare`` does every step that does not depend on the hard penalty
    (integer and slack encodings, the objective, the soft penalties) once;
    the returned :class:`PreparedModel` finishes a compile per penalty. A
    compiler without it is simply compiled from scratch each time, with the
    same result. The compiler knows nothing about why a caller compiles one
    problem more than once.
    """

    def prepare(self, problem: OptimizationProblem) -> PreparedModel:
        """Do the hard-penalty-independent part of compiling ``problem``."""
        ...


def select_business_columns(
    compiled: CompiledProblem, raw: "RawSolverResult"
) -> "RawSolverResult":
    """The bit-for-bit part of ``decode`` shared by the compilers.

    Keeps the columns of ``raw`` that are business variables, in
    ``compiled.original_problem.variables`` order, preserving dtype, row
    order, energies and metadata. A backend result that lacks a business
    variable violates the backend contract and raises ``ValueError``.
    """
    names = [variable.name for variable in compiled.original_problem.variables]
    column = {name: index for index, name in enumerate(raw.variables)}
    try:
        columns = [column[name] for name in names]
    except KeyError as exc:
        raise ValueError(
            f"solver result from backend '{raw.backend}' lacks business "
            f"variable {exc.args[0]!r}"
        ) from None
    samples = np.ascontiguousarray(raw.samples[:, columns])
    return raw.model_copy(update={"variables": names, "samples": samples})

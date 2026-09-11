"""Domain exception hierarchy.

Solver/backend libraries' exceptions must never leak out of the domain
layer; they are wrapped in these types instead.
"""

from typing import Literal, get_args

# The ``SolveResult.status`` values a backend may name on a
# SolverExecutionError (3b spec §20.8). A subset of the result vocabulary:
# ``solver_error`` is the default the service applies when nothing is
# named, ``configuration_error`` the one value a backend is authorised to
# choose. Kept here rather than imported from ``models`` so this module
# stays a leaf; the subset relation is pinned by a test.
SolverErrorStatus = Literal["solver_error", "configuration_error"]


class OptimizerError(Exception):
    """Base class for all optimizer domain errors."""


class ProblemValidationError(OptimizerError):
    """The optimization problem failed validation before compilation."""


class CompilationError(OptimizerError):
    """The problem could not be compiled into a solver model."""


class NonFiniteModelError(CompilationError):
    """The compiled model holds a non-finite bias (2026-09-09 review F-07).

    Raised by a compiler whose arithmetic overflowed — the hard penalty
    multiplied into a squared-constraint expansion on the BQM path, or a soft
    weight times a squared coefficient on the CQM path (2026-09-11 review
    F06) — so the service
    can report a structured result instead of handing a model with ``inf``
    biases to a backend, which would surface as an unclassified solver
    failure.
    """


class SolverExecutionError(OptimizerError):
    """The solver backend failed while executing.

    ``code`` optionally carries a catalog error code (e.g.
    ``REMOTE_AUTH_FAILED``) so the service layer can build a structured
    ``SolveError`` without re-classifying the failure.

    ``status`` optionally names the ``SolveResult.status`` the failure
    should be reported under (3b spec §20.8). ``None`` means the default
    ``"solver_error"``; a backend sets ``"configuration_error"`` when the
    remote side rejected *our* request shape (e.g. a malformed header),
    because telling the agent to change its problem would be misleading.

    A value outside :data:`SolverErrorStatus` is refused here, at
    construction (2026-09-09 review F-03): backends are third-party code
    the type checker never sees, and a bad value would otherwise surface
    only as a pydantic error thrown from inside the service's ``except``
    handler, where nothing can turn it into a structured result.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: SolverErrorStatus | None = None,
    ) -> None:
        if status is not None and status not in get_args(SolverErrorStatus):
            raise ValueError(
                f"SolverExecutionError.status must be None or one of "
                f"{get_args(SolverErrorStatus)}, got {status!r}"
            )
        super().__init__(message)
        self.code = code
        self.status = status

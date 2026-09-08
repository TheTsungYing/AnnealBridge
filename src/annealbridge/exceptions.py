"""Domain exception hierarchy.

Solver/backend libraries' exceptions must never leak out of the domain
layer; they are wrapped in these types instead.
"""


class OptimizerError(Exception):
    """Base class for all optimizer domain errors."""


class ProblemValidationError(OptimizerError):
    """The optimization problem failed validation before compilation."""


class CompilationError(OptimizerError):
    """The problem could not be compiled into a solver model."""


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
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status

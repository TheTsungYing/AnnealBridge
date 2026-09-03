"""Compiler interface (spec §13; 3a spec §14)."""

from typing import Protocol

from annealbridge.models import CompiledProblem, ModelType, OptimizationProblem


class ModelCompiler(Protocol):
    """Compiles an :class:`OptimizationProblem` into a solver-specific model.

    ``model_type`` is what the service matches against a backend's
    ``supported_model_types``; ``uses_hard_penalty`` tells the service
    whether to compute a hard-constraint penalty (and retry with a larger
    one) or to pass ``None`` because the model expresses hard constraints
    natively.
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

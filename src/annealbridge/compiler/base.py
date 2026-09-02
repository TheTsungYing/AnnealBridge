"""Compiler interface (spec §13)."""

from typing import Protocol

from annealbridge.models import CompiledProblem, OptimizationProblem


class ModelCompiler(Protocol):
    """Compiles an :class:`OptimizationProblem` into a solver-specific model."""

    def compile(
        self,
        problem: OptimizationProblem,
        hard_penalty: float,
    ) -> CompiledProblem:
        """Compile ``problem`` using ``hard_penalty`` for hard constraints."""
        ...

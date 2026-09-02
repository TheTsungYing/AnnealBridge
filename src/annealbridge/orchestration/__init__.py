"""Orchestration service (spec §28)."""

from annealbridge.orchestration.optimizer import (
    OptimizationService,
    deduplicate_samples,
    evaluate_objective,
    process_candidates,
)
from annealbridge.orchestration.policy import ExecutionPolicy

__all__ = [
    "ExecutionPolicy",
    "OptimizationService",
    "deduplicate_samples",
    "evaluate_objective",
    "process_candidates",
]

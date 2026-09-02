"""Orchestration service (spec §28)."""

from annealbridge.orchestration.optimizer import (
    CandidateSet,
    OptimizationService,
    deduplicate_samples,
    evaluate_objective,
    evaluate_objective_batch,
    process_candidates,
)
from annealbridge.orchestration.policy import ExecutionPolicy

__all__ = [
    "CandidateSet",
    "ExecutionPolicy",
    "OptimizationService",
    "deduplicate_samples",
    "evaluate_objective",
    "evaluate_objective_batch",
    "process_candidates",
]

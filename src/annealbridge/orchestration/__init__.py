"""Orchestration service (spec §28)."""

from annealbridge.orchestration.optimizer import (
    CandidateSet,
    OptimizationService,
    ProcessedCandidates,
    deduplicate_samples,
    evaluate_objective,
    evaluate_objective_batch,
    process_candidates,
)
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.orchestration.routing import REASON_DESCRIPTIONS, recommend

__all__ = [
    "CandidateSet",
    "ExecutionPolicy",
    "OptimizationService",
    "ProcessedCandidates",
    "REASON_DESCRIPTIONS",
    "recommend",
    "deduplicate_samples",
    "evaluate_objective",
    "evaluate_objective_batch",
    "process_candidates",
]

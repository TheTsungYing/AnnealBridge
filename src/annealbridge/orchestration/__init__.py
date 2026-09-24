"""Orchestration service (spec §28)."""

from annealbridge.exceptions import SolveCancelled
from annealbridge.interrupt import CancelToken
from annealbridge.orchestration.candidates import (
    CandidateSet,
    ProcessedCandidates,
    deduplicate_samples,
    diagnose_infeasibility,
    evaluate_objective,
    evaluate_objective_batch,
    process_candidates,
)
from annealbridge.orchestration.optimizer import OptimizationService
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.orchestration.progress import ProgressCallback, SolveProgress
from annealbridge.orchestration.routing import REASON_DESCRIPTIONS, recommend

__all__ = [
    "CancelToken",
    "CandidateSet",
    "ExecutionPolicy",
    "OptimizationService",
    "ProcessedCandidates",
    "ProgressCallback",
    "REASON_DESCRIPTIONS",
    "SolveCancelled",
    "SolveProgress",
    "recommend",
    "deduplicate_samples",
    "diagnose_infeasibility",
    "evaluate_objective",
    "evaluate_objective_batch",
    "process_candidates",
]

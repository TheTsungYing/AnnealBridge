"""Solve result message wording (split out of ``optimizer.py``)."""

from typing import Literal

from annealbridge.models import (
    Solution,
    SolveAttempt,
    SolveError,
    catalog_error,
)

_BudgetCut = Literal["exhaustive", "no_hard_penalty", "remote_retries_disabled"]


def _infeasible_message(
    proven: bool, cut_reason: _BudgetCut | None, attempts_made: int
) -> tuple[str, list[SolveError]]:
    """The infeasible result's message and warnings (3a §16.2 steps 16–17).

    Pure so each wording can be pinned by a test without a backend, a
    compiler and a policy behind it.
    """
    warnings: list[SolveError] = []
    if proven:
        message = (
            "No feasible solution exists: the exhaustive backend "
            "enumerated every assignment"
        )
    elif cut_reason == "exhaustive":
        message = (
            "No feasible solution found: the exhaustive backend "
            "returned no samples, so infeasibility is not proven"
        )
    elif cut_reason == "no_hard_penalty":
        # §16.2 step 16 / §16.3: one attempt, nothing to retune.
        message = (
            "No feasible solution found: the constraint-model backend "
            "returned no sample satisfying the hard constraints under "
            "independent validation; infeasibility is not proven"
        )
    elif cut_reason == "remote_retries_disabled":
        message = (
            "No feasible solution found in 1 attempt; server policy "
            "disables automatic remote retries"
        )
        warnings.append(
            catalog_error(
                "REMOTE_RETRIES_DISABLED",
                "1 attempt was made; server policy disables "
                "automatic remote retries to protect quota",
            )
        )
    else:
        message = (
            f"No feasible solution found in {attempts_made} attempt(s); "
            "the problem may still be feasible under a different "
            "solver configuration"
        )
    return message, warnings


def _exact_number(value: float) -> str:
    """Render integral floats without a trailing ``.0`` (``17``, not ``17.0``).

    Non-integral values keep their full ``repr``: the message is meant to
    be quoted, so it must not say ``1.23457e+06`` for an objective the
    result records as ``1234567.5``. Deliberately not the CLI's
    ``_format_number``, which rounds with ``:g`` for a table.
    """
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def _success_message(
    backend: str,
    direction: str,
    optimality_proven: bool,
    attempt: SolveAttempt,
    solutions: list[Solution],
) -> str:
    """The success result's one-line summary, for an agent to relay as is.

    Built only from facts the result already carries — the backend, the
    proof flag, the rank-1 objective and soft violation, and the attempt's
    candidate counts — so it is deterministic: no timings, no random
    values, and (like every recommended_action) no configuration value or
    limit. Pure so each wording can be pinned by a test without a backend.

    ``solutions`` must be non-empty with rank 1 first, as ``process_candidates``
    returns it. The soft violation is quoted exactly as ``soft_violation_score``
    records it, tolerance-free like the field itself. Only the unproven
    wording can name a later attempt: an exhaustive backend, the only kind
    that proves optimality, is given a single attempt (``_max_attempts``).
    """
    best = solutions[0]
    objective = f"rank 1 has objective {_exact_number(best.objective_value)}"
    if best.soft_violation_score > 0:
        objective += (
            f" with soft violation {_exact_number(best.soft_violation_score)}"
        )
    objective += f" ({direction})"
    if optimality_proven:
        return (
            f"{backend} proved optimality: {objective}; "
            f"{attempt.feasible_samples} of {attempt.unique_samples} distinct "
            f"candidates were feasible, {len(solutions)} returned."
        )
    on_attempt = f" on attempt {attempt.attempt}" if attempt.attempt > 1 else ""
    return (
        f"{backend} found {attempt.unique_samples} distinct candidates "
        f"({attempt.feasible_samples} feasible), {len(solutions)} "
        f"returned{on_attempt}: {objective}; optimality is not proven."
    )

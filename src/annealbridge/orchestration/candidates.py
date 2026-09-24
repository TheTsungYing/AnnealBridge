"""Candidate evaluation, dedup and ranking (split out of ``optimizer.py``)."""

import time
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from annealbridge.models import (
    ClosestCandidate,
    HardViolationRate,
    InfeasibilityDiagnostics,
    Objective,
    OptimizationProblem,
    PostprocessStats,
    Solution,
)
from annealbridge.orchestration.postprocess import (
    PostprocessRequest,
    PostprocessRun,
    run_postprocess,
)
from annealbridge.solvers import RawSolverResult
from annealbridge.validation import (
    BatchValidation,
    validate_batch,
    validate_solution,
)


def evaluate_objective(objective: Objective, sample: dict[str, int]) -> float:
    """Compute the business objective value of ``sample`` (spec §8.1, §24).

    Always uses the original coefficients and constant; the value is the
    same regardless of direction (direction only says how to interpret it).
    """
    value = objective.constant
    for term in objective.linear_terms:
        value += term.coefficient * sample[term.variable]
    for term in objective.quadratic_terms:
        value += term.coefficient * sample[term.variable1] * sample[term.variable2]
    return value


def evaluate_objective_batch(
    objective: Objective, variables: list[str], samples: np.ndarray
) -> np.ndarray:
    """Vectorised :func:`evaluate_objective` over the rows of ``samples``.

    Column ``j`` of ``samples`` is the integer value of ``variables[j]``. Terms
    are accumulated in the same order and association as the scalar
    version (constant first, then linear, then quadratic), so the two agree
    bit for bit; the unit tests assert that equality.
    """
    column = {name: index for index, name in enumerate(variables)}
    value = np.full(samples.shape[0], objective.constant, dtype=np.float64)
    for term in objective.linear_terms:
        value += term.coefficient * samples[:, column[term.variable]]
    for term in objective.quadratic_terms:
        value += (
            term.coefficient
            * samples[:, column[term.variable1]]
            * samples[:, column[term.variable2]]
        )
    return value


def _pack_rows(matrix: np.ndarray) -> np.ndarray:
    """Pack each 0/1 row of ``matrix`` into big-endian 64-bit words.

    Returns an ``uint64`` array of shape ``(rows, ceil(columns / 64))``.
    Column 0 of the input lands in the most significant bit of word 0, so
    comparing rows word by word (word 0 first) orders them exactly like
    comparing the original rows as tuples. Any number of columns is
    supported — wider rows simply produce more words — so there is no
    fallback path to keep correct separately.
    """
    rows, columns = matrix.shape
    words = max(1, -(-columns // 64))
    packed = np.packbits(matrix, axis=1, bitorder="big")  # (rows, ceil(cols/8))
    padded = np.zeros((rows, words * 8), dtype=np.uint8)
    padded[:, : packed.shape[1]] = packed
    return padded.view(">u8").astype(np.uint64)


def _pack_integer_rows(matrix: np.ndarray) -> np.ndarray:
    """Pack each integer row of ``matrix`` into big-endian 64-bit words.

    The integer counterpart of :func:`_pack_rows`, with the same contract:
    an ``uint64`` array of shape ``(rows, words)`` whose word-by-word
    comparison (word 0 first) orders rows exactly like comparing them as
    tuples. Each column gets a field wide enough for every value it holds
    (the bit length of ``max - min``, at least one bit); fields are laid
    out column by column from the most significant end of word 0, and a
    field that does not fit in the current word starts the next one, so
    no field straddles a word. A column is stored as ``value - min``: a
    monotone shift to a non-negative range, computed in wrapping ``int64``
    arithmetic and reinterpreted as ``uint64``, which is exact for any
    ``int64`` range including the full one (that field is then 64 bits
    wide and owns its word).

    Why the order is the tuple order: within one word the earlier column
    sits in the higher bits and its field holds every value of that
    column, so unsigned comparison of the word is decided by the first
    differing column; across words the caller compares word 0 first. A
    matrix with no rows or no columns packs to one all-zero word per row.
    """
    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        return np.zeros((rows, 1), dtype=np.uint64)
    values = matrix.astype(np.int64, copy=False)
    lows = values.min(axis=0)
    highs = values.max(axis=0)
    # Python ints: ``high - low`` may exceed int64, and bit_length is exact.
    widths = [
        max(1, (high - low).bit_length())
        for low, high in zip(lows.tolist(), highs.tolist())
    ]
    layout: list[tuple[int, int]] = []  # (word, shift) of each column's field
    word, used = 0, 0
    for width in widths:
        if used + width > 64:
            word, used = word + 1, 0
        used += width
        layout.append((word, 64 - used))
    packed = np.zeros((rows, word + 1), dtype=np.uint64)
    for column, ((word, shift), low) in enumerate(zip(layout, lows)):
        field = (values[:, column] - low).view(np.uint64)
        packed[:, word] |= field << np.uint64(shift)
    return packed


def _lexsort(keys: list[np.ndarray]) -> np.ndarray:
    """Stable multi-key argsort with ``keys`` given in *priority* order.

    ``np.lexsort`` treats its last key as the most significant, which is
    easy to get backwards; this wrapper takes the natural order instead.
    """
    return np.lexsort(tuple(reversed(keys)))


def _words(packed: np.ndarray) -> list[np.ndarray]:
    """Columns of a :func:`_pack_rows` result, most significant word first."""
    return [packed[:, word] for word in range(packed.shape[1])]


def _row_keys(matrix: np.ndarray) -> list[np.ndarray]:
    """Sort keys (priority order) that compare rows of ``matrix`` as tuples.

    A 0/1 ``int8`` matrix -- the BQM backends' bit path -- packs into the
    :func:`_pack_rows` words, exactly as before 3b. Anything else (integer
    values from a CQM backend or a decoded integer problem) packs into the
    :func:`_pack_integer_rows` words, one field per column sized to that
    column's range, so a few words stand in for what used to be one
    ``int64`` key per column. Both orderings equal the lexicographic order
    of the rows, so deduplication and tie-breaking behave identically on
    either path (3b spec §11). The 0/1 check is a min/max scan, not
    ``np.isin`` (two orders of magnitude slower on 16M rows), and it is
    mandatory: ``np.packbits`` would silently treat 2 or -1 as 1.
    """
    if matrix.dtype == np.int8 and (
        matrix.size == 0 or (matrix.min() >= 0 and matrix.max() <= 1)
    ):
        return _words(_pack_rows(matrix))
    return _words(_pack_integer_rows(matrix))


@dataclass(frozen=True)
class CandidateSet:
    """Deduplicated business candidates, aligned row by row.

    ``samples`` is an integer matrix with one row per distinct business
    assignment (column ``j`` is ``variables[j]``, in the input's variable
    order minus the internal ones); ``energies`` is the minimum energy the
    solver reported for that assignment — kept for reporting and debugging
    only, never used for feasibility or ranking (overview principle 2).
    ``counts`` is how many rows of the solver's raw output collapsed into
    each deduplicated assignment, so it sums to ``raw.num_samples``.
    """

    variables: list[str]
    samples: np.ndarray  # integer matrix, shape (candidates, len(variables))
    energies: np.ndarray  # float64, shape (candidates,)
    counts: np.ndarray  # int64, shape (candidates,)

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def sample_dict(self, index: int) -> dict[str, int]:
        return dict(zip(self.variables, self.samples[index].tolist()))

    def as_pairs(self) -> list[tuple[dict[str, int], float]]:
        """``[(business_sample, energy), ...]``.

        Test-facing helper; no production caller. It materialises one dict
        per candidate, so it is only for small-scale use.
        """
        return [
            (self.sample_dict(index), float(self.energies[index]))
            for index in range(len(self))
        ]


def deduplicate_samples(
    raw: RawSolverResult, internal_variables: Collection[str] = frozenset()
) -> CandidateSet:
    """Strip internal variables and deduplicate by business assignment.

    Duplicates keep the minimum energy seen for that assignment (spec §25);
    among equal energies the earliest read wins, and candidates come back
    in order of first appearance, so the result is fully deterministic and
    identical to a row-by-row pass — it is just computed on the arrays.
    Energy is used here only to pick which duplicate's energy to report.
    How many raw rows each survivor stands for is kept in ``counts``.

    Two array paths compute that one rule. When the assignments fit a
    single packed word (up to 64 binary variables -- every exact-backend
    enumeration) the rows are stably sorted on that word alone, so each
    group is contiguous and in read order; the group's minimum energy and
    the earliest row carrying it are then found with ``reduceat`` and one
    equality mask. Otherwise (more words, or integer columns) the rows are
    sorted on all keys with energy last, which puts each group's winner
    first. The single-key path steps aside for NaN energies, which an
    equality mask cannot locate; the multi-key sort orders them last as
    before. The two paths return the same arrays for the same input.

    ``internal_variables`` is optional since 3b: the service hands over a
    result the compiler has already decoded, so nothing is left to strip.
    """
    business_columns = [
        index
        for index, name in enumerate(raw.variables)
        if name not in internal_variables
    ]
    variables = [raw.variables[index] for index in business_columns]
    business = raw.samples[:, business_columns]
    count = business.shape[0]
    if count == 0:
        return CandidateSet(
            variables=variables,
            samples=business,
            energies=np.asarray(raw.energies, dtype=np.float64),
            counts=np.zeros(0, dtype=np.int64),
        )

    keys = _row_keys(business)
    energies = np.asarray(raw.energies, dtype=np.float64)
    group_start = np.zeros(count, dtype=bool)
    group_start[0] = True
    if len(keys) == 1 and not np.isnan(energies).any():
        # One key: a stable sort on it alone groups equal assignments and
        # keeps each group in read order, so the group's first row is its
        # earliest read and the first row matching the group's minimum
        # energy is the earliest such read.
        order = np.argsort(keys[0], kind="stable")
        sorted_key = keys[0][order]
        group_start[1:] = sorted_key[1:] != sorted_key[:-1]
        starts = np.flatnonzero(group_start)
        first_seen = order[starts]  # earliest read of each assignment
        sorted_energy = energies[order]
        group_of = np.cumsum(group_start) - 1
        minimum = np.minimum.reduceat(sorted_energy, starts)
        hits = np.flatnonzero(sorted_energy == minimum[group_of])
        # ``hits`` is ascending, so a group's first hit is where its id
        # first appears; every group has one (its own minimum).
        hit_group = group_of[hits]
        first_hit = np.ones(len(hits), dtype=bool)
        first_hit[1:] = hit_group[1:] != hit_group[:-1]
        representatives = order[hits[first_hit]]  # min-energy read of each
    else:
        # Sort by assignment, then energy; the sort is stable, so within
        # one assignment equal energies stay in read order.
        order = _lexsort([*keys, energies])
        # A new group starts wherever any key differs from the previous row.
        for key in keys:
            sorted_key = key[order]
            group_start[1:] |= sorted_key[1:] != sorted_key[:-1]
        starts = np.flatnonzero(group_start)
        representatives = order[starts]  # min-energy read of each assignment
        first_seen = np.minimum.reduceat(order, starts)  # earliest read of each
    # Group ``g`` spans ``starts[g]`` up to the next start (or the end), so
    # its size is how many raw rows carried that assignment.
    group_sizes = np.diff(np.append(starts, count))
    # One permutation for all three arrays, or ``counts`` would describe a
    # different candidate than the row next to it. ``first_seen`` holds
    # distinct row indices, so sorting it is a counting pass: mark the
    # indices, then read back each group's position in ascending order.
    position = np.empty(count, dtype=np.intp)
    position[first_seen] = np.arange(len(starts))
    seen = np.zeros(count, dtype=bool)
    seen[first_seen] = True
    permutation = position[np.flatnonzero(seen)]
    keep = representatives[permutation]
    return CandidateSet(
        variables=variables,
        samples=np.ascontiguousarray(business[keep]),
        energies=energies[keep],
        counts=group_sizes[permutation].astype(np.int64, copy=False),
    )


@dataclass(frozen=True)
class ProcessedCandidates:
    """What :func:`process_candidates` found in one attempt's raw output.

    ``unique_samples`` and ``feasible_samples`` are the candidate counts the
    attempt reports; both are counted over deduplicated assignments, not
    over the raw rows. ``infeasibility`` is set only when there were
    candidates and none was feasible. ``postprocess`` is the attempt's
    post-processing statistics, ``None`` when post-processing was not
    requested; the two counts above stay solver-only either way.
    """

    solutions: list[Solution]
    unique_samples: int
    feasible_samples: int
    infeasibility: InfeasibilityDiagnostics | None = None
    postprocess: PostprocessStats | None = None
    postprocess_ms: float | None = None


def diagnose_infeasibility(
    problem: OptimizationProblem,
    candidates: CandidateSet,
    verdict: BatchValidation,
) -> InfeasibilityDiagnostics:
    """Explain an attempt whose candidates were all infeasible.

    Picks the candidate with the smallest total hard violation — the first
    such candidate in first-seen order on a tie, so the choice is
    deterministic for a given raw output — and re-runs the full validator
    on it, so the reported evaluations come from the same arithmetic as a
    ranked solution's. Only the original problem is consulted; the
    solver's energy plays no part (overview principle 2).
    """
    closest = int(np.argmin(verdict.hard_violation_total))
    sample = candidates.sample_dict(closest)
    validation = validate_solution(problem, sample)
    # Same order and association as the batch kernel, so the two agree.
    hard_total = 0.0
    for evaluation in validation.evaluations:
        if evaluation.constraint_type == "hard":
            hard_total += evaluation.violation_amount
    total = len(candidates)
    rates = [
        HardViolationRate(
            constraint_id=constraint_id,
            violated_candidates=violated,
            candidates=total,
            violated_fraction=violated / total,
        )
        for constraint_id, violated in zip(
            verdict.hard_constraint_ids, verdict.hard_violated_counts.tolist()
        )
    ]
    return InfeasibilityDiagnostics(
        closest_candidate=ClosestCandidate(
            variables=sample,
            hard_violation_total=hard_total,
            constraint_evaluations=validation.evaluations,
        ),
        hard_violation_rates=rates,
    )


_SHORTLIST_MIN_ROWS = 256


def _top_k_shortlist(
    primary: np.ndarray, secondary: np.ndarray, top_k: int
) -> np.ndarray | None:
    """Indices of every feasible row that can rank among the first ``top_k``.

    The ranking sorts by ``(primary, secondary, assignment tuple)``, the
    two floats ascending. Let ``(thr_p, thr_s)`` be the first two keys of
    the row in position ``top_k`` of that order. A row with ``primary <
    thr_p``, or with ``primary == thr_p`` and ``secondary < thr_s``,
    precedes it whatever its tuple; a row equal on both keys precedes or
    follows it depending on the tuple; every other row follows it. So the
    rows with ``primary < thr_p`` or ``primary == thr_p and secondary <=
    thr_s`` are a superset of the first ``top_k``, and sorting only them
    with the full key yields the same first ``top_k`` as sorting every
    row: the keys are the same and, the assignments being distinct after
    deduplication, the order is total. The ``==`` on floats is deliberate
    -- the thresholds are copies of array elements, not computed values.
    ``thr_s`` is the ``needed``-th smallest ``secondary`` among the rows
    tied on ``thr_p``, where ``needed`` is ``top_k`` minus the rows below
    ``thr_p``; that is at least 1 and at most the size of the tie.

    Returns ``None`` when the shortlist would not pay or cannot be located:
    ``top_k`` covers every row, the rows are few, or a threshold is NaN
    (``==`` cannot find it; the full sort still puts NaN last).
    """
    count = len(primary)
    if not 1 <= top_k < count or count <= _SHORTLIST_MIN_ROWS:
        return None
    thr_p = np.partition(primary, top_k - 1)[top_k - 1]
    if np.isnan(thr_p):
        return None
    below = primary < thr_p
    equal = primary == thr_p
    needed = top_k - int(below.sum())
    thr_s = np.partition(secondary[equal], needed - 1)[needed - 1]
    if np.isnan(thr_s):
        return None
    return np.flatnonzero(below | (equal & (secondary <= thr_s)))


def _new_assignments(
    samples: np.ndarray, run: PostprocessRun
) -> tuple[np.ndarray, list[str]]:
    """Post-processing rows the solver did not return, deduplicated.

    An assignment the solver also returned stays the solver's candidate
    (its energy and sample count are real); among equal produced rows the
    first one produced -- the lowest selection rank -- keeps its label
    (postprocess spec §4).
    """
    width = samples.shape[1]
    if len(run.samples) == 0:
        return np.zeros((0, width), dtype=np.int64), []
    seen = {row.tobytes() for row in np.ascontiguousarray(samples, dtype=np.int64)}
    rows: list[np.ndarray] = []
    sources: list[str] = []
    for row, source in zip(np.ascontiguousarray(run.samples, dtype=np.int64), run.sources):
        key = row.tobytes()
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        sources.append(source)
    if not rows:
        return np.zeros((0, width), dtype=np.int64), []
    return np.array(rows, dtype=np.int64), sources


def process_candidates(
    problem: OptimizationProblem,
    raw: RawSolverResult,
    internal_variables: Collection[str] = frozenset(),
    top_k: int = 5,
    postprocess: PostprocessRequest | None = None,
) -> ProcessedCandidates:
    """Run the §25 candidate pipeline on raw solver output.

    Deduplicates, validates *every* candidate against the original problem,
    keeps feasible candidates only, computes objective/soft-violation/ranking
    scores, and returns the top ``top_k`` solutions ranked from 1, plus the
    unique and feasible candidate counts.

    Validation runs in two layers with one arithmetic: the vectorised
    :func:`validate_batch` judges all candidates (feasibility, soft score)
    and :func:`evaluate_objective_batch` their objective; the ranking is
    computed from those. Only the top-k then go through
    :func:`validate_solution` to build the full per-constraint evaluations
    for the report. Energy is never consulted for either step.

    The full sort key is ``(ranking_score, objective_value, name-sorted
    assignment tuple)``, each in the objective's direction. On large
    candidate sets only the rows that can still rank among the top-k --
    :func:`_top_k_shortlist` -- are given the tuple key and sorted; the
    first ``top_k`` of that order are provably the first ``top_k`` of the
    full order, so the ranking is unchanged.

    With ``postprocess`` (batch 4 G, postprocess spec 2026-09-23) the best
    distinct samples are also repaired / locally searched
    (:func:`~annealbridge.orchestration.postprocess.run_postprocess`); the
    assignments that produces which the solver did not return join the
    pool, are validated by the very same :func:`validate_batch` and ranked
    by the very same key, with ``source`` naming where they came from, no
    energy and a zero sample count. ``unique_samples``,
    ``feasible_samples`` and the infeasibility diagnostics stay about the
    solver's samples only.
    """
    candidates = deduplicate_samples(raw, internal_variables)
    if len(candidates) == 0:
        if postprocess is None:
            return ProcessedCandidates([], 0, 0)
        return ProcessedCandidates([], 0, 0, postprocess=_empty_stats(), postprocess_ms=0.0)
    minimize = problem.objective.direction == "minimize"
    sign = 1.0 if minimize else -1.0

    verdict = validate_batch(problem, candidates.variables, candidates.samples)
    raw_feasible = int(np.count_nonzero(verdict.feasible))
    pool_samples = candidates.samples
    pool_feasible = verdict.feasible
    pool_soft = verdict.soft_violation_score
    sources: list[str] | None = None
    stats: PostprocessStats | None = None
    postprocess_ms: float | None = None
    if postprocess is not None:
        postprocess_started = time.perf_counter()
        cost = (
            sign
            * evaluate_objective_batch(
                problem.objective, candidates.variables, candidates.samples
            )
            + verdict.soft_violation_score
        )
        run = run_postprocess(
            problem,
            candidates.variables,
            candidates.samples,
            verdict.feasible,
            verdict.hard_violation_total,
            cost,
            postprocess,
        )
        new_rows, new_sources = _new_assignments(candidates.samples, run)
        new_verdict = validate_batch(problem, candidates.variables, new_rows)
        stats = PostprocessStats(
            candidates_selected=run.candidates_selected,
            repair_attempted=run.repair_attempted,
            repair_succeeded=run.repair_succeeded,
            local_search_started=run.local_search_started,
            local_search_improved=run.local_search_improved,
            new_candidates=len(new_rows),
            feasible_added=int(np.count_nonzero(new_verdict.feasible)),
            limit_reached=list(run.limit_reached),
        )
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
        if len(new_rows):
            pool_samples = np.concatenate(
                [candidates.samples.astype(np.int64, copy=False), new_rows]
            )
            pool_feasible = np.concatenate([verdict.feasible, new_verdict.feasible])
            pool_soft = np.concatenate(
                [verdict.soft_violation_score, new_verdict.soft_violation_score]
            )
            sources = ["solver"] * len(candidates) + new_sources

    feasible = np.flatnonzero(pool_feasible)
    if len(feasible) == 0:
        return ProcessedCandidates(
            [],
            len(candidates),
            0,
            diagnose_infeasibility(problem, candidates, verdict),
            stats,
            postprocess_ms,
        )

    feasible_samples = pool_samples[feasible]
    objective_value = evaluate_objective_batch(
        problem.objective, candidates.variables, feasible_samples
    )
    soft_violation_score = pool_soft[feasible]
    if minimize:
        ranking_score = objective_value + soft_violation_score
    else:
        ranking_score = objective_value - soft_violation_score

    # §25.1: ascending ranking_score for minimize, descending for maximize;
    # ties broken by objective_value in the same direction, then by the
    # name-sorted assignment tuple so the order is fully deterministic.
    name_order = sorted(
        range(len(candidates.variables)), key=lambda j: candidates.variables[j]
    )
    primary = sign * ranking_score
    secondary = sign * objective_value
    shortlist = _top_k_shortlist(primary, secondary, top_k)
    if shortlist is None:
        tie_break = _row_keys(feasible_samples[:, name_order])
        order = _lexsort([primary, secondary, *tie_break])
    else:
        tie_break = _row_keys(feasible_samples[shortlist][:, name_order])
        order = shortlist[
            _lexsort([primary[shortlist], secondary[shortlist], *tie_break])
        ]

    solutions: list[Solution] = []
    for rank, position in enumerate(order[:top_k].tolist(), start=1):
        index = int(feasible[position])
        sample = dict(zip(candidates.variables, pool_samples[index].tolist()))
        validation = validate_solution(problem, sample)
        # Rows past the solver's are post-processing products: no compiled
        # sample behind them, so no energy and no raw rows (spec §5).
        produced = index >= len(candidates)
        solutions.append(
            Solution(
                rank=rank,
                variables=sample,
                objective_value=float(objective_value[position]),
                soft_violation_score=validation.soft_violation_score,
                ranking_score=float(ranking_score[position]),
                energy=None if produced else float(candidates.energies[index]),
                sample_count=0 if produced else int(candidates.counts[index]),
                source="solver" if sources is None else sources[index],
                hard_constraints_satisfied=validation.feasible,
                constraint_evaluations=validation.evaluations,
            )
        )
    return ProcessedCandidates(
        solutions,
        len(candidates),
        raw_feasible,
        postprocess=stats,
        postprocess_ms=postprocess_ms,
    )


def _empty_stats() -> PostprocessStats:
    """Statistics of a post-processing run that had no sample to start from."""
    return PostprocessStats(
        candidates_selected=0,
        repair_attempted=0,
        repair_succeeded=0,
        local_search_started=0,
        local_search_improved=0,
        new_candidates=0,
        feasible_added=0,
        limit_reached=[],
    )

"""Opt-in post-processing over the original variables (batch 4 G).

Postprocess spec 2026-09-23 (``.ai-docs/annealbridge_postprocess_spec_2026-09-23.md``),
gated by the experiment in ``.ai-docs/postprocess_exp_2026-09-23/``: the best
few distinct samples of an attempt are greedily repaired when infeasible
and moved to a local optimum when feasible, by single-variable steps
(``x_i ± 1`` inside the bounds) and pair moves over two variables that
share a hard constraint (a swap inside a one-hot group, "take one, drop
one" in a knapsack).

Everything here works on the *business* assignment matrix the compiler
decoded: no slack, no encoding bit, no hard penalty λ (overview principles
3 and 6). The numbers computed here only steer the search; the caller
re-validates every produced assignment against the original problem and
ranks it exactly like a solver sample (principle 2). The one feasibility
check done here (:func:`validate_batch` on the end point) only decides
whether an end point is worth handing back.

The search is deterministic -- no random numbers, fixed move numbering,
ties to the lowest move number, count-based ceilings, one thread -- so the
same samples always give the same output. Both ceilings are counts, not
wall-clock time: ``max_evaluations`` per attempt, every neighbourhood scan
being charged :func:`scan_cost` (the elementary evaluations it performs),
and at most ``4 · n`` steps per repair or local search. Reaching either
stops the search and is reported, never hidden (principle 5).
"""

from dataclasses import dataclass, field

import numpy as np

from annealbridge.models import OptimizationProblem
from annealbridge.validation import tolerance_array, validate_batch

# Operator codes of the constraint rows.
_EQ, _LE, _GE = 0, 1, 2
_OPERATOR_CODES = {"==": _EQ, "<=": _LE, ">=": _GE}

# Relative improvement a step must make (spec §1): repair on the hard
# violation V, local search on the ranking cost g. Relative, so a problem
# with large coefficients is not stalled by rounding noise.
_REPAIR_RTOL = 1e-9
_SEARCH_RTOL = 1e-9

# Scratch-memory bounds of one scan: pair moves are evaluated at most
# _PAIR_CHUNK pairs and _SHARED_CHUNK shared rows at a time, single moves
# at most _ENTRY_CHUNK constraint entries at a time (a variable with more
# entries than that is a chunk of its own). The same bound applies while
# the shared rows are built.
_PAIR_CHUNK = 1 << 16
_SHARED_CHUNK = 1 << 18
_ENTRY_CHUNK = 1 << 18

# A step's direction d, and the pair-move combos (d_i, d_j) in move order,
# as indices into _DIRECTIONS.
_DIRECTIONS = np.array([-1.0, 1.0])
_COMBOS = ((0, 0), (0, 1), (1, 0), (1, 1))
_COMBO_SIDE_I = np.array([si for si, _ in _COMBOS])
_COMBO_SIDE_J = np.array([sj for _, sj in _COMBOS])
_COMBO_I = _DIRECTIONS[_COMBO_SIDE_I]
_COMBO_J = _DIRECTIONS[_COMBO_SIDE_J]

# Steps per repair and per local search: ``_STEP_FACTOR · n``.
_STEP_FACTOR = 4

# Pair-key compaction threshold of ``_hard_pairs`` (see there).
_COMPACT_MIN = 1 << 20

LimitName = str  # "evaluations" | "steps"
_LIMIT_ORDER = ("evaluations", "steps")


@dataclass(frozen=True)
class PostprocessRequest:
    """What the service asks for: how many samples, and the evaluation budget."""

    candidates: int
    max_evaluations: int


@dataclass
class PostprocessRun:
    """What one run produced, in production order.

    ``samples`` rows are candidate assignments over the caller's
    ``variables`` columns, ``sources`` their labels ("repaired",
    "local_search" or "repaired_local_search"). Rows may repeat each other
    or a solver sample; deduplication is the caller's.
    """

    samples: np.ndarray
    sources: list[str]
    candidates_selected: int = 0
    repair_attempted: int = 0
    repair_succeeded: int = 0
    local_search_started: int = 0
    local_search_improved: int = 0
    limit_reached: list[LimitName] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scan cost (pre-solve ceiling check and per-scan charge, spec §6)
# ---------------------------------------------------------------------------
def _supports(problem: OptimizationProblem, column: dict[str, int]):
    """Per constraint: ``(is_hard, sorted columns with a non-zero merged coefficient)``."""
    for constraint in problem.constraints:
        merged: dict[int, float] = {}
        for term in constraint.terms:
            index = column[term.variable]
            merged[index] = merged.get(index, 0.0) + term.coefficient
        yield constraint.type == "hard", np.array(
            sorted(index for index, value in merged.items() if value != 0),
            dtype=np.int64,
        )


def _hard_pairs(
    problem: OptimizationProblem, column: dict[str, int], limit: int
) -> np.ndarray | None:
    """Sorted unique keys ``i * n + j`` (``i < j``) of the variable pairs that
    share a hard constraint, or ``None`` as soon as there are more than
    ``limit`` of them.

    Memory stays bounded by the limit: one constraint with more pairs than
    the limit fails before its pairs are built, and the accumulated keys
    are compacted (``np.unique``) whenever they outgrow twice the last
    compacted size, so they never hold much more than ``3 · limit`` keys.
    """
    n = len(column)
    if limit < 0:
        return None
    chunks: list[np.ndarray] = []
    pending = 0
    base = 0
    for hard, indices in _supports(problem, column):
        k = len(indices)
        if not hard or k < 2:
            continue
        if k * (k - 1) // 2 > limit:
            return None
        first, second = np.triu_indices(k, 1)
        chunks.append(indices[first] * n + indices[second])
        pending += len(first)
        if pending > max(2 * base, _COMPACT_MIN):
            merged = np.unique(np.concatenate(chunks))
            if len(merged) > limit:
                return None
            chunks = [merged]
            pending = base = len(merged)
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    merged = np.unique(np.concatenate(chunks))
    return merged if len(merged) <= limit else None


def _scan_cost(n: int, entries: int, pairs: int, within: int) -> int:
    """``2·(n + z) + 4·(p + s)``: what one neighbourhood scan evaluates.

    Single moves (two per variable) touch every constraint entry of their
    variable twice (``z`` entries in all); pair moves (four per pair ``p``)
    add a correction on every row both variables share, bounded by ``s`` =
    Σ over constraints of ``k·(k−1)/2`` for ``k`` variables in it.
    """
    return 2 * (n + entries) + 4 * (pairs + within)


def _problem_costs(problem: OptimizationProblem, cap: int) -> tuple[int, int] | None:
    """``(setup, scan)`` for ``problem``, or ``None`` once the scan exceeds ``cap``."""
    n = len(problem.variables)
    column = {variable.name: index for index, variable in enumerate(problem.variables)}
    degree = np.zeros(n, dtype=np.int64)
    entries = 0
    within = 0
    for _, indices in _supports(problem, column):
        k = len(indices)
        degree[indices] += 1
        entries += k
        within += k * (k - 1) // 2
        if _scan_cost(n, entries, 0, within) > cap:
            return None
    pairs = _hard_pairs(
        problem, column, (cap - _scan_cost(n, entries, 0, within)) // 4
    )
    if pairs is None:
        return None
    setup = int((degree[pairs // max(n, 1)] + degree[pairs % max(n, 1)]).sum())
    return setup, _scan_cost(n, entries, len(pairs), within)


def scan_cost(problem: OptimizationProblem, cap: int) -> int | None:
    """The evaluations one neighbourhood scan of ``problem`` is charged.

    ``2·(n + z) + 4·(p + s)`` with ``n`` variables, ``z`` non-zero
    constraint coefficients, ``p`` variable pairs sharing a hard constraint
    and ``s`` = Σ over constraints of ``k·(k−1)/2``. It tracks the scan's
    actual work (and so its time and scratch memory), and it does not
    depend on the current assignment, so it is checked before solving.
    Returns ``None`` when it exceeds ``cap``; the pairs are never
    materialised beyond what that decision needs.
    """
    costs = _problem_costs(problem, cap)
    return None if costs is None else costs[1]


def postprocess_costs(problem: OptimizationProblem, cap: int) -> tuple[int, int] | None:
    """``(setup, scan)`` evaluations, or ``None`` when ``setup + scan > cap``.

    ``setup`` is charged once per attempt for finding the rows each pair
    move shares -- Σ over the pairs of both variables' constraint entries
    (2026-09-24 review follow-up: that work is not bounded by the scan
    cost) -- and ``scan`` (:func:`scan_cost`) per neighbourhood scan. A
    problem that cannot afford the setup and one scan within ``cap`` could
    not take a single step, so it is refused before solving.
    """
    costs = _problem_costs(problem, cap)
    if costs is None or costs[0] + costs[1] > cap:
        return None
    return costs


def _chunks(weights: np.ndarray, max_weight: int, max_items: int) -> list[tuple[int, int]]:
    """Consecutive ``[start, stop)`` ranges of at most ``max_items`` items and
    ``max_weight`` total weight (an item heavier than that is a range alone)."""
    total = np.concatenate([[0], np.cumsum(weights, dtype=np.int64)])
    count = len(weights)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < count:
        stop = int(np.searchsorted(total, total[start] + max_weight, side="right")) - 1
        stop = min(max(stop, start + 1), start + max_items, count)
        ranges.append((start, stop))
        start = stop
    return ranges


# ---------------------------------------------------------------------------
# Neighbourhood
# ---------------------------------------------------------------------------
@dataclass
class _State:
    """One assignment and everything a scan needs, kept incrementally."""

    x: np.ndarray  # int64 (n,)
    lhs: np.ndarray  # float64 (m,)   constraint left-hand sides
    sx: np.ndarray  # float64 (n,)   S · x, S = off-diagonal objective matrix
    row_hard: np.ndarray  # float64 (m,)   hard violation per row (0 if satisfied)
    row_unsat: np.ndarray  # int64 (m,)     1 for an unsatisfied hard row
    row_soft: np.ndarray  # float64 (m,)   weight · violation² per soft row
    f: float  # objective
    soft: float  # soft violation score
    violation: float  # V
    unsat: int  # unsatisfied hard rows


class _Neighbourhood:
    """The move set of spec §1 over ``variables`` (the caller's column order)."""

    def __init__(
        self,
        problem: OptimizationProblem,
        variables: list[str],
        pair_limit: int,
    ) -> None:
        self.problem = problem
        self.variables = variables
        n = len(variables)
        self.n = n
        column = {name: index for index, name in enumerate(variables)}
        by_name = {variable.name: variable for variable in problem.variables}
        bounds = [by_name[name].bounds() for name in variables]
        self.lower = np.array([b[0] for b in bounds], dtype=np.int64)
        self.upper = np.array([b[1] for b in bounds], dtype=np.int64)
        self.sign = 1.0 if problem.objective.direction == "minimize" else -1.0

        # Objective: constant + c·x + Σ_{a<b} w_ab x_a x_b + Σ diag_a x_a².
        objective = problem.objective
        self.constant = float(objective.constant)
        self.linear = np.zeros(n)
        for term in objective.linear_terms:
            self.linear[column[term.variable]] += term.coefficient
        self.diagonal = np.zeros(n)
        keys: list[int] = []
        weights: list[float] = []
        for term in objective.quadratic_terms:
            a, b = column[term.variable1], column[term.variable2]
            if a == b:
                self.diagonal[a] += term.coefficient
            else:
                keys.append(min(a, b) * n + max(a, b))
                weights.append(term.coefficient)
        if keys:
            unique_keys, inverse = np.unique(
                np.asarray(keys, dtype=np.int64), return_inverse=True
            )
            summed = np.bincount(inverse, weights=np.asarray(weights), minlength=len(unique_keys))
        else:
            unique_keys = np.zeros(0, dtype=np.int64)
            summed = np.zeros(0)
        first, second = unique_keys // max(n, 1), unique_keys % max(n, 1)
        rows = np.concatenate([first, second])
        cols = np.concatenate([second, first])
        vals = np.concatenate([summed, summed])
        order = np.argsort(rows, kind="stable")
        self.nbr_index = cols[order]
        self.nbr_weight = vals[order]
        self.nbr_ptr = np.concatenate(
            [[0], np.cumsum(np.bincount(rows, minlength=n))]
        ).astype(np.int64)

        # Constraints: one row each, entries merged per (variable, row).
        constraints = problem.constraints
        m = len(constraints)
        self.m = m
        self.rhs = np.array([float(c.rhs) for c in constraints])
        self.op = np.array([_OPERATOR_CODES[c.operator] for c in constraints], dtype=np.int64)
        self.hard = np.array([c.type == "hard" for c in constraints], dtype=bool)
        self.weight = np.array(
            [0.0 if c.type == "hard" else float(c.weight) for c in constraints]
        )
        entry_var: list[np.ndarray] = []
        entry_row: list[np.ndarray] = []
        entry_coef: list[np.ndarray] = []
        within = 0
        for row, (constraint, (_, indices)) in enumerate(
            zip(constraints, _supports(problem, column))
        ):
            merged: dict[int, float] = {}
            for term in constraint.terms:
                index = column[term.variable]
                merged[index] = merged.get(index, 0.0) + term.coefficient
            entry_var.append(indices)
            entry_row.append(np.full(len(indices), row, dtype=np.int64))
            entry_coef.append(np.array([merged[int(i)] for i in indices], dtype=np.float64))
            within += len(indices) * (len(indices) - 1) // 2
        ev = np.concatenate(entry_var) if entry_var else np.zeros(0, dtype=np.int64)
        er = np.concatenate(entry_row) if entry_row else np.zeros(0, dtype=np.int64)
        ec = np.concatenate(entry_coef) if entry_coef else np.zeros(0)
        order = np.lexsort((er, ev))  # by variable, then row
        self.entry_var = ev[order]
        self.entry_row = er[order]
        self.entry_coef = ec[order]
        degree = np.bincount(self.entry_var, minlength=n)
        self.entry_ptr = np.concatenate([[0], np.cumsum(degree)]).astype(np.int64)
        self.single_chunks = _chunks(degree, _ENTRY_CHUNK, max(n, 1))

        # Pair moves: variables sharing a hard constraint, in (i, j) order.
        pair_keys = _hard_pairs(problem, column, pair_limit)
        if pair_keys is None:
            raise ValueError("post-processing pair moves exceed the evaluation budget")
        self.num_pairs = len(pair_keys)
        position = np.searchsorted(unique_keys, pair_keys)
        position = np.minimum(position, max(len(unique_keys) - 1, 0))
        hit = (
            unique_keys[position] == pair_keys
            if len(unique_keys)
            else np.zeros(len(pair_keys), dtype=bool)
        )
        self.pair_weight = np.where(hit, summed[position] if len(summed) else 0.0, 0.0)
        del position, hit
        self.pair_i = (pair_keys // max(n, 1)).astype(np.int32)
        self.pair_j = (pair_keys % max(n, 1)).astype(np.int32)
        del pair_keys
        self._build_shared_rows(degree)
        self.scan_cost = _scan_cost(n, len(self.entry_var), self.num_pairs, within)

    def _build_shared_rows(self, degree: np.ndarray) -> None:
        """Rows both variables of a pair touch, as entry positions of each.

        A pair move's effect is the two single moves' effects plus a
        correction on exactly these rows (the violation is not additive
        where both coefficients act on one row). Built a bounded number of
        entries at a time; stored as int32 positions plus a per-pair offset
        array, so the memory is ``8`` bytes per shared row plus ``8`` per pair.
        """
        m = max(self.m, 1)
        left_out: list[np.ndarray] = []
        right_out: list[np.ndarray] = []
        counts_out: list[np.ndarray] = []
        work = degree[self.pair_i] + degree[self.pair_j]
        for start, stop in _chunks(work, _ENTRY_CHUNK, _PAIR_CHUNK):
            local = np.arange(stop - start, dtype=np.int64)
            left_keys, left_pos = self._entries_of(self.pair_i[start:stop], local, m)
            right_keys, right_pos = self._entries_of(self.pair_j[start:stop], local, m)
            _, li, ri = np.intersect1d(
                left_keys, right_keys, assume_unique=True, return_indices=True
            )
            counts_out.append(np.bincount(left_keys[li] // m, minlength=stop - start))
            left_out.append(left_pos[li].astype(np.int32))
            right_out.append(right_pos[ri].astype(np.int32))
        # What building them cost, charged once per attempt (the same value
        # :func:`postprocess_costs` computes from the problem).
        self.setup_cost = int(work.sum())
        del work
        counts = np.concatenate(counts_out) if counts_out else np.zeros(0, dtype=np.int64)
        self.shared_left = (
            np.concatenate(left_out) if left_out else np.zeros(0, dtype=np.int32)
        )
        self.shared_right = (
            np.concatenate(right_out) if right_out else np.zeros(0, dtype=np.int32)
        )
        self.shared_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.pair_chunks = _chunks(counts, _SHARED_CHUNK, _PAIR_CHUNK)

    def _entries_of(
        self, variables: np.ndarray, local: np.ndarray, m: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keys ``local_pair · m + row`` and entry positions of ``variables``' rows."""
        starts = self.entry_ptr[variables]
        counts = self.entry_ptr[variables.astype(np.int64) + 1] - starts
        owner = np.repeat(local, counts)
        offsets = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        positions = np.repeat(starts, counts) + offsets
        return owner * m + self.entry_row[positions], positions

    # -- row arithmetic (validate_batch's rules, spec §1) --------------------
    def _row_terms(self, actual: np.ndarray, rows: np.ndarray):
        """(hard violation, unsatisfied, soft penalty) of ``rows`` at ``actual``."""
        rhs = self.rhs[rows]
        op = self.op[rows]
        tol = tolerance_array(actual, rhs)
        difference = actual - rhs
        violation = np.where(
            op == _EQ,
            np.abs(difference),
            np.where(op == _LE, np.maximum(0.0, difference), np.maximum(0.0, rhs - actual)),
        )
        satisfied = np.where(
            op == _EQ,
            violation <= tol,
            np.where(op == _LE, actual <= rhs + tol, actual >= rhs - tol),
        )
        hard = self.hard[rows]
        unsat = hard & ~satisfied
        hard_violation = np.where(unsat, violation, 0.0)
        soft = np.where(hard, 0.0, self.weight[rows] * violation * violation)
        return hard_violation, unsat.astype(np.int64), soft

    def state(self, x: np.ndarray) -> _State:
        x = np.array(x, dtype=np.int64, copy=True)
        lhs = np.zeros(self.m)
        np.add.at(lhs, self.entry_row, self.entry_coef * x[self.entry_var])
        rows = np.arange(self.m, dtype=np.int64)
        row_hard, row_unsat, row_soft = self._row_terms(lhs, rows)
        sx = np.zeros(self.n)
        owner = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.nbr_ptr))
        np.add.at(sx, owner, self.nbr_weight * x[self.nbr_index])
        xf = x.astype(np.float64)
        f = (
            self.constant
            + float(self.linear @ xf)
            + 0.5 * float(xf @ sx)
            + float(self.diagonal @ (xf * xf))
        )
        unsat = int(row_unsat.sum())
        return _State(
            x=x,
            lhs=lhs,
            sx=sx,
            row_hard=row_hard,
            row_unsat=row_unsat,
            row_soft=row_soft,
            f=f,
            soft=float(row_soft.sum()),
            violation=0.0 if unsat == 0 else float(row_hard.sum()),
            unsat=unsat,
        )

    def cost(self, state: _State) -> float:
        return self.sign * state.f + state.soft

    # -- scan -----------------------------------------------------------------
    def _single_deltas(self, st: _State):
        """Per direction d ∈ (−1, +1), per variable: ΔV, Δunsat, Δsoft, Δf, valid.

        Both directions go through one ``_row_terms`` call per chunk of
        variables (rows of a 2-D array); a variable's entries never straddle
        two chunks, so every per-variable sum is accumulated in entry order.
        """
        n = self.n
        dv = np.zeros((2, n))
        du = np.zeros((2, n))
        ds = np.zeros((2, n))
        for first_var, stop_var in self.single_chunks:
            lo, hi = self.entry_ptr[first_var], self.entry_ptr[stop_var]
            if lo == hi:
                continue
            rows = self.entry_row[lo:hi]
            width = stop_var - first_var
            actual = (
                st.lhs[rows][None, :]
                + self.entry_coef[lo:hi][None, :] * _DIRECTIONS[:, None]
            )
            hard_v, unsat, soft = self._row_terms(actual, rows)
            bins = (
                (self.entry_var[lo:hi] - first_var)[None, :]
                + width * np.arange(2)[:, None]
            ).ravel()

            def per_variable(values: np.ndarray, old: np.ndarray) -> np.ndarray:
                return np.bincount(
                    bins,
                    weights=(values - old[None, :]).astype(np.float64).ravel(),
                    minlength=2 * width,
                ).reshape(2, width)

            dv[:, first_var:stop_var] = per_variable(hard_v, st.row_hard[rows])
            du[:, first_var:stop_var] = per_variable(unsat, st.row_unsat[rows])
            ds[:, first_var:stop_var] = per_variable(soft, st.row_soft[rows])
        du_int = np.rint(du).astype(np.int64)
        xf = st.x.astype(np.float64)
        out = []
        for side, d in enumerate((-1.0, 1.0)):
            df = self.linear * d + d * st.sx + self.diagonal * (2.0 * xf * d + d * d)
            moved = st.x + int(d)
            valid = (moved >= self.lower) & (moved <= self.upper)
            out.append((dv[side], du_int[side], ds[side], df, valid))
        return out

    def _pair_chunk(self, st: _State, singles, start: int, stop: int):
        """ΔV, Δunsat, Δsoft, Δf, valid for pairs [start, stop), shape (pairs, 4)."""
        pi = self.pair_i[start:stop]
        pj = self.pair_j[start:stop]
        count = stop - start
        lo, hi = self.shared_ptr[start], self.shared_ptr[stop]
        owner = np.repeat(
            np.arange(count, dtype=np.int64), np.diff(self.shared_ptr[start : stop + 1])
        )
        left = self.shared_left[lo:hi]
        right = self.shared_right[lo:hi]
        rows = self.entry_row[left]
        base = st.lhs[rows]
        a_left = self.entry_coef[left]
        a_right = self.entry_coef[right]
        old = (st.row_hard[rows], st.row_unsat[rows], st.row_soft[rows])
        # Row k of these 2-D arrays is combo k of _COMBOS; the three
        # _row_terms calls cover all four combos at once.
        both = self._row_terms(
            base[None, :]
            + a_left[None, :] * _COMBO_I[:, None]
            + a_right[None, :] * _COMBO_J[:, None],
            rows,
        )
        only_i = self._row_terms(base[None, :] + a_left[None, :] * _DIRECTIONS[:, None], rows)
        only_j = self._row_terms(base[None, :] + a_right[None, :] * _DIRECTIONS[:, None], rows)
        bins = (owner[None, :] + count * np.arange(4)[:, None]).ravel()
        corr = [
            np.bincount(
                bins,
                weights=(
                    both[t]
                    - only_i[t][_COMBO_SIDE_I]
                    - only_j[t][_COMBO_SIDE_J]
                    + old[t][None, :]
                ).astype(np.float64).ravel(),
                minlength=4 * count,
            ).reshape(4, count)
            for t in range(3)
        ]
        shape = (count, 4)
        dv = np.empty(shape)
        du = np.empty(shape, dtype=np.int64)
        ds = np.empty(shape)
        df = np.empty(shape)
        valid = np.empty(shape, dtype=bool)
        for k, (si, sj) in enumerate(_COMBOS):
            di, dj = _DIRECTIONS[si], _DIRECTIONS[sj]
            vi, ui, s_i, fi, oki = singles[si]
            vj, uj, s_j, fj, okj = singles[sj]
            dv[:, k] = vi[pi] + vj[pj] + corr[0][k]
            du[:, k] = ui[pi] + uj[pj] + np.rint(corr[1][k]).astype(np.int64)
            ds[:, k] = s_i[pi] + s_j[pj] + corr[2][k]
            df[:, k] = fi[pi] + fj[pj] + self.pair_weight[start:stop] * di * dj
            valid[:, k] = oki[pi] & okj[pj]
        return dv, du, ds, df, valid

    def _segments(self, st: _State):
        """Yield (first move number, ΔV, Δunsat, Δsoft, Δf, valid), flat, in move order."""
        singles = self._single_deltas(st)
        # Moves 2i + 0 (d = −1) and 2i + 1 (d = +1).
        yield (
            0,
            *(np.stack([singles[0][t], singles[1][t]], axis=1).ravel() for t in range(5)),
        )
        for start, stop in self.pair_chunks:
            chunk = self._pair_chunk(st, singles, start, stop)
            yield (2 * self.n + 4 * start, *(array.ravel() for array in chunk))

    def best_repair(self, st: _State):
        """Lowest V', then lowest g', then lowest move number; None if V cannot drop."""
        threshold = st.violation - _REPAIR_RTOL * max(1.0, st.violation)
        best = None
        for first, dv, du, ds, df, valid in self._segments(st):
            unsat = st.unsat + du
            violation = np.where(unsat == 0, 0.0, st.violation + dv)
            cost = self.sign * (st.f + df) + st.soft + ds
            ok = valid & (violation < threshold)
            if not ok.any():
                continue
            v_min = violation[ok].min()
            tie = ok & (violation == v_min)
            g_min = cost[tie].min()
            k = int(np.argmax(tie & (cost == g_min)))
            if best is None or v_min < best[1] or (v_min == best[1] and g_min < best[2]):
                best = (first + k, float(v_min), float(g_min), int(unsat[k]), float(ds[k]), float(df[k]))
        return best

    def best_search(self, st: _State):
        """Lowest g' among feasibility-keeping improving moves; None at a local optimum."""
        g = self.cost(st)
        threshold = g - _SEARCH_RTOL * max(1.0, abs(g))
        best = None
        for first, _dv, du, ds, df, valid in self._segments(st):
            cost = self.sign * (st.f + df) + st.soft + ds
            ok = valid & (st.unsat + du == 0) & (cost < threshold)
            if not ok.any():
                continue
            g_min = cost[ok].min()
            k = int(np.argmax(ok & (cost == g_min)))
            if best is None or g_min < best[2]:
                best = (first + k, 0.0, float(g_min), 0, float(ds[k]), float(df[k]))
        return best

    def apply(self, st: _State, move) -> None:
        """Apply ``move`` (a best_* result) to ``st`` in place."""
        number, violation, _cost, unsat, ds, df = move
        if number < 2 * self.n:
            changes = [(number // 2, (-1, 1)[number % 2])]
        else:
            pair, k = divmod(number - 2 * self.n, 4)
            changes = [
                (int(self.pair_i[pair]), (-1, 1)[k // 2]),
                (int(self.pair_j[pair]), (-1, 1)[k % 2]),
            ]
        touched = []
        for index, d in changes:
            st.x[index] += d
            lo, hi = self.entry_ptr[index], self.entry_ptr[index + 1]
            rows = self.entry_row[lo:hi]
            st.lhs[rows] += self.entry_coef[lo:hi] * float(d)
            touched.append(rows)
            nlo, nhi = self.nbr_ptr[index], self.nbr_ptr[index + 1]
            st.sx[self.nbr_index[nlo:nhi]] += self.nbr_weight[nlo:nhi] * float(d)
        rows = np.unique(np.concatenate(touched)) if touched else np.zeros(0, np.int64)
        hard_v, row_unsat, soft = self._row_terms(st.lhs[rows], rows)
        st.row_hard[rows] = hard_v
        st.row_unsat[rows] = row_unsat
        st.row_soft[rows] = soft
        st.f += df
        st.soft += ds
        st.unsat = unsat
        st.violation = 0.0 if unsat == 0 else violation


class _Budget:
    """The per-attempt evaluation budget; every scan costs :func:`scan_cost`."""

    def __init__(self, total: int, per_scan: int) -> None:
        self.remaining = total
        self.per_scan = per_scan

    def take(self, amount: int | None = None) -> bool:
        """Charge one scan (or ``amount``); False, charging nothing, if it does not fit."""
        cost = self.per_scan if amount is None else amount
        if self.remaining < cost:
            return False
        self.remaining -= cost
        return True


def _repair(hood: _Neighbourhood, budget: _Budget, x: np.ndarray):
    """(end state or None, limit hit or None) — greedy descent on V to zero."""
    st = hood.state(x)
    for _ in range(_STEP_FACTOR * hood.n):
        if not budget.take():
            return None, "evaluations"
        move = hood.best_repair(st)
        if move is None:
            return None, None
        hood.apply(st, move)
        if st.unsat == 0:
            return st, None
    return None, "steps"


def _local_search(hood: _Neighbourhood, budget: _Budget, x: np.ndarray):
    """(end state, steps taken, limit hit or None) — best improvement, stays feasible."""
    st = hood.state(x)
    for step in range(_STEP_FACTOR * hood.n):
        if not budget.take():
            return st, step, "evaluations"
        move = hood.best_search(st)
        if move is None:
            return st, step, None
        hood.apply(st, move)
    return st, _STEP_FACTOR * hood.n, "steps"


def run_postprocess(
    problem: OptimizationProblem,
    variables: list[str],
    samples: np.ndarray,
    feasible: np.ndarray,
    hard_violation: np.ndarray,
    cost: np.ndarray,
    request: PostprocessRequest,
) -> PostprocessRun:
    """Repair and locally search the best ``request.candidates`` samples.

    ``samples`` are the attempt's deduplicated business assignments (first
    appearance order), with their re-validated ``feasible`` verdict, total
    ``hard_violation`` and ranking ``cost`` (sign · objective + soft score,
    lower is better). Selection is by ``(hard_violation, cost, row)``.
    Each selected infeasible sample is repaired; each feasible one -- and
    each successful repair -- starts one local search (a start already
    searched from is not searched again). Stops at the evaluation budget.
    """
    empty = np.zeros((0, len(variables)), dtype=np.int64)
    run = PostprocessRun(samples=empty, sources=[])
    count = samples.shape[0]
    if count == 0:
        return run
    order = np.lexsort((np.arange(count), cost, hard_violation))
    selected = order[: request.candidates]
    hood = _Neighbourhood(problem, variables, request.max_evaluations // 4)
    budget = _Budget(request.max_evaluations, hood.scan_cost)
    if not budget.take(hood.setup_cost):
        # Only reachable by a caller that skipped the pre-solve check.
        run.limit_reached = ["evaluations"]
        return run
    rows: list[np.ndarray] = []
    limits: set[str] = set()
    searched: set[bytes] = set()

    def still_feasible(x: np.ndarray) -> bool:
        return bool(validate_batch(problem, variables, x[None, :]).feasible[0])

    for index in selected.tolist():
        run.candidates_selected += 1
        start = samples[index]
        repaired = False
        if not feasible[index]:
            run.repair_attempted += 1
            end, limit = _repair(hood, budget, start)
            if limit is not None:
                limits.add(limit)
            if end is None or not still_feasible(end.x):
                if limit == "evaluations":
                    break
                continue
            run.repair_succeeded += 1
            repaired = True
            start = end.x
            rows.append(start.copy())
            run.sources.append("repaired")
        key = np.ascontiguousarray(start, dtype=np.int64).tobytes()
        if key in searched:
            continue
        searched.add(key)
        run.local_search_started += 1
        end, steps, limit = _local_search(hood, budget, start)
        if limit is not None:
            limits.add(limit)
        if steps > 0 and still_feasible(end.x):
            run.local_search_improved += 1
            rows.append(end.x.copy())
            run.sources.append("repaired_local_search" if repaired else "local_search")
        if limit == "evaluations":
            break

    run.samples = np.array(rows, dtype=np.int64) if rows else empty
    run.limit_reached = [name for name in _LIMIT_ORDER if name in limits]
    return run

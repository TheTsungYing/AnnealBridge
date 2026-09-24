"""The backends and post-processing under an ``Interrupt`` (batch 6 J).

Time-limit spec 2026-09-24 §5 and §9 items 2-4, one layer below the
service: each backend's ``solve`` is called directly with an interrupt, so
the checkpoints can be driven exactly. The interrupt is the real
:class:`Interrupt` on a counting clock -- every ``should_stop()`` reads the
clock once and the clock returns 1, 2, 3, ..., so ``deadline=k`` makes the
``k``-th check the first to say stop -- or a stub where a test needs a
check that raises.

For every backend that declares ``supports_interrupt``:

* a check that never fires returns exactly the uninterrupted result;
* a stop returns only reads that completed their whole schedule, as a
  prefix of the uninterrupted rows (sequential shards / batches), marked
  ``interrupted``; a stop before any read gives zero rows over the BQM's
  own variables;
* every shard thread has returned by the time ``solve`` does.

The annealer's callback check that raises becomes a solver failure, never
a silently cut run; ``run_shards_interruptible`` and ``run_postprocess``
are pinned on their own.
"""

import itertools
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import dimod
import numpy as np
import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.exceptions import SolverExecutionError
from annealbridge.interrupt import CancelToken, Interrupt
from annealbridge.models import CompiledProblem, OptimizationProblem, SolverPreferences
from annealbridge.orchestration.candidates import process_candidates
from annealbridge.orchestration.postprocess import PostprocessRequest
from annealbridge.solvers import (
    RawSolverResult,
    SimulatedAnnealingBackend,
    SimulatedBifurcationBackend,
    TabuBackend,
    sharding,
)
from annealbridge.solvers.sharding import (
    READS_PER_SHARD,
    SHARD_THREAD_PREFIX,
    run_shards,
    run_shards_interruptible,
)
from annealbridge.solvers.simulated_bifurcation import AGENTS_PER_BATCH
from tests.conftest import EXAMPLES_DIR


def counting_interrupt(k: int) -> Interrupt:
    """An interrupt whose ``k``-th ``should_stop()`` is the first True.

    Thread-safe, so a shard pool may poll it concurrently.
    """
    counter = itertools.count(1)
    lock = threading.Lock()

    def clock() -> float:
        with lock:
            return float(next(counter))

    return Interrupt(deadline=float(k), token=None, clock=clock)


def never() -> Interrupt:
    """An interrupt that exists but never fires (a token nobody cancels)."""
    return Interrupt(deadline=None, token=CancelToken())


def at_once() -> Interrupt:
    """An interrupt that has already fired."""
    token = CancelToken()
    token.cancel()
    return Interrupt(deadline=None, token=token)


def shard_threads() -> list[str]:
    return [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(SHARD_THREAD_PREFIX)
    ]


def knapsack_problem() -> OptimizationProblem:
    payload = json.loads((EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8"))
    return OptimizationProblem.model_validate(payload)


@pytest.fixture(scope="module")
def compiled() -> CompiledProblem:
    return BQMCompiler().compile(knapsack_problem(), hard_penalty=100.0)


def assert_same(left: RawSolverResult, right: RawSolverResult) -> None:
    assert left.variables == right.variables
    assert left.samples.dtype == right.samples.dtype
    assert np.array_equal(left.samples, right.samples)
    assert np.array_equal(left.energies, right.energies)
    assert left.metadata == right.metadata
    assert left.interrupted is right.interrupted is False


def assert_prefix(cut: RawSolverResult, full: RawSolverResult) -> None:
    assert cut.interrupted is True
    assert cut.variables == full.variables
    assert cut.num_samples < full.num_samples
    assert np.array_equal(cut.samples, full.samples[: cut.num_samples])
    assert np.array_equal(cut.energies, full.energies[: cut.num_samples])
    # The metadata still says what was asked for, not what completed.
    assert cut.metadata == full.metadata


def assert_empty(result: RawSolverResult, compiled: CompiledProblem) -> None:
    assert result.interrupted is True
    assert result.num_samples == 0
    assert result.variables == [str(v) for v in compiled.model.variables]
    assert result.samples.shape == (0, len(result.variables))
    assert result.samples.dtype == np.int8
    assert result.energies.shape == (0,)


# --------------------------------------------------------------------------
# simulated_annealing
# --------------------------------------------------------------------------


class TestSimulatedAnnealing:
    @pytest.mark.parametrize(
        ("num_reads", "workers"), [(10, 1), (25, 4), (60, 1), (130, 4)]
    )
    def test_an_interrupt_that_never_fires_changes_nothing(
        self, compiled, num_reads, workers
    ):
        backend = SimulatedAnnealingBackend(workers=workers)
        prefs = SolverPreferences(num_reads=num_reads, seed=7)

        assert_same(backend.solve(compiled, prefs, interrupt=never()), backend.solve(compiled, prefs))

    def test_one_shard_stops_after_the_read_the_check_follows(self, compiled):
        backend = SimulatedAnnealingBackend(workers=1)
        prefs = SolverPreferences(num_reads=10, seed=7)
        full = backend.solve(compiled, prefs)

        # Check 1 is before the call, then one after each completed read:
        # the 4th check, after read 3, stops it with 3 reads.
        cut = backend.solve(compiled, prefs, interrupt=counting_interrupt(4))

        assert_prefix(cut, full)
        assert cut.num_samples == 3

    def test_sequential_shards_return_a_prefix(self, compiled):
        backend = SimulatedAnnealingBackend(workers=1)
        prefs = SolverPreferences(num_reads=60, seed=7)
        full = backend.solve(compiled, prefs)

        cut = backend.solve(compiled, prefs, interrupt=counting_interrupt(30))

        assert_prefix(cut, full)
        # Past the first shard (25 reads), inside the second.
        assert READS_PER_SHARD < cut.num_samples < 2 * READS_PER_SHARD

    @pytest.mark.parametrize(("num_reads", "workers"), [(10, 1), (60, 1), (130, 4)])
    def test_a_stop_before_any_read_returns_no_rows(self, compiled, num_reads, workers):
        backend = SimulatedAnnealingBackend(workers=workers)
        prefs = SolverPreferences(num_reads=num_reads, seed=7)

        result = backend.solve(compiled, prefs, interrupt=at_once())

        assert_empty(result, compiled)
        assert result.metadata.num_reads_requested == num_reads
        assert result.backend == "simulated_annealing"

    def test_concurrent_shards_keep_only_completed_reads(self, compiled):
        backend = SimulatedAnnealingBackend(workers=4)
        prefs = SolverPreferences(num_reads=130, seed=7)
        full = backend.solve(compiled, prefs)
        full_rows = {tuple(row) for row in full.samples.tolist()}

        cut = backend.solve(compiled, prefs, interrupt=counting_interrupt(40))

        assert cut.interrupted is True
        assert 0 < cut.num_samples < full.num_samples
        # Not necessarily a prefix with several workers, but every row is
        # one of the uninterrupted run's reads with its own energy.
        assert {tuple(row) for row in cut.samples.tolist()} <= full_rows
        assert np.array_equal(
            cut.energies, compiled.model.energies((cut.samples, cut.variables))
        )
        assert shard_threads() == []

    def test_a_failing_check_is_a_solver_failure_not_a_cut(self, compiled):
        class Exploding:
            calls = 0

            def should_stop(self) -> bool:
                # The first check is the backend's own, before the sampler;
                # the second is inside the sampler's callback.
                self.calls += 1
                if self.calls >= 2:
                    raise RuntimeError("clock exploded")
                return False

        backend = SimulatedAnnealingBackend(workers=1)

        with pytest.raises(SolverExecutionError, match="clock exploded") as exc_info:
            backend.solve(
                compiled, SolverPreferences(num_reads=10, seed=7), interrupt=Exploding()
            )
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @staticmethod
    def exploding_on(k: int):
        """A thread-safe stub whose ``k``-th ``should_stop()`` raises."""

        class ExplodesOnce:
            def __init__(self) -> None:
                self.calls = 0
                self.lock = threading.Lock()

            def should_stop(self) -> bool:
                with self.lock:
                    self.calls += 1
                    calls = self.calls
                if calls == k:
                    raise RuntimeError("clock exploded")
                return False

        return ExplodesOnce()

    @staticmethod
    def count_shards(backend, monkeypatch) -> list[int]:
        """Record the read count of every sampler call (one per shard)."""
        started: list[int] = []
        lock = threading.Lock()
        original = backend._sample

        def recording(model, num_reads, *args, **kwargs):
            with lock:
                started.append(num_reads)
            return original(model, num_reads, *args, **kwargs)

        monkeypatch.setattr(backend, "_sample", recording)
        return started

    def test_a_failing_check_in_a_shard_skips_every_later_shard(
        self, compiled, monkeypatch
    ):
        # Shards of 25, 25, 10 run one at a time. Check 1 is before shard 0
        # (through the same wrapper); check 3 is the callback after read 2
        # of shard 0, and it fails. The wrapper keeps the error, stops the
        # sampler, and answers "stop" before shards 1 and 2 as well, so
        # neither starts; the kept error then fails the solve.
        backend = SimulatedAnnealingBackend(workers=1)
        started = self.count_shards(backend, monkeypatch)
        interrupt = self.exploding_on(3)

        with pytest.raises(SolverExecutionError, match="clock exploded") as exc_info:
            backend.solve(
                compiled, SolverPreferences(num_reads=60, seed=7), interrupt=interrupt
            )

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert started == [25]
        # Nothing asked the interrupt again after it failed.
        assert interrupt.calls == 3

    def test_a_failing_check_with_several_workers_starts_no_new_shard(
        self, compiled, monkeypatch
    ):
        # 130 reads = 6 shards on 3 workers. Check 5 fails, whether it is a
        # shard's start check or a read callback: at most the shards already
        # running (one per worker) have started, never all six.
        backend = SimulatedAnnealingBackend(workers=3)
        started = self.count_shards(backend, monkeypatch)

        with pytest.raises(SolverExecutionError, match="clock exploded"):
            backend.solve(
                compiled,
                SolverPreferences(num_reads=130, seed=7),
                interrupt=self.exploding_on(5),
            )

        assert 1 <= len(started) <= 3
        assert shard_threads() == []

    def test_no_shard_thread_outlives_a_solve(self, compiled):
        backend = SimulatedAnnealingBackend(workers=4)
        prefs = SolverPreferences(num_reads=130, seed=7)

        backend.solve(compiled, prefs, interrupt=never())
        assert shard_threads() == []
        backend.solve(compiled, prefs, interrupt=counting_interrupt(10))
        assert shard_threads() == []
        backend.solve(compiled, prefs, interrupt=at_once())
        assert shard_threads() == []


# --------------------------------------------------------------------------
# tabu
# --------------------------------------------------------------------------


class TestTabu:
    @pytest.mark.parametrize(("num_reads", "workers"), [(10, 1), (60, 1), (60, 3)])
    def test_an_interrupt_that_never_fires_changes_nothing(
        self, compiled, num_reads, workers
    ):
        backend = TabuBackend(workers=workers)
        prefs = SolverPreferences(backend="tabu", num_reads=num_reads, seed=7)

        assert_same(backend.solve(compiled, prefs, interrupt=never()), backend.solve(compiled, prefs))

    def test_the_check_is_before_each_shard_only(self, compiled):
        backend = TabuBackend(workers=1)
        prefs = SolverPreferences(backend="tabu", num_reads=60, seed=7)
        full = backend.solve(compiled, prefs)

        # Shards of 25, 25, 10: the third check stops the third shard.
        cut = backend.solve(compiled, prefs, interrupt=counting_interrupt(3))

        assert_prefix(cut, full)
        assert cut.num_samples == 2 * READS_PER_SHARD

    def test_one_started_shard_runs_to_its_end(self, compiled):
        # One shard: one check before it, none inside.
        backend = TabuBackend(workers=1)
        prefs = SolverPreferences(backend="tabu", num_reads=10, seed=7)
        interrupt = counting_interrupt(2)

        result = backend.solve(compiled, prefs, interrupt=interrupt)

        assert result.num_samples == 10
        assert result.interrupted is False
        assert interrupt.should_stop() is True  # the 2nd check: only one was made

    @pytest.mark.parametrize(("num_reads", "workers"), [(10, 1), (60, 3)])
    def test_a_stop_before_any_shard_returns_no_rows(self, compiled, num_reads, workers):
        backend = TabuBackend(workers=workers)
        prefs = SolverPreferences(backend="tabu", num_reads=num_reads, seed=7)

        result = backend.solve(compiled, prefs, interrupt=at_once())

        assert_empty(result, compiled)
        assert result.metadata.num_reads_requested == num_reads
        assert shard_threads() == []


# --------------------------------------------------------------------------
# simulated_bifurcation
# --------------------------------------------------------------------------


STEPS = 50


class TestSimulatedBifurcation:
    @pytest.mark.parametrize("num_reads", [100, AGENTS_PER_BATCH + 76])
    def test_an_interrupt_that_never_fires_changes_nothing(self, compiled, num_reads):
        backend = SimulatedBifurcationBackend()
        prefs = SolverPreferences(
            backend="simulated_bifurcation", num_reads=num_reads, num_sweeps=STEPS, seed=7
        )

        assert_same(backend.solve(compiled, prefs, interrupt=never()), backend.solve(compiled, prefs))

    def test_a_batch_cut_mid_schedule_is_dropped(self, compiled):
        backend = SimulatedBifurcationBackend()
        prefs = SolverPreferences(
            backend="simulated_bifurcation",
            num_reads=AGENTS_PER_BATCH + 76,
            num_sweeps=STEPS,
            seed=7,
        )
        full = backend.solve(compiled, prefs)

        # Check 1 before batch 1, one after each of its steps but the first
        # (STEPS - 1), check STEPS + 1 before batch 2, then a few steps in.
        cut = backend.solve(compiled, prefs, interrupt=counting_interrupt(STEPS + 6))

        assert_prefix(cut, full)
        assert cut.num_samples == AGENTS_PER_BATCH

    def test_a_stop_before_the_second_batch_keeps_the_first(self, compiled):
        backend = SimulatedBifurcationBackend()
        prefs = SolverPreferences(
            backend="simulated_bifurcation",
            num_reads=AGENTS_PER_BATCH + 76,
            num_sweeps=STEPS,
            seed=7,
        )
        full = backend.solve(compiled, prefs)

        cut = backend.solve(compiled, prefs, interrupt=counting_interrupt(STEPS + 1))

        assert_prefix(cut, full)
        assert cut.num_samples == AGENTS_PER_BATCH

    @pytest.mark.parametrize("k", [1, 2, STEPS])
    def test_a_cut_first_batch_leaves_no_rows(self, compiled, k):
        backend = SimulatedBifurcationBackend()
        prefs = SolverPreferences(
            backend="simulated_bifurcation", num_reads=100, num_sweeps=STEPS, seed=7
        )

        result = backend.solve(compiled, prefs, interrupt=counting_interrupt(k))

        assert_empty(result, compiled)
        assert result.metadata.num_reads_requested == 100

    def test_the_last_step_is_not_checked(self, compiled):
        # STEPS checks in all (before the batch, after steps 1..STEPS-1):
        # a stop due only on the next check lets the batch complete.
        backend = SimulatedBifurcationBackend()
        prefs = SolverPreferences(
            backend="simulated_bifurcation", num_reads=100, num_sweeps=STEPS, seed=7
        )

        result = backend.solve(compiled, prefs, interrupt=counting_interrupt(STEPS + 1))

        assert_same(result, backend.solve(compiled, prefs))


# --------------------------------------------------------------------------
# run_shards_interruptible
# --------------------------------------------------------------------------


def _shard_sampleset(shard: int) -> dimod.SampleSet:
    return dimod.SampleSet.from_samples(
        [{"x": shard % 2}], vartype=dimod.BINARY, energy=[float(shard)]
    )


class TestRunShardsInterruptible:
    @pytest.mark.parametrize("workers", [1, 3])
    def test_never_stopping_is_run_shards(self, workers):
        merged, skipped = run_shards_interruptible(
            _shard_sampleset, 5, workers, lambda: False
        )
        plain = run_shards(_shard_sampleset, 5, workers)

        assert skipped is False
        assert merged.record.energy.tolist() == plain.record.energy.tolist()

    def test_sequential_shards_stop_at_the_check(self):
        calls: list[int] = []

        def run(shard: int) -> dimod.SampleSet:
            calls.append(shard)
            return _shard_sampleset(shard)

        checks = itertools.count(1)
        merged, skipped = run_shards_interruptible(
            run, 5, 1, lambda: next(checks) >= 3
        )

        assert calls == [0, 1]
        assert skipped is True
        assert merged.record.energy.tolist() == [0.0, 1.0]

    @pytest.mark.parametrize("workers", [1, 3])
    def test_stopping_at_once_runs_nothing(self, workers):
        calls: list[int] = []

        def run(shard: int) -> dimod.SampleSet:
            calls.append(shard)
            return _shard_sampleset(shard)

        merged, skipped = run_shards_interruptible(run, 5, workers, lambda: True)

        assert (merged, skipped, calls) == (None, True, [])
        assert shard_threads() == []

    def test_concurrent_shards_merge_in_shard_order(self):
        # Shard 0 is held until shard 1 (on the other worker) has said
        # stop, so shard 0 finishes last and every later shard is skipped:
        # the merge must still put 0 before 1.
        stop = threading.Event()

        def run(shard: int) -> dimod.SampleSet:
            if shard == 0:
                assert stop.wait(timeout=10)
            if shard == 1:
                stop.set()
            return _shard_sampleset(shard)

        merged, skipped = run_shards_interruptible(run, 8, 2, stop.is_set)

        assert skipped is True
        assert merged.record.energy.tolist() == [0.0, 1.0]
        assert shard_threads() == []


class TestRunShardsInterruptibleFailure:
    """A failed shard: the ones not started are skipped, the running ones
    are waited for, then the failure propagates unchanged."""

    def test_the_failure_waits_for_the_running_shard_and_skips_the_rest(self):
        # Shard 1 is running (and slow) when shard 0 fails; ``map`` sees
        # shard 0's failure first, so without waiting the exception would
        # leave while shard 1 still runs.
        called: list[int] = []
        ended: dict[int, float] = {}
        shard_one_running = threading.Event()
        lock = threading.Lock()

        def run(shard: int) -> dimod.SampleSet:
            with lock:
                called.append(shard)
            if shard == 0:
                assert shard_one_running.wait(timeout=10)
                raise RuntimeError("shard exploded")
            if shard == 1:
                shard_one_running.set()
                time.sleep(0.3)
                ended[shard] = time.perf_counter()
            return _shard_sampleset(shard)

        with pytest.raises(RuntimeError) as exc_info:
            run_shards_interruptible(run, 6, 2, lambda: False)
        caught = time.perf_counter()

        assert type(exc_info.value) is RuntimeError
        assert str(exc_info.value) == "shard exploded"
        assert 1 in ended and ended[1] < caught
        # Shards 2..5 had not started when shard 0 failed: never run.
        assert sorted(called) == [0, 1]
        assert shard_threads() == []

    def test_the_pool_is_shut_down_waiting(self, monkeypatch):
        record: dict = {}

        class Recording(ThreadPoolExecutor):
            def __init__(self, max_workers=None, thread_name_prefix=""):
                record["thread_name_prefix"] = thread_name_prefix
                super().__init__(
                    max_workers=max_workers, thread_name_prefix=thread_name_prefix
                )

            def shutdown(self, wait=True, *, cancel_futures=False):
                record.setdefault("shutdown", []).append((wait, cancel_futures))
                super().shutdown(wait=wait, cancel_futures=cancel_futures)

        monkeypatch.setattr(sharding, "ThreadPoolExecutor", Recording)

        def run(shard: int) -> dimod.SampleSet:
            if shard == 1:
                raise RuntimeError("shard exploded")
            return _shard_sampleset(shard)

        with pytest.raises(RuntimeError, match="shard exploded"):
            run_shards_interruptible(run, 6, 2, lambda: False)

        # Waits, as the uninterruptible run_shards does too (pinned in
        # tests/unit/test_sharding.py).
        assert record["shutdown"] == [(True, True)]
        assert record["thread_name_prefix"] == SHARD_THREAD_PREFIX

    def test_a_sequential_failure_runs_no_later_shard(self):
        called: list[int] = []

        def run(shard: int) -> dimod.SampleSet:
            called.append(shard)
            if shard == 1:
                raise RuntimeError("shard exploded")
            return _shard_sampleset(shard)

        with pytest.raises(RuntimeError, match="shard exploded"):
            run_shards_interruptible(run, 5, 1, lambda: False)

        assert called == [0, 1]


# --------------------------------------------------------------------------
# Post-processing (spec §9 item 3, "後處理停止")
# --------------------------------------------------------------------------


def assignment_problem() -> OptimizationProblem:
    """Two tasks, three workers, one worker per task; optimum cost 3."""
    names = [f"t{t}_w{w}" for t in (1, 2) for w in range(3)]
    costs = {"t1_w0": 1, "t1_w1": 2, "t1_w2": 5, "t2_w0": 1, "t2_w1": 3, "t2_w2": 5}
    return OptimizationProblem.model_validate(
        {
            "name": "two-task assignment",
            "variables": [{"name": name} for name in names],
            "objective": {
                "direction": "minimize",
                "linear_terms": [
                    {"variable": name, "coefficient": cost} for name, cost in costs.items()
                ],
                "quadratic_terms": [
                    {"variable1": f"t1_w{k}", "variable2": f"t2_w{k}", "coefficient": 10}
                    for k in range(3)
                ],
            },
            "constraints": [
                {
                    "id": f"t{t}_one_worker",
                    "type": "hard",
                    "terms": [{"variable": f"t{t}_w{k}", "coefficient": 1} for k in range(3)],
                    "operator": "==",
                    "rhs": 1,
                }
                for t in (1, 2)
            ],
        }
    )


def business_rows(problem: OptimizationProblem, rows: list[dict[str, int]]) -> RawSolverResult:
    variables = [variable.name for variable in problem.variables]
    return RawSolverResult(
        variables=variables,
        samples=np.array([[row.get(v, 0) for v in variables] for row in rows], dtype=np.int8),
        energies=np.zeros(len(rows)),
        backend="test",
    )


# Nobody assigned, t1 only, and a feasible but poor pair: two repairs and
# three local searches, several steps each.
POSTPROCESS_ROWS = [{}, {"t1_w2": 1}, {"t1_w2": 1, "t2_w2": 1}]


def postprocess(should_stop=None):
    problem = assignment_problem()
    return process_candidates(
        problem,
        business_rows(problem, POSTPROCESS_ROWS),
        frozenset(),
        5,
        PostprocessRequest(candidates=10, max_evaluations=10**6, should_stop=should_stop),
    )


def counting_stop(n: int):
    """True from the ``n``-th call on."""
    counter = itertools.count(1)
    return lambda: next(counter) >= n


class TestPostprocessStop:
    def test_a_stop_that_never_fires_changes_nothing(self):
        plain = postprocess()
        watched = postprocess(lambda: False)

        assert watched.postprocess == plain.postprocess
        assert watched.postprocess.wall_clock_limit_reached is False
        assert watched.solutions == plain.solutions

    def test_the_full_run_has_work_to_cut(self):
        stats = postprocess().postprocess
        assert stats.candidates_selected == 3
        assert stats.new_candidates >= 2

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_a_stop_after_n_checks_is_reported_apart_from_the_ceilings(self, n):
        full = postprocess().postprocess
        cut = postprocess(counting_stop(n))
        stats = cut.postprocess

        assert stats.wall_clock_limit_reached is True
        assert stats.limit_reached == []
        assert stats.candidates_selected <= full.candidates_selected
        assert stats.new_candidates <= full.new_candidates
        # Whatever it produced is still a re-validated feasible solution.
        assert all(solution.hard_constraints_satisfied for solution in cut.solutions)

    def test_a_stop_at_the_first_check_does_nothing_at_all(self):
        stats = postprocess(lambda: True).postprocess

        assert stats.wall_clock_limit_reached is True
        assert stats.candidates_selected == 0
        assert stats.new_candidates == 0
        assert stats.limit_reached == []

    def test_no_samples_leave_nothing_to_cut(self):
        problem = assignment_problem()
        processed = process_candidates(
            problem,
            business_rows(problem, []),
            frozenset(),
            5,
            PostprocessRequest(candidates=10, max_evaluations=10**6, should_stop=lambda: True),
        )

        assert processed.postprocess.wall_clock_limit_reached is False

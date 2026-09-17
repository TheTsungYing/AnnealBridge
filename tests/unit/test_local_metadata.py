"""The local backends report execution metadata too.

``exact``, ``simulated_annealing``, ``tabu`` and ``simulated_bifurcation``
have no vendor facts to relay, but a run still happened here: the metadata
names the backend, says the run did not leave this machine, and carries the
model type the service stamped on it. What a vendor would fill in — timing,
solver id, quota, embedding — is empty or null, and stays that way.

The service still never invents metadata for a solve that produced no
attempt; that contract is tested with fake backends elsewhere
(``test_service_compilers.py``, ``test_parameter_ceilings.py``).
"""

import pytest

from annealbridge.orchestration import OptimizationService
from tests.unit.test_solve_timing import _infeasible_pair, _knapsack

LOCAL_BACKENDS = ["exact", "simulated_annealing", "tabu", "simulated_bifurcation"]


def _assert_no_vendor_facts(metadata) -> None:
    """Nothing a remote sampler would report is present on a local run."""
    assert metadata.timing_us == {}
    assert metadata.solver_id is None
    assert metadata.effective_time_limit_seconds is None
    assert metadata.average_chain_break_fraction is None
    assert metadata.embedding_max_chain_length is None
    assert metadata.sampler_reported_feasible is None


class TestLocalBackendMetadata:
    @pytest.mark.parametrize("backend", LOCAL_BACKENDS)
    def test_success_carries_local_metadata(self, backend):
        result = OptimizationService().solve(_knapsack(backend))

        assert result.status == "success"
        assert result.metadata is not None
        assert result.metadata.backend == backend
        assert result.metadata.remote is False
        # Stamped by the service: both local backends compile to a BQM.
        assert result.metadata.model_type == "bqm"
        _assert_no_vendor_facts(result.metadata)

    def test_exact_reports_no_read_count(self):
        # ``exact`` enumerates the space once and takes no read count, so
        # the problem's ``num_reads`` is not echoed back as if it applied.
        result = OptimizationService().solve(_knapsack("exact", num_reads=40))

        assert result.metadata is not None
        assert result.metadata.num_reads_requested is None

    def test_sa_reports_the_requested_read_count(self):
        problem = _knapsack("simulated_annealing", num_reads=40, seed=7)
        result = OptimizationService().solve(problem)

        assert result.status == "success"
        assert result.metadata is not None
        assert result.metadata.num_reads_requested == problem.solver.num_reads == 40

    def test_sa_default_read_count_is_reported(self):
        problem = _knapsack("simulated_annealing")
        result = OptimizationService().solve(problem)

        assert result.metadata is not None
        assert result.metadata.num_reads_requested == problem.solver.num_reads

    @pytest.mark.parametrize("backend", LOCAL_BACKENDS)
    def test_infeasible_result_still_carries_metadata(self, backend):
        # Metadata describes the last completed attempt, not the verdict.
        result = OptimizationService().solve(_infeasible_pair(backend))

        assert result.status == "infeasible"
        assert result.solutions == []
        assert result.attempts
        assert result.metadata is not None
        assert result.metadata.backend == backend
        assert result.metadata.remote is False
        assert result.metadata.model_type == "bqm"
        _assert_no_vendor_facts(result.metadata)

    @pytest.mark.parametrize("backend", LOCAL_BACKENDS)
    def test_metadata_survives_serialisation(self, backend):
        result = OptimizationService().solve(_knapsack(backend))

        payload = result.model_dump(mode="json")
        assert payload["metadata"]["backend"] == backend
        assert payload["metadata"]["remote"] is False
        assert payload["metadata"]["model_type"] == "bqm"
        assert payload["metadata"]["timing_us"] == {}

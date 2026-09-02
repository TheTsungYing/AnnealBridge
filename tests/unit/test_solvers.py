"""Unit tests for the solver backends (spec §20–§22, §33)."""

from pathlib import Path

import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.models import CompiledProblem, OptimizationProblem, SolverPreferences
from annealbridge.solvers import (
    ExactSolverBackend,
    SimulatedAnnealingBackend,
    SolverCapabilities,
)

KNAPSACK_PATH = Path(__file__).resolve().parents[2] / "examples" / "knapsack.json"

# Knapsack (spec §30): capacity 10, items A(w6,v10) B(w5,v8) C(w4,v7) D(w3,v6).
# {A, C} has weight 6 + 4 = 10 (feasible) and value 10 + 7 = 17; every other
# feasible subset is worth less ({B,C}=15, {B,D}=14, {C,D}=13, ...), so the
# global optimum objective value is 17.
KNAPSACK_OPTIMUM_VALUE = 17.0
KNAPSACK_OPTIMUM_SELECTION = {"item_a": 1, "item_b": 0, "item_c": 1, "item_d": 0}

# Well above the objective range (max 31) so violating the hard constraint
# always costs more than any objective gain.
HARD_PENALTY = 100.0


def load_knapsack() -> OptimizationProblem:
    return OptimizationProblem.model_validate_json(KNAPSACK_PATH.read_text())


def compile_knapsack() -> CompiledProblem:
    return BQMCompiler().compile(load_knapsack(), hard_penalty=HARD_PENALTY)


class TestKnapsackExample:
    def test_example_file_parses(self):
        problem = load_knapsack()
        assert problem.name == "knapsack"
        assert [v.name for v in problem.variables] == [
            "item_a",
            "item_b",
            "item_c",
            "item_d",
        ]
        assert problem.solver.backend == "exact"


class TestExactSolverBackend:
    def test_properties(self):
        backend = ExactSolverBackend()
        assert backend.name == "exact"
        assert backend.is_exhaustive is True

    def test_capabilities(self):
        capabilities = ExactSolverBackend().capabilities
        assert isinstance(capabilities, SolverCapabilities)
        assert capabilities.name == "exact"
        assert capabilities.remote is False
        assert capabilities.heuristic is False
        assert capabilities.exhaustive is True
        assert capabilities.supports_seed is False
        assert capabilities.supports_num_reads is False
        assert capabilities.supports_time_limit is False
        assert capabilities.supported_model_types == ["bqm"]
        assert capabilities.returns_multiple_samples is True
        assert capabilities.description.strip()

    def test_is_available(self):
        assert ExactSolverBackend().is_available() == (True, None)

    def test_properties_alias_capabilities(self):
        backend = ExactSolverBackend()
        assert backend.name == backend.capabilities.name
        assert backend.is_exhaustive == backend.capabilities.exhaustive

    def test_resolve_time_limit_is_none(self):
        # A local backend never submits a time limit (Phase 2 spec §10).
        compiled = compile_knapsack()
        assert ExactSolverBackend().resolve_time_limit(compiled, SolverPreferences()) is None

    def test_finds_known_optimum(self):
        compiled = compile_knapsack()
        result = ExactSolverBackend().solve(compiled, SolverPreferences())

        best_index = min(range(len(result.energies)), key=result.energies.__getitem__)
        best_sample = result.samples[best_index]
        business = {
            name: value
            for name, value in best_sample.items()
            if name not in compiled.internal_variables
        }
        assert business == KNAPSACK_OPTIMUM_SELECTION
        # Maximization compiles with sign -1 and the optimum is feasible
        # (zero penalty), so energy == -objective.
        assert result.energies[best_index] == pytest.approx(-KNAPSACK_OPTIMUM_VALUE)

    def test_returns_all_samples(self):
        compiled = compile_knapsack()
        result = ExactSolverBackend().solve(compiled, SolverPreferences())

        assert result.backend == "exact"
        assert len(result.samples) == 2**compiled.num_variables
        assert len(result.energies) == len(result.samples)
        assert result.samples

    def test_samples_include_internal_variables(self):
        compiled = compile_knapsack()
        result = ExactSolverBackend().solve(compiled, SolverPreferences())

        assert compiled.internal_variables
        assert compiled.internal_variables <= set(result.samples[0])


class TestSimulatedAnnealingBackend:
    def test_properties(self):
        backend = SimulatedAnnealingBackend()
        assert backend.name == "simulated_annealing"
        assert backend.is_exhaustive is False

    def test_capabilities(self):
        capabilities = SimulatedAnnealingBackend().capabilities
        assert isinstance(capabilities, SolverCapabilities)
        assert capabilities.name == "simulated_annealing"
        assert capabilities.remote is False
        assert capabilities.heuristic is True
        assert capabilities.exhaustive is False
        assert capabilities.supports_seed is True
        assert capabilities.supports_num_reads is True
        assert capabilities.supports_time_limit is False
        assert capabilities.supported_model_types == ["bqm"]
        assert capabilities.returns_multiple_samples is True
        assert capabilities.description.strip()

    def test_is_available(self):
        assert SimulatedAnnealingBackend().is_available() == (True, None)

    def test_properties_alias_capabilities(self):
        backend = SimulatedAnnealingBackend()
        assert backend.name == backend.capabilities.name
        assert backend.is_exhaustive == backend.capabilities.exhaustive

    def test_resolve_time_limit_is_none(self):
        compiled = compile_knapsack()
        assert (
            SimulatedAnnealingBackend().resolve_time_limit(compiled, SolverPreferences())
            is None
        )

    def test_same_seed_is_reproducible(self):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=20, num_sweeps=100, seed=42)
        backend = SimulatedAnnealingBackend()

        first = backend.solve(compiled, preferences)
        second = backend.solve(compiled, preferences)

        assert first.samples == second.samples
        assert first.energies == second.energies

    def test_returns_all_reads(self):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=20, num_sweeps=100, seed=7)
        result = SimulatedAnnealingBackend().solve(compiled, preferences)

        assert result.backend == "simulated_annealing"
        assert len(result.samples) == preferences.num_reads
        assert len(result.energies) == preferences.num_reads
        assert result.samples

    def test_none_seed_still_solves(self):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=5, num_sweeps=50, seed=None)
        result = SimulatedAnnealingBackend().solve(compiled, preferences)

        assert len(result.samples) == preferences.num_reads

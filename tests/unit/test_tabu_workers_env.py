"""``ANNEALBRIDGE_TABU_WORKERS`` end to end.

The worker count is one of the settings that is not policy: it tunes how
fast the ``tabu`` backend samples and never what it returns, so it travels
environment → ``ServerSettings`` → composition root →
``SolverRegistry.default(tabu_workers=...)`` → the backend, bypassing
``ExecutionPolicy``. This pins that path and the "same result for any
worker count" contract through the service.
"""

import json
import os

import pytest

from annealbridge.interfaces.composition import build_state
from annealbridge.models import OptimizationProblem
from tests.conftest import EXAMPLES_DIR

KNAPSACK = EXAMPLES_DIR / "knapsack.json"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in list(os.environ):
        if name.upper().startswith("ANNEALBRIDGE_"):
            monkeypatch.delenv(name, raising=False)
    return monkeypatch


def seeded_knapsack(num_reads: int) -> OptimizationProblem:
    data = json.loads(KNAPSACK.read_text(encoding="utf-8"))
    data["solver"] = {
        **data.get("solver", {}),
        "backend": "tabu",
        "seed": 1,
        "num_reads": num_reads,
    }
    return OptimizationProblem.model_validate(data)


def test_env_reaches_the_backend(clean_env):
    clean_env.setenv("ANNEALBRIDGE_TABU_WORKERS", "2")

    state = build_state()

    assert state.registry.get("tabu").workers == 2


def test_unset_detects_at_least_one_worker(clean_env):
    state = build_state()

    assert state.registry.get("tabu").workers >= 1


def test_worker_count_changes_nothing_but_speed(clean_env):
    # 200 reads = 8 shards; one worker and eight must agree on every field
    # the caller can see, ranked solutions and attempt counters included.
    clean_env.setenv("ANNEALBRIDGE_TABU_WORKERS", "1")
    serial = build_state().service.solve(seeded_knapsack(200))
    clean_env.setenv("ANNEALBRIDGE_TABU_WORKERS", "8")
    parallel = build_state().service.solve(seeded_knapsack(200))

    assert serial.status == "success"
    # The service's own wall clock is the one thing allowed to differ.
    clock = {
        "elapsed_ms": True,
        "attempts": {"__all__": {"compile_ms", "solve_ms", "validate_ms"}},
    }
    assert serial.model_dump(exclude=clock) == parallel.model_dump(exclude=clock)
    for result in (serial, parallel):
        assert result.elapsed_ms >= 0
        for attempt in result.attempts:
            assert attempt.compile_ms >= 0
            assert attempt.solve_ms >= 0
            assert attempt.validate_ms >= 0

"""``ANNEALBRIDGE_SA_WORKERS`` end to end.

The worker count is the one setting that is not policy: it tunes how fast
the ``simulated_annealing`` backend samples and never what it returns, so
it travels environment → ``ServerSettings`` → composition root →
``SolverRegistry.default(sa_workers=...)`` → the backend, bypassing
``ExecutionPolicy``. This pins that path and the "same result for any
worker count" contract through the service.
"""

import json
import os
from pathlib import Path

import pytest

from annealbridge.interfaces.composition import build_state
from annealbridge.models import OptimizationProblem

KNAPSACK = Path(__file__).resolve().parents[2] / "examples" / "knapsack.json"


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
        "backend": "simulated_annealing",
        "seed": 1,
        "num_reads": num_reads,
    }
    return OptimizationProblem.model_validate(data)


def test_env_reaches_the_backend(clean_env):
    clean_env.setenv("ANNEALBRIDGE_SA_WORKERS", "2")

    state = build_state()

    assert state.registry.get("simulated_annealing").workers == 2


def test_unset_detects_at_least_one_worker(clean_env):
    state = build_state()

    assert state.registry.get("simulated_annealing").workers >= 1


def test_worker_count_changes_nothing_but_speed(clean_env):
    # 200 reads = 8 shards; one worker and eight must agree on every field
    # the caller can see, ranked solutions and attempt counters included.
    clean_env.setenv("ANNEALBRIDGE_SA_WORKERS", "1")
    serial = build_state().service.solve(seeded_knapsack(200))
    clean_env.setenv("ANNEALBRIDGE_SA_WORKERS", "8")
    parallel = build_state().service.solve(seeded_knapsack(200))

    assert serial.status == "success"
    assert serial.model_dump() == parallel.model_dump()

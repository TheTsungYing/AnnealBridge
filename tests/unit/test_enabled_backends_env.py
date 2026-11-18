"""``ANNEALBRIDGE_ENABLED_BACKENDS`` end to end (2026-09-09 review F-18).

``ExecutionPolicy.enabled_backends`` and its ``BACKEND_DISABLED_BY_POLICY``
gate existed since Phase 2, but no environment variable ever filled the
field, so a CLI / MCP deployment could not reach the gate. This pins the
whole path: environment → ``ServerSettings`` → policy → composition root →
``service.solve`` → the catalog error, plus the capabilities view an agent
reads before choosing a backend.
"""

import json
import logging
import os

import pytest

from annealbridge.interfaces.capabilities import build_capabilities
from annealbridge.interfaces.composition import build_state
from annealbridge.models import OptimizationProblem
from tests.conftest import EXAMPLES_DIR

KNAPSACK = EXAMPLES_DIR / "knapsack.json"


@pytest.fixture
def only_exact_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.upper().startswith("ANNEALBRIDGE_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANNEALBRIDGE_ENABLED_BACKENDS", "exact")


def knapsack(backend: str) -> OptimizationProblem:
    data = json.loads(KNAPSACK.read_text(encoding="utf-8"))
    data["solver"] = {**data.get("solver", {}), "backend": backend}
    return OptimizationProblem.model_validate(data)


def test_env_reaches_the_policy(only_exact_enabled):
    state = build_state()

    assert state.policy.enabled_backends == {"exact"}


def test_backend_outside_the_list_is_refused_by_policy(only_exact_enabled):
    state = build_state()

    result = state.service.solve(knapsack("simulated_annealing"))

    assert result.status == "backend_unavailable"
    assert [error.code for error in result.errors] == ["BACKEND_DISABLED_BY_POLICY"]
    assert "simulated_annealing" in result.errors[0].message
    assert result.solutions == []


def test_backend_inside_the_list_still_solves(only_exact_enabled):
    state = build_state()

    result = state.service.solve(knapsack("exact"))

    assert result.status == "success"


def test_capabilities_view_reports_the_gate(only_exact_enabled):
    state = build_state()

    enabled = {
        backend.name: backend.enabled
        for backend in build_capabilities(state.registry, state.policy).backends
    }

    assert enabled["exact"] is True
    assert enabled["simulated_annealing"] is False


COMPOSITION_LOGGER = "annealbridge.interfaces.composition"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.upper().startswith("ANNEALBRIDGE_"):
            monkeypatch.delenv(name, raising=False)


def composition_warnings(caplog: pytest.LogCaptureFixture) -> list:
    return [
        record
        for record in caplog.records
        if record.name == COMPOSITION_LOGGER and record.levelno == logging.WARNING
    ]


def test_unknown_name_is_warned_and_kept(clean_env, monkeypatch, caplog):
    monkeypatch.setenv("ANNEALBRIDGE_ENABLED_BACKENDS", "exact,simulated_anealing")
    caplog.set_level(logging.WARNING, logger=COMPOSITION_LOGGER)

    state = build_state()

    warnings = composition_warnings(caplog)
    assert len(warnings) == 1
    assert "simulated_anealing" in warnings[0].getMessage()
    assert state.policy.enabled_backends == {"exact", "simulated_anealing"}


def test_known_names_are_not_warned(clean_env, monkeypatch, caplog):
    monkeypatch.setenv("ANNEALBRIDGE_ENABLED_BACKENDS", "exact,simulated_annealing")
    caplog.set_level(logging.WARNING, logger=COMPOSITION_LOGGER)

    build_state()

    assert composition_warnings(caplog) == []


def test_unset_allow_list_is_not_warned(clean_env, caplog):
    caplog.set_level(logging.WARNING, logger=COMPOSITION_LOGGER)

    state = build_state()

    assert state.policy.enabled_backends is None
    assert composition_warnings(caplog) == []

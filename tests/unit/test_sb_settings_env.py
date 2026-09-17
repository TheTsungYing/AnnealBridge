"""``ANNEALBRIDGE_SB_DEVICE`` and ``ANNEALBRIDGE_SB_MAX_VARIABLES`` end to end.

Neither is policy: one says where the ``simulated_bifurcation`` backend runs
its dynamics, the other sizes the dense ``N x N`` coupling matrix it is
willing to allocate. Both travel environment → ``ServerSettings`` →
composition root → ``SolverRegistry.default(sb_device=..., sb_max_variables=...)``
→ the backend, bypassing ``ExecutionPolicy``. This pins that path and the
rejection of values outside each field's declared domain.

Nothing here asks whether a CUDA device exists: the setting is carried as
written, and whether the device is usable is what ``is_available()`` answers
(covered with the backend's own tests).
"""

import os

import pytest
from pydantic import ValidationError

from annealbridge.config.settings import ServerSettings
from annealbridge.interfaces.composition import build_state


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in list(os.environ):
        if name.upper().startswith("ANNEALBRIDGE_"):
            monkeypatch.delenv(name, raising=False)
    return monkeypatch


def sb_backend():
    return build_state().registry.get("simulated_bifurcation")


class TestEnvReachesTheBackend:
    def test_max_variables_reaches_the_backend(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_SB_MAX_VARIABLES", "123")

        assert sb_backend().max_variables == 123

    def test_device_reaches_the_backend(self, clean_env):
        # The attribute only: constructing the backend must not probe the
        # driver, so a machine without CUDA still records the setting.
        clean_env.setenv("ANNEALBRIDGE_SB_DEVICE", "cuda")

        assert sb_backend().device == "cuda"

    def test_unset_leaves_the_declared_defaults(self, clean_env):
        backend = sb_backend()

        assert backend.device == "cpu"
        assert backend.max_variables == 10_000


class TestInvalidValuesAreRefused:
    """A value outside the domain is refused at the environment boundary,
    never clamped to the default and never carried into the registry."""

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_non_positive_max_variables_is_rejected(self, clean_env, value):
        clean_env.setenv("ANNEALBRIDGE_SB_MAX_VARIABLES", value)

        with pytest.raises(ValidationError) as exc_info:
            ServerSettings()

        assert [error["loc"] for error in exc_info.value.errors()] == [
            ("sb_max_variables",)
        ]

    def test_smallest_positive_max_variables_is_accepted(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_SB_MAX_VARIABLES", "1")

        assert sb_backend().max_variables == 1

    @pytest.mark.parametrize("value", ["tpu", "cuda:0", "opencl"])
    def test_unknown_device_is_rejected(self, clean_env, value):
        clean_env.setenv("ANNEALBRIDGE_SB_DEVICE", value)

        with pytest.raises(ValidationError) as exc_info:
            ServerSettings()

        assert [error["loc"] for error in exc_info.value.errors()] == [("sb_device",)]

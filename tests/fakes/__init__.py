"""Test-only fakes (backends, samplers); never shipped, never production.

Kept out of ``tests/unit`` so the fakes can be shared across the unit,
architecture and scenario suites without duplication. Module names do not
match ``test_*.py`` so pytest never collects them as tests.
"""

from tests.fakes.declared_backend import FAKE_DECLARED_NAME, FakeDeclaredBackend
from tests.fakes.local_cqm_backend import (
    FAKE_LOCAL_CQM_NAME,
    FakeLocalCQMBackend,
    all_feasible_sampleset,
)

__all__ = [
    "FAKE_DECLARED_NAME",
    "FAKE_LOCAL_CQM_NAME",
    "FakeDeclaredBackend",
    "FakeLocalCQMBackend",
    "all_feasible_sampleset",
]

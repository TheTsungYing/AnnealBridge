"""Shared fakes and fixtures for the remote-backend mock tests.

Nothing here talks to real D-Wave. The backends expose a
``sampler_factory`` seam, so every test in this package injects one of the
fakes below and gets a real ``dimod.SampleSet`` back.

The fake Ocean exception classes exist because the backends classify
Ocean failures *by class name* (``dwave-cloud-client`` and ``dwave-system``
are not installed in mock CI), so the fakes only have to carry the real
names.
"""

import dimod

from annealbridge.models import AvailabilityStatus, OptimizationProblem

# Matches the DEV-[A-Za-z0-9]{20,} redaction pattern; never a real token.
FAKE_TOKEN = "DEV-FAKETOKEN1234567890abcdefghij"

# Hyphenated on purpose: NOT matched by the DEV- pattern, so masking has to
# come from the live environment-variable candidate. Never a real token.
FAKE_UNPATTERNED_TOKEN = "DEV-FAKE-TOKEN-1234567890abcdefghij"

FAKE_MIN_TIME_LIMIT = 3.0

# Nested QPU timing (whitelist keys) plus dirty keys that sanitization must
# drop. The embedding context is NOT part of this dict: EmbeddingComposite
# only populates it when return_embedding=True is requested, so the fake
# injects it separately (see FakeQPUSampler.embedding_context).
FAKE_QPU_SAMPLESET_INFO = {
    "timing": {
        "qpu_access_time": 12345,
        "qpu_sampling_time": 6789,
        "qpu_anneal_time_per_sample": 20,
    },
    "problem_id": "fake-problem-id-456",
    "messages": [{"nested": "structure"}],
    "raw_blob": b"\x00\x01\x02",
    "unexpected": {"deep": ("tuple", object())},
}

# Whitelisted timing keys only; used where the point of the test is the
# service flow rather than sanitization.
FAKE_MINIMAL_SAMPLESET_INFO = {"timing": {"qpu_access_time": 12345}}

# Hybrid solvers report timing at the top level. Whitelist keys plus dirty
# keys that sanitization must drop.
FAKE_HYBRID_SAMPLESET_INFO = {
    "run_time": 2900000,
    "charge_time": 2871000,
    "qpu_access_time": 12345,
    "problem_id": "fake-problem-id-123",
    "messages": [{"nested": "structure"}],
    "raw_blob": b"\x00\x01\x02",
    "unexpected": {"deep": ("tuple", object())},
}

# Longest chain is "a" (3 qubits) → embedding_max_chain_length == 3.
FAKE_EMBEDDING_CONTEXT = {"embedding": {"a": (0, 1, 2), "b": (3,), "slack": (4, 5)}}


# The backends classify Ocean exceptions by class name (dwave-cloud-client
# and dwave-system are not installed in mock CI), so the fakes carry the
# real names.
class SolverAuthenticationError(Exception):
    """Fake of dwave.cloud's SolverAuthenticationError."""


class RequestTimeout(Exception):
    """Fake of dwave.cloud's RequestTimeout (classified as REMOTE_TIMEOUT)."""


class EmbeddingError(Exception):
    """Fake of dwave.embedding's EmbeddingError."""


class SolverFailureError(Exception):
    """Fake of dwave.cloud's SolverFailureError (→ REMOTE_SOLVER_ERROR)."""


class SolverNotFoundError(Exception):
    """Fake of dwave.cloud's SolverNotFoundError."""


class ConfigFileError(Exception):
    """Fake of dwave.cloud's ConfigFileError."""


class ValidationError(ValueError):
    """Fake of pydantic's ValidationError, which subclasses ValueError."""


def lazy_sampleset(build, failure: Exception | None) -> dimod.SampleSet:
    """Mimic Ocean: the cloud round-trip only happens on first attribute access.

    ``build()`` produces the real sampleset; ``failure`` (if any) is raised
    at resolve time — exactly where RequestTimeout / SolverFailureError
    surface with a real ``DWaveSampler`` / ``LeapHybridSampler``.
    """

    def hook(future):
        if failure is not None:
            raise failure
        return build()

    return dimod.SampleSet.from_future(object(), hook)


def expand(bqm, assignment: dict[str, int]) -> dict:
    """Widen a business assignment over every compiled variable (slack = 0)."""
    return {
        variable: int(assignment.get(str(variable), 0)) for variable in bqm.variables
    }


class FakeQPUSampler:
    """Fake with the EmbeddingComposite surface the QPU backend touches.

    ``assignments`` are business assignments widened over the compiled BQM,
    so a test can decide whether the service sees feasible samples or none
    at all; ``None`` keeps the default two rows (all-zero, then the first
    compiled variable flipped on).

    ``lazy=True`` reproduces Ocean's real shape: ``sample()`` returns
    immediately with a ``SampleSet.from_future`` whose hook only runs (and
    only fails) when the sampleset is resolved.
    """

    def __init__(
        self,
        assignments: list[dict[str, int]] | None = None,
        raise_on_sample: Exception | None = None,
        include_chain_break_fraction: bool = True,
        info: dict | None = None,
        lazy: bool = False,
        embedding_context: dict | str | None = FAKE_EMBEDDING_CONTEXT,
    ) -> None:
        self.assignments = assignments
        self.raise_on_sample = raise_on_sample
        self.include_chain_break_fraction = include_chain_break_fraction
        self.info = FAKE_QPU_SAMPLESET_INFO if info is None else info
        self.lazy = lazy
        self.embedding_context = embedding_context
        self.sample_bqm = None
        self.sample_kwargs = None
        self.sample_calls = 0

    def _rows(self, bqm) -> list[dict]:
        if self.assignments is not None:
            return [expand(bqm, assignment) for assignment in self.assignments]
        variables = list(bqm.variables)
        return [
            {variable: 0 for variable in variables},
            {**{variable: 0 for variable in variables}, variables[0]: 1},
        ]

    def _build_sampleset(self, bqm, kwargs: dict) -> dimod.SampleSet:
        if self.raise_on_sample is not None:
            raise self.raise_on_sample
        rows = self._rows(bqm)
        vectors = {}
        if self.include_chain_break_fraction:
            vectors["chain_break_fraction"] = [
                0.25 if index % 2 else 0.0 for index in range(len(rows))
            ]
        info = dict(self.info)
        # EmbeddingComposite only reports the embedding when asked to.
        if self.embedding_context is not None and kwargs.get("return_embedding"):
            info["embedding_context"] = self.embedding_context
        return dimod.SampleSet.from_samples(
            rows,
            vartype=dimod.BINARY,
            energy=[bqm.energy(row) for row in rows],
            info=info,
            **vectors,
        )

    def sample(self, bqm, **kwargs) -> dimod.SampleSet:
        self.sample_bqm = bqm
        self.sample_kwargs = kwargs
        self.sample_calls += 1
        if self.lazy:
            return lazy_sampleset(
                lambda: self._build_sampleset(bqm, kwargs), self.raise_on_sample
            )
        return self._build_sampleset(bqm, kwargs)


class FakeLeapHybridSampler:
    """Fake with the LeapHybridSampler surface the hybrid backend touches.

    ``assignments`` behaves as in :class:`FakeQPUSampler`; ``None`` keeps
    the default single all-zero sample (hybrid solvers typically return
    exactly one).

    ``lazy=True`` reproduces Ocean's real shape: ``sample()`` returns
    immediately with a ``SampleSet.from_future`` whose hook only runs (and
    only fails) when the sampleset is resolved.
    """

    def __init__(
        self,
        assignments: list[dict[str, int]] | None = None,
        min_time_limit: float = FAKE_MIN_TIME_LIMIT,
        raise_on_sample: Exception | None = None,
        raise_on_min_time_limit: Exception | None = None,
        lazy: bool = False,
    ) -> None:
        self.assignments = assignments
        self._min_time_limit = min_time_limit
        self.raise_on_sample = raise_on_sample
        self.raise_on_min_time_limit = raise_on_min_time_limit
        self.lazy = lazy
        self.min_time_limit_bqm = None
        self.min_time_limit_calls = 0
        self.sample_bqm = None
        self.sample_kwargs = None
        self.sample_calls = 0

    def min_time_limit(self, bqm) -> float:
        self.min_time_limit_bqm = bqm
        self.min_time_limit_calls += 1
        if self.raise_on_min_time_limit is not None:
            raise self.raise_on_min_time_limit
        return self._min_time_limit

    def _rows(self, bqm) -> list[dict]:
        if self.assignments is not None:
            return [expand(bqm, assignment) for assignment in self.assignments]
        return [{variable: 0 for variable in bqm.variables}]

    def _build_sampleset(self, bqm) -> dimod.SampleSet:
        if self.raise_on_sample is not None:
            raise self.raise_on_sample
        rows = self._rows(bqm)
        return dimod.SampleSet.from_samples(
            rows,
            vartype=dimod.BINARY,
            energy=[bqm.energy(row) for row in rows],
            info=dict(FAKE_HYBRID_SAMPLESET_INFO),
        )

    def sample(self, bqm, **kwargs) -> dimod.SampleSet:
        self.sample_bqm = bqm
        self.sample_kwargs = kwargs
        self.sample_calls += 1
        if self.lazy:
            return lazy_sampleset(
                lambda: self._build_sampleset(bqm), self.raise_on_sample
            )
        return self._build_sampleset(bqm)


class CountingFactory:
    """``sampler_factory`` seam that counts constructions and can fail on demand.

    Each entry in ``failures`` is raised by one call, in order; every later
    call returns ``sampler``.
    """

    def __init__(self, sampler, failures: list[Exception] | None = None) -> None:
        self.sampler = sampler
        self.failures = list(failures or [])
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.sampler


def make_remote_available(monkeypatch, module) -> None:
    """Force ``module``'s backend to report itself installed and configured."""
    monkeypatch.setattr(
        module, "dwave_availability", lambda: AvailabilityStatus(category="available")
    )


def make_problem(backend: str = "dwave_qpu", **solver_overrides) -> OptimizationProblem:
    """maximize 2a + b s.t. a + b <= 1; the ``<=`` adds an internal slack var."""
    return OptimizationProblem.model_validate(
        {
            "name": "remote mock problem",
            "variables": [{"name": "a"}, {"name": "b"}],
            "objective": {
                "direction": "maximize",
                "linear_terms": [
                    {"variable": "a", "coefficient": 2},
                    {"variable": "b", "coefficient": 1},
                ],
            },
            "constraints": [
                {
                    "id": "at_most_one",
                    "type": "hard",
                    "terms": [
                        {"variable": "a", "coefficient": 1},
                        {"variable": "b", "coefficient": 1},
                    ],
                    "operator": "<=",
                    "rhs": 1,
                }
            ],
            "solver": {"backend": backend, **solver_overrides},
        }
    )

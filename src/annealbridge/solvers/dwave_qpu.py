"""D-Wave QPU solver backend (Phase 2 spec §15).

``dwave.system`` is imported lazily inside the default sampler factory so
this module — and the default registry that instantiates the backend —
imports cleanly without the ``dwave`` extra installed (spec §4).
"""

import logging
import threading
from typing import Any, Callable, TypeVar

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import (
    CompiledProblem,
    DWaveQPUOptions,
    SolverPreferences,
)
from annealbridge.solvers.base import (
    AvailabilityStatus,
    RawSolverResult,
    SolverCapabilities,
    sampleset_to_arrays,
)
from annealbridge.solvers.metadata import (
    classify_exception,
    dwave_availability,
    redact,
    sanitize_sampleset_info,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_CAPABILITIES = SolverCapabilities(
    name="dwave_qpu",
    remote=True,
    heuristic=True,
    exhaustive=False,
    supports_seed=False,
    supports_num_reads=True,
    supports_time_limit=False,
    supported_model_types=["bqm"],
    returns_multiple_samples=True,
    requires_embedding=True,
    description=(
        "D-Wave quantum annealer accessed through "
        "EmbeddingComposite(DWaveSampler()). Minor-embedding and chain-break "
        "resolution (Ocean's default majority_vote) are handled by Ocean; "
        "samples are returned in logical variables."
    ),
)

# Ocean exception class names → catalog error codes, matched by name across
# the exception's MRO so classification works (and is testable with fakes)
# without dwave-cloud-client installed. The MRO walk starts at the
# most-derived class, so a named Ocean exception that happens to subclass
# ValueError still wins over the ValueError entry.
#
# Two tables because the *same* exception type means different things
# depending on the stage:
#
# - Sampling stage (spec §15): EmbeddingComposite raises a bare ValueError
#   when no embedding is found → EMBEDDING_FAILED.
# - Sampler construction: DWaveSampler() runs Client.from_config() and
#   get_solver(); a ValueError there (including pydantic's ValidationError,
#   a ValueError subclass) means an invalid region / endpoint / profile /
#   timeout / solver selection — a configuration problem, never an
#   embedding one. Telling the agent to "shrink the problem" would be wrong.
_SAMPLE_EXCEPTION_CODES = {
    "EmbeddingError": "EMBEDDING_FAILED",
    "ValueError": "EMBEDDING_FAILED",
    "SolverAuthenticationError": "REMOTE_AUTH_FAILED",
    "RequestTimeout": "REMOTE_TIMEOUT",
}
_SAMPLER_INIT_EXCEPTION_CODES = {
    "SolverAuthenticationError": "REMOTE_AUTH_FAILED",
    "RequestTimeout": "REMOTE_TIMEOUT",
    "SolverNotFoundError": "DWAVE_CONFIG_INVALID",
    "ConfigFileError": "DWAVE_CONFIG_INVALID",
    "ValueError": "DWAVE_CONFIG_INVALID",
}


def _default_sampler_factory() -> Any:
    """Create a real embedded QPU sampler (requires the ``dwave`` extra)."""
    from dwave.system import DWaveSampler, EmbeddingComposite

    return EmbeddingComposite(DWaveSampler())


def _call_ocean(what: str, codes: dict[str, str], fn: Callable[[], _T]) -> _T:
    """Run ``fn`` and convert any failure into a redacted SolverExecutionError.

    The wrapped error is raised *after* the ``except`` block has finished,
    so it carries neither ``__cause__`` nor ``__context__``: the original
    exception (whose text may embed credentials) is not reachable from the
    error that leaves the solver layer, and ``traceback.format_exception``
    / ``logger.exception`` cannot print it (spec §19). The original class
    name is kept in the message because it is categorical, not secret.
    """
    try:
        return fn()
    except Exception as exc:
        error = SolverExecutionError(
            redact(f"{what}: {type(exc).__name__}: {exc}"),
            code=classify_exception(exc, codes),
        )
    raise error


def _resolved(sampleset: Any) -> Any:
    """Force a lazy (``SampleSet.from_future``) sampleset to resolve now.

    Ocean samplers return samplesets whose cloud request only completes on
    first attribute access; resolving inside the guarded call is what
    routes RequestTimeout / SolverFailureError through classification and
    redaction instead of letting them escape later, unclassified.
    """
    sampleset.resolve()
    return sampleset


def _average_chain_break_fraction(sampleset: Any) -> float | None:
    """Mean of ``record.chain_break_fraction`` when safely available.

    Samplesets built without that vector have no such record field at all
    (attribute access would raise), so presence is checked via
    ``record.dtype.names``.
    """
    try:
        record = sampleset.record
        if "chain_break_fraction" not in (record.dtype.names or ()):
            return None
        values = record.chain_break_fraction
        if len(values) == 0:
            return None
        return float(values.mean())
    except Exception:
        return None


def _embedding_max_chain_length(info: dict) -> int | None:
    """Max chain length from ``info["embedding_context"]``, if well-formed.

    Any malformed shape yields None as a whole — no partial extraction from
    data that cannot be trusted.
    """
    context = info.get("embedding_context")
    if not isinstance(context, dict):
        return None
    embedding = context.get("embedding")
    if not isinstance(embedding, dict) or not embedding:
        return None
    lengths: list[int] = []
    for chain in embedding.values():
        try:
            lengths.append(len(chain))
        except TypeError:
            return None
    return max(lengths)


class DWaveQPUBackend:
    """Solves the compiled BQM directly on a D-Wave QPU.

    ``sampler_factory`` is the single test seam: production uses the default
    (``EmbeddingComposite(DWaveSampler())``), tests inject a fake. The
    sampler is built lazily on first use and cached for the lifetime of the
    backend instance — ``DWaveSampler()`` fetches the whole working graph
    and starts worker threads, so rebuilding it per solve (or per retry)
    would be wasteful; a failed construction is never cached.

    Chain strength is the user-provided option or Ocean's default — never
    derived from the compiled hard penalty (spec §15.1). Chain breaks are
    resolved by Ocean's default ``majority_vote``; no chain-break kwarg is
    forwarded.
    """

    def __init__(self, sampler_factory: Callable[[], Any] | None = None) -> None:
        self._sampler_factory = sampler_factory
        self._sampler: Any | None = None
        self._sampler_lock = threading.Lock()

    @property
    def capabilities(self) -> SolverCapabilities:
        return _CAPABILITIES

    def is_available(self) -> AvailabilityStatus:
        """Installability and credentials, via the shared check. No network I/O."""
        return dwave_availability()

    @property
    def name(self) -> str:
        """Alias for ``capabilities.name``."""
        return self.capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        """Alias for ``capabilities.exhaustive``."""
        return self.capabilities.exhaustive

    def resolve_time_limit(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> float | None:
        """The QPU takes no time limit: always None."""
        return None

    def _get_sampler(self) -> Any:
        """Return the cached sampler, building it on first use.

        Raw factory exceptions propagate; callers wrap them. Only a
        successful construction is cached, so a transient failure does not
        poison the backend.
        """
        with self._sampler_lock:
            if self._sampler is None:
                factory = self._sampler_factory or _default_sampler_factory
                self._sampler = factory()
            return self._sampler

    def solve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> RawSolverResult:
        """Submit the BQM to the QPU and return all reads as logical samples.

        Forwarded parameters (spec §15): ``num_reads``, ``annealing_time``
        (µs, only when the user set one), ``chain_strength`` (omitted when
        None so Ocean's default applies) and ``auto_scale``. ``seed`` /
        ``num_sweeps`` are meaningless for the QPU and never forwarded.
        ``return_embedding=True`` is always passed: EmbeddingComposite only
        populates ``info["embedding_context"]`` (the source of
        ``embedding_max_chain_length``) when asked to. Policy ceilings are
        enforced by the service layer, not here.
        """
        bqm = compiled_problem.model
        options = preferences.dwave_qpu or DWaveQPUOptions()

        # Assembled outside any guarded call: only Ocean exceptions may
        # reach the classifier (which maps bare ValueError to
        # EMBEDDING_FAILED at the sampling stage).
        sample_kwargs: dict[str, Any] = {
            "num_reads": preferences.num_reads,
            "auto_scale": options.auto_scale,
            "return_embedding": True,
        }
        if options.annealing_time_us is not None:
            sample_kwargs["annealing_time"] = options.annealing_time_us
        if options.chain_strength is not None:
            sample_kwargs["chain_strength"] = options.chain_strength

        sampler = _call_ocean(
            "D-Wave QPU sampler could not be created",
            _SAMPLER_INIT_EXCEPTION_CODES,
            self._get_sampler,
        )
        sampleset = _call_ocean(
            "D-Wave QPU solve failed",
            _SAMPLE_EXCEPTION_CODES,
            lambda: _resolved(sampler.sample(bqm, **sample_kwargs)),
        )

        variables, samples, energies = sampleset_to_arrays(sampleset)
        metadata = sanitize_sampleset_info(sampleset.info, backend=self.name)
        metadata.num_reads_requested = preferences.num_reads
        metadata.average_chain_break_fraction = _average_chain_break_fraction(
            sampleset
        )
        metadata.embedding_max_chain_length = _embedding_max_chain_length(
            sampleset.info
        )
        logger.info(
            "Backend %s solved problem %s: %d variables, %d samples "
            "(num_reads=%d)",
            self.name,
            compiled_problem.original_problem.name,
            compiled_problem.num_variables,
            len(samples),
            preferences.num_reads,
        )
        return RawSolverResult(
            variables=variables,
            samples=samples,
            energies=energies,
            backend=self.name,
            metadata=metadata,
        )

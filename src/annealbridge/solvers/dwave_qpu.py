"""D-Wave QPU solver backend (Phase 2 spec §15).

``dwave.system`` is imported lazily inside the default sampler factory so
this module — and the default registry that instantiates the backend —
imports cleanly without the ``dwave`` extra installed (spec §4).
"""

import logging
from typing import Any, Callable

from annealbridge.models import (
    CompiledProblem,
    DWaveQPUOptions,
    SolverPreferences,
)
from annealbridge.solvers.base import (
    AvailabilityStatus,
    BackendAliases,
    ParameterLimit,
    RawSolverResult,
    SolverCapabilities,
    sampleset_to_arrays,
)
from annealbridge.solvers.metadata import sanitize_sampleset_info
from annealbridge.solvers.ocean import (
    OCEAN_CREDENTIALS,
    LazySampler,
    call_ocean,
    create_sampler,
    dwave_availability,
    register_ocean_config_token,
    resolved,
)

logger = logging.getLogger(__name__)

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
    credentials=OCEAN_CREDENTIALS,
    parameter_limits=[
        ParameterLimit(
            preference="num_reads", limit="reads", error_code="QPU_READS_LIMIT"
        ),
        ParameterLimit(
            preference="dwave_qpu.annealing_time_us",
            limit="annealing_time_us",
            error_code="QPU_ANNEALING_TIME_LIMIT",
        ),
    ],
    description=(
        "D-Wave quantum annealer accessed through "
        "EmbeddingComposite(DWaveSampler()). Minor-embedding and chain-break "
        "resolution (Ocean's default majority_vote) are handled by Ocean; "
        "samples are returned in logical variables."
    ),
)

# Sampling-stage classification (spec §15), matched by class name across
# the exception's MRO (see ``solvers.ocean``). This table is QPU-specific:
# EmbeddingComposite raises a bare ValueError when no embedding is found →
# EMBEDDING_FAILED. Sampler construction uses the shared
# ``SAMPLER_INIT_EXCEPTION_CODES``, where the *same* ValueError means an
# invalid region / endpoint / profile / solver selection instead.
_SAMPLE_EXCEPTION_CODES = {
    "EmbeddingError": "EMBEDDING_FAILED",
    "ValueError": "EMBEDDING_FAILED",
    "SolverAuthenticationError": "REMOTE_AUTH_FAILED",
    "RequestTimeout": "REMOTE_TIMEOUT",
}


def _default_sampler_factory() -> Any:
    """Create a real embedded QPU sampler (requires the ``dwave`` extra)."""
    from dwave.system import DWaveSampler, EmbeddingComposite

    return EmbeddingComposite(DWaveSampler())


def _average_chain_break_fraction(sampleset: Any) -> float | None:
    """Mean of ``record.chain_break_fraction`` when the sampler provided one.

    Samplesets built without that vector have no such record field at all
    (attribute access would raise), so presence is checked via
    ``record.dtype.names``. Nothing else is guarded (2026-09-09 review
    F-26): with presence settled, any further exception is a bug of our
    own, and the service's fallback reports it rather than a silent None.
    """
    record = sampleset.record
    if "chain_break_fraction" not in (record.dtype.names or ()):
        return None
    values = record.chain_break_fraction
    if len(values) == 0:
        return None
    return float(values.mean())


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


class DWaveQPUBackend(BackendAliases):
    """Solves the compiled BQM directly on a D-Wave QPU.

    ``sampler_factory`` is the single test seam: production uses the default
    (``EmbeddingComposite(DWaveSampler())``), tests inject a fake. The
    sampler is built lazily on first use and cached while the credential
    sources are unchanged (``DWaveSampler()`` fetches the whole working
    graph and starts worker threads); a rotated token or edited Ocean
    config, or an authentication failure, makes the next solve rebuild it
    (review F-21); a failed construction is never cached.

    Chain strength is the user-provided option or Ocean's default — never
    derived from the compiled hard penalty (spec §15.1). Chain breaks are
    resolved by Ocean's default ``majority_vote``; no chain-break kwarg is
    forwarded.
    """

    def __init__(self, sampler_factory: Callable[[], Any] | None = None) -> None:
        self._sampler = LazySampler(sampler_factory, _default_sampler_factory)
        register_ocean_config_token()

    @property
    def capabilities(self) -> SolverCapabilities:
        return _CAPABILITIES

    def is_available(self) -> AvailabilityStatus:
        """Installability and credentials, via the shared check. No network I/O."""
        return dwave_availability()

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

        sampler = create_sampler(self._sampler, "D-Wave QPU")
        sampleset = call_ocean(
            "D-Wave QPU solve failed",
            _SAMPLE_EXCEPTION_CODES,
            lambda: resolved(sampler.sample(bqm, **sample_kwargs)),
            holder=self._sampler,
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

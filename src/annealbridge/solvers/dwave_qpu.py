"""D-Wave QPU solver backend (Phase 2 spec §15).

``dwave.system`` is imported lazily inside the default sampler factory so
this module — and the default registry that instantiates the backend —
imports cleanly without the ``dwave`` extra installed (spec §4).
"""

import importlib.util
import logging
from typing import Any, Callable

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import (
    CompiledProblem,
    DWaveQPUOptions,
    SolverPreferences,
)
from annealbridge.solvers.base import (
    RawSolverResult,
    SolverCapabilities,
    sampleset_to_lists,
)
from annealbridge.solvers.metadata import (
    ocean_config_status,
    redact,
    sanitize_sampleset_info,
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
    description=(
        "D-Wave quantum annealer accessed through "
        "EmbeddingComposite(DWaveSampler()). Minor-embedding and chain-break "
        "resolution (Ocean's default majority_vote) are handled by Ocean; "
        "samples are returned in logical variables."
    ),
)

# Ocean exception class names → catalog error codes. Matched by name across
# the exception's MRO so classification works (and is testable with fakes)
# without dwave-cloud-client installed. ValueError is listed because
# EmbeddingComposite raises a bare ValueError when no embedding is found
# (spec §15); the MRO walk starts at the most-derived class, so any named
# Ocean exception that happens to subclass ValueError still wins.
_EXCEPTION_NAME_TO_CODE = {
    "EmbeddingError": "EMBEDDING_FAILED",
    "ValueError": "EMBEDDING_FAILED",
    "SolverAuthenticationError": "REMOTE_AUTH_FAILED",
    "RequestTimeout": "REMOTE_TIMEOUT",
}


def _dwave_system_installed() -> bool:
    """Return whether ``dwave.system`` is importable, without importing it."""
    try:
        return importlib.util.find_spec("dwave.system") is not None
    except (ImportError, ValueError):
        return False


def _default_sampler_factory() -> Any:
    """Create a real embedded QPU sampler (requires the ``dwave`` extra)."""
    from dwave.system import DWaveSampler, EmbeddingComposite

    return EmbeddingComposite(DWaveSampler())


def _classify_exception(exc: Exception) -> str:
    for klass in type(exc).__mro__:
        code = _EXCEPTION_NAME_TO_CODE.get(klass.__name__)
        if code is not None:
            return code
    return "REMOTE_SOLVER_ERROR"


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
    (``EmbeddingComposite(DWaveSampler())``), tests inject a fake. Chain
    strength is the user-provided option or Ocean's default — never derived
    from the compiled hard penalty (spec §15.1). Chain breaks are resolved
    by Ocean's default ``majority_vote``; no chain-break kwarg is forwarded.
    """

    def __init__(self, sampler_factory: Callable[[], Any] | None = None) -> None:
        self._sampler_factory = sampler_factory

    @property
    def capabilities(self) -> SolverCapabilities:
        return _CAPABILITIES

    def is_available(self) -> tuple[bool, str | None]:
        """Check installability and credentials. No network I/O.

        Reasons are categorical strings only and never contain config
        values (spec §10).
        """
        if not _dwave_system_installed():
            return (False, "dwave-system not installed")
        status = ocean_config_status()
        if status == "invalid":
            return (False, "D-Wave configuration invalid")
        if status == "missing":
            return (False, "D-Wave credentials not configured")
        return (True, None)

    @property
    def name(self) -> str:
        """Alias for ``capabilities.name``."""
        return self.capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        """Alias for ``capabilities.exhaustive``."""
        return self.capabilities.exhaustive

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
        Policy ceilings are enforced by the service layer, not here.
        """
        bqm = compiled_problem.model
        options = preferences.dwave_qpu or DWaveQPUOptions()

        # Assembled outside the try block: only Ocean exceptions may reach
        # the classifier (which maps bare ValueError to EMBEDDING_FAILED).
        sample_kwargs: dict[str, Any] = {
            "num_reads": preferences.num_reads,
            "auto_scale": options.auto_scale,
        }
        if options.annealing_time_us is not None:
            sample_kwargs["annealing_time"] = options.annealing_time_us
        if options.chain_strength is not None:
            sample_kwargs["chain_strength"] = options.chain_strength

        try:
            factory = self._sampler_factory or _default_sampler_factory
            sampler = factory()
            sampleset = sampler.sample(bqm, **sample_kwargs)
        except Exception as exc:
            code = _classify_exception(exc)
            raise SolverExecutionError(
                redact(f"D-Wave QPU solve failed: {exc}"),
                code=code,
            ) from exc

        samples, energies = sampleset_to_lists(sampleset)
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
            samples=samples,
            energies=energies,
            backend=self.name,
            metadata=metadata,
        )

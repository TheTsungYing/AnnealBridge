"""Server settings read from the environment (Phase 2 spec §9, 3a §11).

Composition root only: just ``annealbridge.interfaces.*`` may import this
package (spec §4). D-Wave credentials deliberately stay out of these
settings — Ocean's native config handles them (spec §18).
"""

import logging
import os
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from pydantic_settings import SettingsError as PydanticSettingsError

from annealbridge.orchestration.policy import ExecutionPolicy, validate_limits

logger = logging.getLogger(__name__)


class SettingsError(ValueError):
    """The environment holds an invalid ``ANNEALBRIDGE_*`` value.

    The message is already formatted for an operator (one line per field),
    so an entry point can print it verbatim to stderr and exit. It names
    the variable and pydantic's reason but never echoes the value
    (2026-09-09 review F-20): a credential mis-set into an ``ANNEALBRIDGE_*``
    variable must not be printed back by the very message that rejects it.
    """


def validate_http_host(value: str) -> str:
    """Return ``value`` unchanged if it can serve as a bind address.

    Rejects an empty, blank or whitespace-bearing host (2026-09-11 review
    F01 / F09). An empty string is not an authorization to bind every
    interface: in asyncio (and uvicorn) it means exactly that, and this
    server has no authentication of any kind, so a mis-set variable must
    fail loudly rather than open the service to the network. A value with a
    stray space is a typo, not a host, and is not silently stripped either.

    Anything else is accepted verbatim — an operator who deliberately names
    a non-loopback address keeps that ability. The message never echoes the
    value (review F-20).
    """
    if not value or any(character.isspace() for character in value):
        raise ValueError("host must not be empty, blank or contain whitespace")
    return value


def validate_http_port(value: int) -> int:
    """Return ``value`` unchanged if it is a usable TCP port (1–65535).

    ``0`` is rejected along with everything outside the range (2026-09-11
    review F01 / F09): the kernel would pick an arbitrary free port, which
    no MCP host can then be pointed at. Out-of-range values used to survive
    until the socket bind raised ``OverflowError`` with a traceback. The
    message never echoes the value (review F-20).
    """
    if not 1 <= value <= 65535:
        raise ValueError("port must be between 1 and 65535")
    return value


class ServerSettings(BaseSettings):
    """Environment-driven configuration with the ``ANNEALBRIDGE_`` prefix.

    The limit fields mirror :class:`ExecutionPolicy`, bounds included, so an
    invalid environment is rejected here — before a policy is built and long
    before the first solve. ``limits`` is read from ``ANNEALBRIDGE_LIMITS``
    as a JSON object (e.g. ``'{"iterations": 100000}'``);
    ``enabled_backends`` from ``ANNEALBRIDGE_ENABLED_BACKENDS`` as a
    comma-separated list of registry names (unset or empty: every
    registered backend), the value being the only way an operator can reach
    the policy's ``enabled_backends`` gate (2026-09-09 review F-18).
    """

    model_config = SettingsConfigDict(env_prefix="ANNEALBRIDGE_")

    allow_remote: bool = False
    allow_remote_retries: bool = False
    exact_max_variables: int = Field(default=24, ge=1)
    max_qpu_reads: int = Field(default=1000, ge=1)
    max_qpu_annealing_time_us: float = Field(default=2000.0, gt=0)
    max_remote_time_seconds: int = Field(default=300, ge=1)
    max_concurrent_solves: int = Field(default=4, ge=1)
    # 2026-09-09 review (F-02 / F-07): ceilings on the caller-controlled
    # parameters; same defaults and bounds as ExecutionPolicy.
    max_local_reads: int = Field(default=100000, ge=1)
    max_sweeps: int = Field(default=100000, ge=1)
    max_local_retries: int = Field(default=10, ge=0)
    max_remote_retries: int = Field(default=3, ge=0)
    max_top_k: int = Field(default=1000, ge=1)
    # Batch 4 (G): post-processing ceilings; same defaults and bounds as
    # ExecutionPolicy.
    max_postprocess_candidates: int = Field(default=100, ge=1)
    max_postprocess_evaluations: int = Field(default=20_000_000, ge=1)
    # Shards the ``simulated_annealing`` backend samples at once; ``None``
    # detects the CPUs available to the process. A speed knob handed to the
    # registry by the composition root, not a policy limit: it never changes
    # a result, so it is not part of ExecutionPolicy (like ``http_*``).
    sa_workers: int | None = Field(default=None, ge=1)
    # Shards the ``tabu`` backend samples at once; ``None`` detects the CPUs
    # available to the process. The same kind of speed knob as
    # ``sa_workers``, per backend because the two samplers are tuned
    # independently.
    tabu_workers: int | None = Field(default=None, ge=1)
    # Where the ``simulated_bifurcation`` backend runs its dynamics: ``cpu``
    # (numpy, core install) or ``cuda`` (PyTorch from the ``gpu`` extra). A
    # machine knob like the worker counts, not a policy: it changes speed
    # and, across devices, floating-point rounding, never the contract. A
    # ``cuda`` setting without a usable CUDA device makes the backend report
    # itself unavailable; it is never substituted by the CPU.
    sb_device: Literal["cpu", "cuda"] = "cpu"
    # Largest compiled problem the ``simulated_bifurcation`` backend accepts:
    # it holds the couplings as a dense single-precision N x N matrix (4 N^2
    # bytes, 400 MB at the default), so this sizes the machine's memory. A
    # problem above it is refused with SB_VARIABLE_LIMIT, never clamped. Not
    # a policy limit key: it is one backend's memory, like a worker count is
    # one backend's CPU.
    sb_max_variables: int = Field(default=10_000, ge=1)
    # ``NoDecode``: pydantic-settings would otherwise parse a set as JSON;
    # the operator writes ``exact,simulated_annealing`` instead.
    enabled_backends: Annotated[set[str] | None, NoDecode] = None
    limits: dict[str, float] = Field(default_factory=dict)
    # Validated by the module-level functions below, which the MCP entry
    # point reuses for ``--host`` / ``--port`` so both paths share one rule.
    http_host: str = "127.0.0.1"
    http_port: int = 8000

    @field_validator("enabled_backends", mode="before")
    @classmethod
    def _split_enabled_backends(cls, value: object) -> object:
        """Comma-separated names → set; blank entries dropped; empty → None."""
        if isinstance(value, str):
            names = {part.strip() for part in value.split(",") if part.strip()}
            return names or None
        return value

    @field_validator("http_host")
    @classmethod
    def _validate_http_host(cls, value: str) -> str:
        # Same rule the ``--host`` argument obeys (2026-09-11 review F01 /
        # F09), so the environment and a CLI override cannot disagree.
        return validate_http_host(value)

    @field_validator("http_port")
    @classmethod
    def _validate_http_port(cls, value: int) -> int:
        # Same rule the ``--port`` argument obeys (2026-09-11 review F01 / F09).
        return validate_http_port(value)

    @field_validator("limits")
    @classmethod
    def _validate_limits(cls, value: dict[str, float]) -> dict[str, float]:
        # Same rule as ExecutionPolicy, applied at the environment boundary
        # so the operator sees ANNEALBRIDGE_LIMITS named in the error.
        return validate_limits(value)

    def to_policy(self) -> ExecutionPolicy:
        """Build the :class:`ExecutionPolicy` these settings describe.

        Copies exactly the fields ``ExecutionPolicy`` declares, by name, so
        a limit added to both models reaches the policy without this method
        having to list it. The settings-only fields (``sa_workers``,
        ``tabu_workers``, ``sb_device``, ``sb_max_variables``, ``http_host``,
        ``http_port``) are left out on purpose rather than
        passed and silently ignored: they configure the composition root,
        not the policy. That every policy field exists here with the same
        type, default and bounds is pinned by ``tests/unit/test_settings.py``.
        """
        return ExecutionPolicy(
            **{name: getattr(self, name) for name in ExecutionPolicy.model_fields}
        )


def unknown_settings_variables(environ: Mapping[str, str] = os.environ) -> list[str]:
    """Names in ``environ`` that carry the ``ANNEALBRIDGE_`` prefix but match
    no :class:`ServerSettings` field, sorted.

    pydantic-settings only ever *looks up* the fields it knows, so a typo
    such as ``ANNEALBRIDGE_MAX_QPU_READ`` is silently ignored and the default
    applies (``extra="forbid"`` does not catch env vars); this scan is what
    makes the mistake visible. Prefix and suffix are compared
    case-insensitively, as the settings lookup itself is. Only names are
    returned — never values.
    """
    prefix = str(ServerSettings.model_config["env_prefix"]).upper()
    known = {name.upper() for name in ServerSettings.model_fields}
    unknown = [
        name
        for name in environ
        if name.upper().startswith(prefix) and name.upper()[len(prefix) :] not in known
    ]
    return sorted(unknown)


def load_settings() -> ServerSettings:
    """Read :class:`ServerSettings` from the environment.

    Raises :class:`SettingsError` with an operator-readable message instead
    of letting pydantic's ``ValidationError`` (and its traceback) escape.
    A value that cannot even be parsed (an ``ANNEALBRIDGE_LIMITS`` that is
    not JSON) surfaces from pydantic-settings as its own ``SettingsError``
    — a ``ValueError`` without ``.errors()`` — and is converted the same way.
    Neither message echoes the offending value (review F-20).

    An unknown ``ANNEALBRIDGE_*`` variable is not an error — the process
    starts with the default for whatever the operator meant — but it is
    logged as a WARNING naming the variable, so a misspelt ceiling does not
    silently stay at its default.
    """
    try:
        settings = ServerSettings()
    except ValidationError as exc:
        lines = [
            f"Invalid server settings ({exc.error_count()} error(s)); "
            "values are not echoed:"
        ]
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            variable = f"{ServerSettings.model_config['env_prefix']}{field}".upper()
            lines.append(f"  {variable}: {error['msg']}")
        raise SettingsError("\n".join(lines)) from None
    except PydanticSettingsError as exc:
        raise SettingsError(f"Invalid server settings: {exc}") from None
    unknown = unknown_settings_variables()
    if unknown:
        logger.warning(
            "Ignoring unknown ANNEALBRIDGE_* variable(s): %s (values not shown); "
            "the corresponding defaults stay in effect",
            ", ".join(unknown),
        )
    return settings

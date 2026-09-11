"""Generic, declaration-driven limit and gate checks (Phase 3a spec §12).

Pure functions shared by the service and by routing. They read a
backend's *declaration* — ``SolverCapabilities.parameter_limits`` and the
capability flags — never a backend name, and this module imports no
concrete backend or compiler (spec §4, overview principle 4).
"""

import logging
import types
import typing
from typing import Annotated, get_args, get_origin

from pydantic import BaseModel

from annealbridge.compiler.base import ModelCompiler
from annealbridge.models import (
    ModelType,
    SolveError,
    SolveStatus,
    SolverCapabilities,
    SolverPreferences,
    catalog_error,
)
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.solvers.base import SolverBackend
from annealbridge.solvers.metadata import redact

logger = logging.getLogger(__name__)

# AvailabilityStatus.category → (result status, default error code), per
# spec §8.2. Keyed on the structured category, never on a backend's reason
# text. A backend may name a more specific ``error_code`` on its status
# (e.g. the D-Wave backends report DWAVE_CONFIG_INVALID); the default here
# only applies when it does not.
AVAILABILITY_MAP: dict[str, tuple[SolveStatus, str]] = {
    "not_installed": ("backend_unavailable", "BACKEND_NOT_INSTALLED"),
    "credentials_missing": ("backend_unavailable", "REMOTE_CREDENTIALS_MISSING"),
    "config_invalid": ("configuration_error", "BACKEND_CONFIG_INVALID"),
    "unavailable": ("backend_unavailable", "BACKEND_UNAVAILABLE"),
}
# What an availability check that cannot be trusted maps to (2026-09-09
# review, service-layer follow-ups): a category outside the map (only
# reachable by bypassing the ``AvailabilityCategory`` Literal) and an
# ``is_available()`` that raises. Both mean "this backend cannot be run
# right now and its own check could not say why", which is what the
# generic entry already expresses.
_AVAILABILITY_FALLBACK: tuple[SolveStatus, str] = AVAILABILITY_MAP["unavailable"]


def _nested_model(annotation: object) -> type[BaseModel] | None:
    """The BaseModel subclass a field annotation refers to, if any.

    Handles both a bare model type and ``Model | None``; anything else
    (``int``, ``float``, ...) is a leaf and returns None.
    """
    for candidate in (annotation, *get_args(annotation)):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def _numeric_leaf(annotation: object) -> bool:
    """Whether a field annotation is a numeric preference (``int`` / ``float``).

    Unwraps ``X | None`` / ``Optional[X]`` and ``Annotated[X, ...]`` (the
    IR's ``Count`` / ``Quantity`` types) and requires every remaining
    member to be exactly ``int`` or ``float``. ``bool`` is not numeric
    even though it subclasses ``int``: a limit on a flag is meaningless.
    """
    origin = get_origin(annotation)
    if origin is Annotated:
        return _numeric_leaf(get_args(annotation)[0])
    if origin is types.UnionType or origin is typing.Union:
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        return bool(members) and all(_numeric_leaf(member) for member in members)
    return annotation is int or annotation is float


def read_preference(preferences: SolverPreferences, path: str) -> float | int | None:
    """Value at the dotted ``path`` into ``preferences``.

    Returns None when an option block along the way is unset (the user did
    not fill it in). The whole path is validated against the model classes
    regardless, so a declaration naming a field that does not exist raises
    ``ValueError`` even when the block is None — a wrong declaration is a
    backend bug the tests should catch, not a silently unlimited parameter.
    The leaf must be a numeric field (2026-09-09 review, service-layer
    follow-ups): a declaration pointing at ``backend``, an option block or
    a flag is refused here, by annotation, so the service rejects it at
    construction instead of failing with a ``TypeError`` on the first
    ``value > maximum`` comparison of a solve.
    """
    parts = path.split(".")
    model: type[BaseModel] | None = type(preferences)
    value: object = preferences
    annotation: object = None
    for index, part in enumerate(parts):
        if model is None or part not in model.model_fields:
            raise ValueError(
                f"preference path '{path}' does not exist on "
                f"{type(preferences).__name__}"
            )
        if value is not None:
            value = getattr(value, part)
        annotation = model.model_fields[part].annotation
        model = _nested_model(annotation)
        if index < len(parts) - 1 and model is None:
            raise ValueError(
                f"preference path '{path}' does not exist on "
                f"{type(preferences).__name__}: '{part}' is not an option block"
            )
    if not _numeric_leaf(annotation):
        raise ValueError(
            f"preference path '{path}' is not a numeric preference "
            f"(annotation {annotation!r}); only int / float fields can carry "
            f"a limit"
        )
    if value is None:
        return None
    # The annotation check above is the real guard; this keeps the
    # returned type honest for a value that somehow bypassed validation.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"preference path '{path}' holds a non-numeric value {value!r}"
        )
    return value


def select_model_type(
    capabilities: SolverCapabilities, compilers: dict[ModelType, ModelCompiler]
) -> ModelType | None:
    """3a §16.1: the first declared model type there is a compiler for.

    The single rule that decides which compiler path a backend takes;
    ``solve``, ``validate`` and ``recommend`` all call it so they agree.
    It dispatches on the backend's *declaration*, never on its name. None
    means no compiler fits (solve reports NO_COMPILER_FOR_MODEL_TYPE).
    """
    for model_type in capabilities.supported_model_types:
        if model_type in compilers:
            return model_type
    return None


def limit_error(code: str, label: str, value: object, maximum: object) -> SolveError:
    """A preference-limit error in the wording every limit shares."""
    return catalog_error(code, f"{label} {value} exceeds the server maximum of {maximum}")


def no_compiler_error(
    backend_name: str, capabilities: SolverCapabilities, *, path: str | None = None
) -> SolveError:
    """NO_COMPILER_FOR_MODEL_TYPE in the one wording solve, validate and recommend share.

    ``path`` is ``solver.backend`` where the report is advisory (validate's
    warning, recommend's blocking entry) and ``None`` for solve's error.
    """
    return catalog_error(
        "NO_COMPILER_FOR_MODEL_TYPE",
        f"Backend '{backend_name}' accepts model types "
        f"[{', '.join(capabilities.supported_model_types)}] but the server "
        f"has no compiler for any of them",
        path=path,
    )


def exact_variable_limit_error(
    num_variables: int, limit: int | float, *, path: str | None = None
) -> SolveError:
    """§14 step 9 / 3a §12.2: the exhaustive backend's variable ceiling.

    One wording for every place it is checked: from the estimate before
    compile, from the compiled model after (2026-09-09 review F-14), and
    from recommend's advisory estimate. The estimate equals the compiled
    count for every validated problem (``estimate_model_variables``), so
    the sentence is true either way.
    """
    return catalog_error(
        "EXACT_VARIABLE_LIMIT",
        f"Compiled problem has {num_variables} variables (including "
        f"internal), exceeding the exhaustive backend limit of {limit}",
        path=path,
    )


def preference_limit_errors(
    capabilities: SolverCapabilities,
    preferences: SolverPreferences,
    policy: ExecutionPolicy,
) -> list[SolveError]:
    """§16.2 step 8: every parameter limit the preferences exceed.

    Walks ``capabilities.parameter_limits``; each violation becomes a
    catalog error under the declared ``error_code``, labelled with the
    dotted preference path. Never clamps, and collects every violation so
    the caller can fix them all at once. A declaration whose limit key the
    policy has no value for raises ``ValueError`` — the service refuses to
    be built in that state (spec §11.3); this is the pure function's own
    guard for callers that bypass it.

    After the declared limits come the two service-level ceilings added by
    the 2026-09-09 review (F-02 / F-07): ``max_retries`` against the
    retry ceiling chosen by the backend's ``remote`` flag, and ``top_k``
    against ``max_top_k``. They are flag-driven rather than declared
    because the retry loop and the candidate cut belong to the service,
    not to any backend, so a backend that declares nothing must still be
    covered. The retry ceiling applies whether or not this solve would
    actually retry (an exhaustive backend, a native-constraint model or
    disabled remote retries): one rule for every backend, and a value that
    is over the ceiling is refused rather than quietly ignored.
    """
    errors: list[SolveError] = []
    for declaration in capabilities.parameter_limits:
        maximum = policy.limit(declaration.limit)
        if maximum is None:
            raise ValueError(
                f"backend '{capabilities.name}' declares limit "
                f"'{declaration.limit}' but the policy has no value for it"
            )
        value = read_preference(preferences, declaration.preference)
        if value is not None and value > maximum:
            errors.append(
                limit_error(
                    declaration.error_code, declaration.preference, value, maximum
                )
            )
    retries_maximum = policy.required_limit(policy.retries_limit_key(capabilities))
    if preferences.max_retries > retries_maximum:
        errors.append(
            limit_error(
                "RETRY_LIMIT", "max_retries", preferences.max_retries, retries_maximum
            )
        )
    top_k_maximum = policy.required_limit("top_k")
    if preferences.top_k > top_k_maximum:
        errors.append(
            limit_error("TOP_K_LIMIT", "top_k", preferences.top_k, top_k_maximum)
        )
    return errors


def gate_errors(
    backend_name: str, backend: SolverBackend, policy: ExecutionPolicy
) -> tuple[SolveStatus, str, list[SolveError]] | None:
    """§16.2 steps 3–5: enabled_backends → allow_remote → availability.

    The gates short-circuit in that order, so ``is_available()`` is only
    called once policy allows the backend at all (the D-Wave availability
    check reads the Ocean config file; this keeps Phase 2's lazy order).

    Returns ``(status, reported_backend_name, errors)`` or None when the
    backend may run. ``enabled_backends`` holds registry keys — the names
    the user requests and the capabilities view reports — so that gate
    compares and reports ``backend_name``; the other two report
    ``capabilities.name``, which a custom registry may register under a
    different key. Never substitutes another backend.

    The availability check is third-party code from the service's point
    of view, so it is guarded like a backend's ``solve`` (2026-09-09
    review, service-layer follow-ups): an ``is_available()`` that raises,
    or a status whose category is outside :data:`AVAILABILITY_MAP`, is
    reported as ``backend_unavailable`` / ``BACKEND_UNAVAILABLE`` with the
    reason in the message (redacted, since it never passed through the
    backend's own wrapping). Both ``solve`` and ``recommend`` go through
    here, so neither can be taken down by one backend's broken check.
    """
    caps = backend.capabilities
    if policy.enabled_backends is not None and backend_name not in policy.enabled_backends:
        return (
            "backend_unavailable",
            backend_name,
            [
                catalog_error(
                    "BACKEND_DISABLED_BY_POLICY",
                    f"Backend '{backend_name}' is disabled by server policy; "
                    f"enabled backends: {', '.join(sorted(policy.enabled_backends))}",
                )
            ],
        )
    if caps.remote and not policy.allow_remote:
        return (
            "backend_unavailable",
            caps.name,
            [
                catalog_error(
                    "REMOTE_DISABLED",
                    f"Backend '{caps.name}' is remote and remote solving is "
                    f"disabled by server policy",
                )
            ],
        )
    try:
        availability = backend.is_available()
    except Exception as exc:
        # Only ``Exception``: KeyboardInterrupt / SystemExit must propagate.
        message = redact(
            f"Backend '{caps.name}' availability check failed: "
            f"unexpected {type(exc).__name__}: {exc}"
        )
        logger.warning("%s", message)
        status, code = _AVAILABILITY_FALLBACK
        return (status, caps.name, [catalog_error(code, message)])
    if not availability.available:
        mapped = AVAILABILITY_MAP.get(availability.category)
        status, default_code = mapped if mapped is not None else _AVAILABILITY_FALLBACK
        detail = availability.detail or "no reason reported"
        if mapped is None:
            detail += f" (unknown availability category {availability.category!r})"
        return (
            status,
            caps.name,
            [
                catalog_error(
                    availability.error_code or default_code,
                    f"Backend '{caps.name}' is unavailable: {detail}",
                )
            ],
        )
    return None

"""Generic, declaration-driven limit and gate checks (Phase 3a spec §12).

Pure functions shared by the service (and, later, routing). They read a
backend's *declaration* — ``SolverCapabilities.parameter_limits`` and the
capability flags — never a backend name, and this module imports no
concrete backend or compiler (spec §4, overview principle 4).
"""

from typing import get_args

from pydantic import BaseModel

from annealbridge.models import (
    SolveError,
    SolverCapabilities,
    SolverPreferences,
    catalog_error,
)
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.solvers.base import SolverBackend

# AvailabilityStatus.category → (result status, default error code), per
# spec §8.2. Keyed on the structured category, never on a backend's reason
# text. A backend may name a more specific ``error_code`` on its status
# (e.g. the D-Wave backends report DWAVE_CONFIG_INVALID); the default here
# only applies when it does not.
AVAILABILITY_MAP: dict[str, tuple[str, str]] = {
    "not_installed": ("backend_unavailable", "BACKEND_NOT_INSTALLED"),
    "credentials_missing": ("backend_unavailable", "REMOTE_CREDENTIALS_MISSING"),
    "config_invalid": ("configuration_error", "BACKEND_CONFIG_INVALID"),
    "unavailable": ("backend_unavailable", "BACKEND_UNAVAILABLE"),
}


def _nested_model(annotation: object) -> type[BaseModel] | None:
    """The BaseModel subclass a field annotation refers to, if any.

    Handles both a bare model type and ``Model | None``; anything else
    (``int``, ``float``, ...) is a leaf and returns None.
    """
    for candidate in (annotation, *get_args(annotation)):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def read_preference(preferences: SolverPreferences, path: str) -> float | int | None:
    """Value at the dotted ``path`` into ``preferences``.

    Returns None when an option block along the way is unset (the user did
    not fill it in). The whole path is validated against the model classes
    regardless, so a declaration naming a field that does not exist raises
    ``ValueError`` even when the block is None — a wrong declaration is a
    backend bug the tests should catch, not a silently unlimited parameter.
    """
    parts = path.split(".")
    model: type[BaseModel] | None = type(preferences)
    value: object = preferences
    for index, part in enumerate(parts):
        if model is None or part not in model.model_fields:
            raise ValueError(
                f"preference path '{path}' does not exist on "
                f"{type(preferences).__name__}"
            )
        if value is not None:
            value = getattr(value, part)
        model = _nested_model(model.model_fields[part].annotation)
        if index < len(parts) - 1 and model is None:
            raise ValueError(
                f"preference path '{path}' does not exist on "
                f"{type(preferences).__name__}: '{part}' is not an option block"
            )
    return value  # type: ignore[return-value]


def limit_error(code: str, label: str, value: object, maximum: object) -> SolveError:
    """A preference-limit error in the wording every limit shares."""
    return catalog_error(code, f"{label} {value} exceeds the server maximum of {maximum}")


def preference_limit_errors(
    capabilities: SolverCapabilities,
    preferences: SolverPreferences,
    policy: ExecutionPolicy,
) -> list[SolveError]:
    """§16.2 step 8: every declared parameter limit the preferences exceed.

    Walks ``capabilities.parameter_limits``; each violation becomes a
    catalog error under the declared ``error_code``, labelled with the
    dotted preference path. Never clamps, and collects every violation so
    the caller can fix them all at once. A declaration whose limit key the
    policy has no value for raises ``ValueError`` — the service refuses to
    be built in that state (spec §11.3); this is the pure function's own
    guard for callers that bypass it.
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
    return errors


def gate_errors(
    backend_name: str, backend: SolverBackend, policy: ExecutionPolicy
) -> tuple[str, str, list[SolveError]] | None:
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
    availability = backend.is_available()
    if not availability.available:
        status, default_code = AVAILABILITY_MAP[availability.category]
        return (
            status,
            caps.name,
            [
                catalog_error(
                    availability.error_code or default_code,
                    f"Backend '{caps.name}' is unavailable: "
                    f"{availability.detail or 'no reason reported'}",
                )
            ],
        )
    return None

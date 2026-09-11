"""The installed package version, readable from every layer.

Standard library only, so the core (``models``, ``orchestration``) can stamp
results with it without touching ``annealbridge.config`` or the interfaces.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["package_version"]


def package_version() -> str:
    """The installed distribution's version, or ``"unknown"`` outside one."""
    try:
        return version("annealbridge")
    except PackageNotFoundError:
        return "unknown"

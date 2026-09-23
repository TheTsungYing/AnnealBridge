"""The package version repeated in the documentation (2026-09-23).

A release sets the version in ``pyproject.toml`` and then has to repeat it by
hand in a few more places. ``server.json`` is one of them and is already
checked by ``test_registry_entry.py``; the others are prose and examples in the
documentation, which nothing else in the suite reads:

* the ``Version X.Y.Z:`` status line near the end of ``README.md``;
* the ``版本 X.Y.Z：`` line in the same place in ``README.zh-TW.md``;
* the ``annealbridge_version`` field of the capabilities example in
  ``docs/output-format.md``.

Each must mention the version exactly once and equal ``pyproject.toml``. No
match means the sentence was reworded and the pattern here needs updating
with it; more than one means a new mention appeared and someone has to decide
whether it belongs in this list too.

The version the running code reports is read from ``importlib.metadata`` and
is not a hand-maintained copy, so it is not checked here.
"""

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

DOCUMENTED_VERSIONS = [
    ("README.md", re.compile(r"^Version (\S+?):", re.MULTILINE)),
    ("README.zh-TW.md", re.compile(r"^版本 (\S+?)：", re.MULTILINE)),
    ("docs/output-format.md", re.compile(r'"annealbridge_version": "([^"]+)"')),
]


def _package_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _documented_version(relative_path: str, text: str, pattern: re.Pattern) -> str:
    found = pattern.findall(text)
    assert len(found) == 1, (
        f"{relative_path}: expected exactly one match of {pattern.pattern!r}, "
        f"found {len(found)}: {found!r}. None means the sentence was reworded "
        "and this test must follow it; several mean a new mention appeared "
        "and should be added to the release checklist or the pattern narrowed."
    )
    return found[0]


@pytest.mark.parametrize(
    ("relative_path", "pattern"),
    DOCUMENTED_VERSIONS,
    ids=[path for path, _ in DOCUMENTED_VERSIONS],
)
def test_documented_version_matches_pyproject(
    relative_path: str, pattern: re.Pattern
) -> None:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    documented = _documented_version(relative_path, text, pattern)
    expected = _package_version()

    assert documented == expected, (
        f"{relative_path} documents version {documented}, but pyproject.toml "
        f"is at {expected}. Update it as part of the release (see "
        "CONTRIBUTING.md, Releasing)."
    )

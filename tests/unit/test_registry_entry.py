"""The MCP Registry entry (2026-09-16).

``server.json`` and ``README.md`` carry two things the registry depends on that
nothing else in the suite would notice going missing:

* the ownership marker. The registry proves a PyPI package belongs to the
  server by looking for an ``mcp-name: <server name>`` line in the description
  PyPI renders from ``README.md``. Deleting it — reordering the README, say —
  still lets PyPI publish and fails only the registry job, after the release is
  already out and its version number spent.
* the version, which ``server.json`` repeats twice and which must equal the one
  in ``pyproject.toml``, or the registry ends up pointing at a release that is
  not the one being published.

The release workflow checks both against the tag as well, but only once a
release is under way; these run on every ``pytest``.
"""

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SERVER_NAME = "io.github.TheTsungYing/annealbridge"
OWNERSHIP_MARKER = f"mcp-name: {SERVER_NAME}"


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def _server_json() -> dict:
    return json.loads((ROOT / "server.json").read_text(encoding="utf-8"))


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def test_readme_carries_a_usable_ownership_marker() -> None:
    """Present, and followed by a boundary.

    The registry needs whitespace, an HTML tag or ``-->`` after the name;
    glued to a trailing character — a sentence-ending period, say — it does
    not match, which fails just as silently as deleting the line.
    """
    readme = _readme()
    hint = (
        "README.md must keep the "
        f"<!-- {OWNERSHIP_MARKER} --> comment: the MCP Registry reads it out "
        "of the description PyPI renders from this file to prove the package "
        "is ours. Without it the next registry publish fails."
    )

    assert OWNERSHIP_MARKER in readme, hint

    after = readme[readme.index(OWNERSHIP_MARKER) + len(OWNERSHIP_MARKER) :]
    assert after[:1].isspace() or after.startswith(("-->", "<")), (
        f"{OWNERSHIP_MARKER!r} is followed by {after[:10]!r}; the registry "
        "needs whitespace, an HTML tag or '-->' there. " + hint
    )


def test_server_json_declares_the_same_name_as_the_marker() -> None:
    assert _server_json()["name"] == SERVER_NAME


def test_server_json_points_at_this_distribution() -> None:
    """The ownership check runs against the package named here, so it has to be
    the distribution this repository publishes."""
    project = _pyproject()
    identifiers = {
        package["identifier"]
        for package in _server_json()["packages"]
        if package["registryType"] == "pypi"
    }

    assert identifiers == {project["name"]}


def test_every_version_in_server_json_matches_pyproject() -> None:
    expected = _pyproject()["version"]
    server = _server_json()
    found = {"server.json version": server["version"]} | {
        f"package {package['identifier']}": package["version"]
        for package in server["packages"]
    }

    assert found == dict.fromkeys(found, expected)

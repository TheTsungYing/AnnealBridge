"""The ``annealbridge-mcp`` console-script shim (2026-09-11 install
verification, gap 3).

A core-only install ships the script but not the ``mcp`` package, so the shim
must turn that one missing dependency into a message and exit code 2, while
letting every other import failure surface unchanged. All of it is exercised
in-process: only ``tests/mcp/test_stdio_entrypoint.py`` may spawn a
subprocess.
"""

import builtins
import sys

import pytest

from annealbridge.interfaces import mcp_entrypoint

MCP_INTERFACE_PACKAGE = "annealbridge.interfaces.mcp"


def _forget_mcp_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the already-imported MCP modules so ``main()`` re-imports them.

    ``monkeypatch`` restores ``sys.modules`` afterwards, so the in-memory MCP
    tests keep the module objects they were loaded with.
    """
    for name in list(sys.modules):
        if (
            name == MCP_INTERFACE_PACKAGE
            or name.startswith(MCP_INTERFACE_PACKAGE + ".")
            or name == "mcp"
            or name.startswith("mcp.")
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_missing_mcp_extra_exits_2_with_one_line_and_no_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """2026-09-11 install verification (gap 3): without the ``[mcp]`` extra the
    console script prints the install hint and exits 2, never a traceback."""
    _forget_mcp_modules(monkeypatch)
    # A ``None`` entry in sys.modules makes the import machinery raise
    # ModuleNotFoundError(name="mcp") without touching the filesystem.
    monkeypatch.setitem(sys.modules, "mcp", None)

    with pytest.raises(SystemExit) as excinfo:
        mcp_entrypoint.main()

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "annealbridge[mcp]" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_core_only_install_without_mcp_or_anyio_still_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """2026-09-15 ruff import-sort review: a real core-only install lacks
    ``anyio`` as well as ``mcp`` (both come with the ``[mcp]`` extra). The
    package ``__init__`` must import the server before the tools, or the
    missing ``anyio`` surfaces first and the shim prints a traceback."""
    _forget_mcp_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "anyio", None)

    with pytest.raises(SystemExit) as excinfo:
        mcp_entrypoint.main()

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "annealbridge[mcp]" in captured.err
    assert "Traceback" not in captured.err


def test_delegates_to_the_server_main_when_the_extra_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-09-11 install verification (gap 3): with the extra present the shim
    is transparent — it calls the real server entry point exactly once. The
    console script passes the server's own defaults, so it keeps reading
    ``sys.argv`` and naming itself ``annealbridge-mcp``."""
    from annealbridge.interfaces.mcp import server

    calls: list[tuple] = []
    monkeypatch.setattr(server, "main", lambda *args, **kwargs: calls.append((args, kwargs)))

    mcp_entrypoint.main()

    assert calls == [((None,), {"prog": "annealbridge-mcp"})]


def test_run_passes_argv_and_prog_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``annealbridge mcp`` subcommand reaches the server through the same
    guard, with the arguments Typer did not consume and its own program name."""
    from annealbridge.interfaces.mcp import server

    calls: list[tuple] = []
    monkeypatch.setattr(server, "main", lambda *args, **kwargs: calls.append((args, kwargs)))

    mcp_entrypoint.run(["--transport", "streamable-http"], prog="annealbridge mcp")

    assert calls == [
        ((["--transport", "streamable-http"],), {"prog": "annealbridge mcp"})
    ]


def test_run_reports_the_missing_extra_for_the_subcommand_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A core-only install must fail the same way whichever entry point is
    used: one stderr line, exit 2, no traceback."""
    _forget_mcp_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "mcp", None)

    with pytest.raises(SystemExit) as excinfo:
        mcp_entrypoint.run([], prog="annealbridge mcp")

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "annealbridge[mcp]" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mcp", True),
        ("mcp.server", True),
        ("anyio", False),
        ("annealbridge.interfaces.mcp.server", False),
        (None, False),
    ],
)
def test_only_the_third_party_mcp_package_counts_as_the_missing_extra(
    name: str | None, expected: bool
) -> None:
    """2026-09-11 install verification (gap 3): the shim must recognise the
    ``mcp`` distribution and nothing else — not another missing dependency and
    not our own ``annealbridge.interfaces.mcp`` modules."""
    assert mcp_entrypoint._is_missing_mcp(ModuleNotFoundError(name=name)) is expected


def test_a_different_missing_dependency_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-09-11 install verification (gap 3): an unrelated missing module
    (here ``anyio``, imported at the top of the tools module) propagates as a
    ModuleNotFoundError instead of being reported as the missing extra."""
    _forget_mcp_modules(monkeypatch)
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "anyio":
            raise ModuleNotFoundError("No module named 'anyio'", name="anyio")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        mcp_entrypoint.main()

    assert excinfo.value.name == "anyio"

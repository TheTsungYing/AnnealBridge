"""The MCP entry points: five real subprocesses, the rest in-process.

This is the only test file allowed to spawn a subprocess (spec §26), and only
five of its tests do. Three serve the four tools over stdio, covering every way
the server is started: ``python -m annealbridge.interfaces.mcp.server``, the
``annealbridge-mcp`` console script the installed distribution generates, and
the ``annealbridge mcp`` subcommand the MCP Registry entry resolves to.
The second is the regression test for the 2026-09-11 install verification
gaps 3 and 4 — the script now enters through
``annealbridge.interfaces.mcp_entrypoint``, and nothing else may notice.
The other two keep one invalid ``--port`` and one ``--version`` run in a real
process, the only place an exit code is seen leaving the process.

Every other argument, settings and version check drives ``server.main()``
in-process, because each subprocess costs about two seconds. Those tests keep
the assertions the subprocess versions made: the ``SystemExit`` code stands in
for the process exit code, ``capsys`` for its stdout and stderr, and
``caplog`` for the unknown-variable ``WARNING`` record, which pytest's logging
capture keeps out of ``capsys``. The ``in_process_main`` fixture replaces
``mcp.run`` with a guard that fails the test, so a regression that lets
``main()`` reach the transport turns red instead of blocking on stdio or
binding a socket. It also wraps ``server.main`` so the root logger's handlers
and level are put back when ``main()`` returns or raises: its log setup
replaces every root handler, pytest's capture handlers included.
"""

import logging
import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from annealbridge.config import SettingsError
from annealbridge.interfaces.mcp import server

pytestmark = pytest.mark.anyio

EXPECTED_TOOLS = [
    "get_optimization_capabilities",
    "recommend_backend",
    "solve_optimization",
    "validate_optimization_problem",
]


async def test_stdio_entrypoint_serves_the_four_tools():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "annealbridge.interfaces.mcp.server"],
    )
    async with Client(params) as client:
        tools = (await client.list_tools()).tools

    assert sorted(tool.name for tool in tools) == EXPECTED_TOOLS


def _console_script() -> str:
    """The ``annealbridge-mcp`` script generated next to this interpreter."""
    script = shutil.which("annealbridge-mcp", path=str(Path(sys.executable).parent))
    if script is None:
        pytest.fail(
            "the annealbridge-mcp console script is missing next to "
            f"{sys.executable}; reinstall the project with "
            'pip install -e ".[all,dev]"'
        )
    return script


async def test_console_script_serves_the_four_tools():
    """2026-09-11 install verification (gaps 3 and 4): the generated
    ``annealbridge-mcp`` script — not ``python -m`` — must start the real
    server through the entry-point shim and serve the same four tools."""
    script = _console_script()

    params = StdioServerParameters(command=script, args=[])
    async with Client(params) as client:
        tools = (await client.list_tools()).tools

    assert sorted(tool.name for tool in tools) == EXPECTED_TOOLS


def _cli_script() -> str:
    """The ``annealbridge`` script generated next to this interpreter."""
    script = shutil.which("annealbridge", path=str(Path(sys.executable).parent))
    if script is None:
        pytest.fail(
            "the annealbridge console script is missing next to "
            f"{sys.executable}; reinstall the project with "
            'pip install -e ".[all,dev]"'
        )
    return script


async def test_cli_subcommand_serves_the_four_tools():
    """2026-09-16 MCP Registry entry: the registry composes a package command
    as ``<runtimeHint> <runtimeArguments> <identifier> <packageArguments>``,
    and ``identifier`` must be the PyPI project name — so what a client
    actually runs is ``uvx --from annealbridge[mcp] annealbridge mcp``. That
    last part has to start the same server as the dedicated console script."""
    script = _cli_script()

    params = StdioServerParameters(command=script, args=["mcp"])
    async with Client(params) as client:
        tools = (await client.list_tools()).tools

    assert sorted(tool.name for tool in tools) == EXPECTED_TOOLS


HOST_REASON = "host must not be empty, blank or contain whitespace"
PORT_REASON = "port must be between 1 and 65535"


def _is_server_setting(name: str) -> bool:
    return name.upper().startswith("ANNEALBRIDGE_")


def _environ_without_server_settings() -> dict[str, str]:
    """This process's environment minus every ``ANNEALBRIDGE_*`` variable, so
    a developer's own server settings cannot change what a real process sees."""
    return {
        name: value for name, value in os.environ.items() if not _is_server_setting(name)
    }


def test_invalid_port_exits_2_in_a_real_process():
    """2026-09-11 review (F01 / F09), in a real process: an out-of-range
    ``--port`` under ``python -m`` ends the process with exit code 2 and one
    line — no traceback, and no socket ever opened.

    Only a real process proves that the exit code survives the ``__main__``
    block's delegation to the canonical module and reaches the outside, and
    that nothing was bound: the run would otherwise block until killed, so
    returning at all is the proof. Every invalid host and port is covered
    in-process by ``test_invalid_bind_argument_exits_2_before_binding_a_socket``.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "annealbridge.interfaces.mcp.server",
            "--transport",
            "streamable-http",
            "--port",
            "65536",
        ],
        env=_environ_without_server_settings(),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 2
    assert f"argument --port: {PORT_REASON}" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert completed.stdout == ""


def test_version_exits_0_with_invalid_settings_in_a_real_process():
    """2026-09-15 consolidation, in a real process: ``--version`` exits 0
    with the version on stdout even while an invalid and an unknown
    ``ANNEALBRIDGE_*`` variable are set, and stderr carries neither a
    settings error nor the unknown-variable warning. Only a real process
    shows the exit code leaving the process and stderr as the operator sees
    it; each settings case is covered in-process by
    ``test_version_exits_0_before_the_settings_are_read``.

    stderr is not required to be empty: an unrelated interpreter warning on
    a developer machine (``PYTHONWARNINGS``, say) may legitimately write to
    it.

    Driven through the console script: under ``python -m`` the interpreter
    itself writes runpy's "found in sys.modules" RuntimeWarning to stderr
    before the server code runs, which is noise this test has no use for."""
    completed = subprocess.run(
        [_console_script(), "--version"],
        env={
            **_environ_without_server_settings(),
            "ANNEALBRIDGE_MAX_CONCURRENT_SOLVES": "0",
            "ANNEALBRIDGE_NOT_A_REAL_SETTING": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0
    assert completed.stdout == f"annealbridge {metadata.version('annealbridge')}\n"
    # The two traces load_settings() leaves: the unknown-variable WARNING
    # log record and the "Error: Invalid server settings" line.
    assert "WARNING" not in completed.stderr
    assert "Error:" not in completed.stderr
    assert "Traceback" not in completed.stderr


def _refuse_to_serve(*args, **kwargs):
    # AssertionError, not SystemExit: ``pytest.raises(SystemExit)`` would
    # swallow the latter, and a real ``mcp.run`` would block on stdio.
    raise AssertionError(
        "main() reached mcp.run(); an in-process test must exit before serving"
    )


@pytest.fixture
def in_process_main(monkeypatch):
    """Let ``server.main()`` run in this process without ever serving.

    ``mcp.run`` becomes :func:`_refuse_to_serve`, and the process-wide
    ``_state`` is restored afterwards in case a regression gets as far as
    ``reset_state()``. The guard goes into the instance ``__dict__`` so that
    undoing it deletes the entry instead of pinning a bound method there.
    Every ``ANNEALBRIDGE_*`` variable is removed first, so a developer's own
    server settings cannot change the outcome; a test sets what it needs.

    ``server.main`` itself is replaced by a wrapper that snapshots the root
    logger's handlers and level just before calling the real ``main()`` and
    restores both in a ``finally``, re-raising whatever ``main()`` raised.
    ``main()`` configures logging with ``force=True``, which removes and
    closes every root handler — pytest's capture handlers included — so they
    must be put back. The snapshot and the restore both happen inside the
    wrapper, during the call phase: pytest attaches its capture handlers to
    the root logger only for the duration of each phase (setup, call,
    teardown), so restoring from this fixture's own setup and teardown would
    leave the setup phase's attachment on the root for good. File handlers
    (``--log-file``) are detached before the call rather than put back after
    it: a closed ``FileHandler`` opened with mode ``"w"`` never reopens, so
    every later record of the session would be lost. A closed stream
    handler keeps working, so the capture handlers stay attached and
    ``caplog`` still sees what ``main()`` logs before it configures logging."""
    for name in list(os.environ):
        if _is_server_setting(name):
            monkeypatch.delenv(name)
    monkeypatch.setitem(vars(server.mcp), "run", _refuse_to_serve)
    monkeypatch.setattr(server, "_state", server._state)

    real_main = server.main

    def main_restoring_the_root_logger():
        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level
        root.handlers[:] = [
            handler
            for handler in saved_handlers
            if not isinstance(handler, logging.FileHandler)
        ]
        try:
            return real_main()
        finally:
            root.handlers[:] = saved_handlers
            root.setLevel(saved_level)

    monkeypatch.setattr(server, "main", main_restoring_the_root_logger)


@pytest.mark.parametrize(
    ("arguments", "expected_argument", "expected_reason"),
    [
        (["--transport", "streamable-http", "--host", ""], "--host", HOST_REASON),
        (
            ["--transport", "streamable-http", "--host", "127.0.0.1 "],
            "--host",
            HOST_REASON,
        ),
        (["--transport", "streamable-http", "--port", "65536"], "--port", PORT_REASON),
        (["--transport", "streamable-http", "--port", "0"], "--port", PORT_REASON),
        (["--port", "-1"], "--port", PORT_REASON),
        (["--port", "not-a-port"], "--port", "port must be an integer"),
    ],
)
@pytest.mark.usefixtures("in_process_main")
def test_invalid_bind_argument_exits_2_before_binding_a_socket(
    arguments, expected_argument, expected_reason, monkeypatch, capsys
):
    """2026-09-11 review (F01 / F09): ``--host`` / ``--port`` are validated
    by argparse, so a bad value ends ``main()`` with exit code 2 and one line
    — no traceback, and no socket ever opened: the ``in_process_main`` guard
    fails the test should ``main()`` get as far as ``mcp.run``.

    In-process for speed; ``test_invalid_port_exits_2_in_a_real_process``
    keeps one case in a real process, where the exit code must also leave
    the process."""
    monkeypatch.setattr(sys, "argv", ["annealbridge-mcp", *arguments])

    with pytest.raises(SystemExit) as excinfo:
        server.main()

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert f"argument {expected_argument}: {expected_reason}" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.usefixtures("in_process_main")
def test_invalid_settings_exit_2_with_a_message_and_no_traceback(monkeypatch, capsys):
    """An invalid ``ANNEALBRIDGE_*`` value ends ``main()`` with exit code 2
    and an ``Error:`` line naming the variable, before the full argument
    parser runs (only ``--version`` is looked for first) and without serving
    (the ``in_process_main`` guard)."""
    monkeypatch.setenv("ANNEALBRIDGE_MAX_CONCURRENT_SOLVES", "0")
    monkeypatch.setattr(sys, "argv", ["annealbridge-mcp"])

    with pytest.raises(SystemExit) as excinfo:
        server.main()

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "Error: Invalid server settings" in captured.err
    assert "ANNEALBRIDGE_MAX_CONCURRENT_SOLVES" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "extra_env",
    [
        {},
        {
            "ANNEALBRIDGE_MAX_CONCURRENT_SOLVES": "0",
            "ANNEALBRIDGE_NOT_A_REAL_SETTING": "1",
        },
        {"ANNEALBRIDGE_NOT_A_REAL_SETTING": "1"},
    ],
    ids=["valid-settings", "invalid-settings", "unknown-variable"],
)
@pytest.mark.usefixtures("in_process_main")
def test_version_exits_0_before_the_settings_are_read(
    extra_env, monkeypatch, capsys, caplog
):
    """2026-09-15 consolidation: ``--version`` is answered before
    ``load_settings()``, so an invalid value cannot block it and an unknown
    variable is not even warned about — stdout carries the version, and
    neither a settings error nor the unknown-variable warning is produced.

    Had the settings been read, ``invalid-settings`` would end with
    ``Error: Invalid server settings`` and exit 2 (the invalid value is
    refused before unknown variables are checked), and ``unknown-variable``
    would log a ``WARNING`` record naming the variable.

    In-process for speed; the ``in_process_main`` guard fails the test
    should ``main()`` get past the version check as far as ``mcp.run``.
    ``test_version_exits_0_with_invalid_settings_in_a_real_process`` keeps a
    console-script run, where the real stderr is inspected."""
    for name, value in extra_env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["annealbridge-mcp", "--version"])
    # In-process, pytest intercepts logging: the unknown-variable WARNING
    # never reaches the stderr capsys sees, so only caplog can keep the
    # subprocess version's "no unknown-variable warning" assertion alive.
    caplog.set_level(logging.WARNING)

    with pytest.raises(SystemExit) as excinfo:
        server.main()

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == f"annealbridge {metadata.version('annealbridge')}\n"
    # The two traces load_settings() leaves: the unknown-variable WARNING
    # log record and the "Error: Invalid server settings" line.
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert "WARNING" not in captured.err
    assert "Error:" not in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.usefixtures("in_process_main")
def test_settings_refused_while_wiring_the_service_exit_2_with_a_message_and_no_traceback(
    monkeypatch, capsys
):
    """``load_settings()`` accepts every variable, but wiring the service from
    those settings raises ``SettingsError`` — a backend declaring a limit the
    policy has no value for, say. ``main()`` ends exactly as it does for an
    invalid variable: exit code 2, one ``Error:`` line, no traceback, and no
    serving (the ``in_process_main`` guard)."""

    def refuse_to_wire(settings=None):
        raise SettingsError(
            "Invalid server settings: backend 'acme' declares limit "
            "'iterations' but the policy has no value for it"
        )

    monkeypatch.setattr(server, "build_state", refuse_to_wire)
    monkeypatch.setattr(sys, "argv", ["annealbridge-mcp"])

    with pytest.raises(SystemExit) as excinfo:
        server.main()

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert (
        "Error: Invalid server settings: backend 'acme' declares limit 'iterations'"
        in captured.err
    )
    assert "Traceback" not in captured.err
    assert captured.out == ""


SERVER_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


@pytest.mark.usefixtures("in_process_main")
def test_server_logs_through_one_stderr_handler_in_its_own_format(monkeypatch, capsys):
    """By the time ``main()`` starts serving, the root logger has exactly one
    handler: a plain ``StreamHandler`` on stderr with the server's format, at
    ``INFO``.

    Importing the server module already configured the root logger: the SDK's
    ``MCPServer`` calls ``logging.basicConfig`` with a ``RichHandler`` on a
    stderr console. A ``basicConfig`` without ``force=True`` is a no-op once
    the root has any handler, so ``main()``'s own setup would silently never
    apply. No handler may write to stdout, which is the stdio transport's
    protocol channel.

    ``mcp.run`` records the root logger when ``main()`` reaches it and
    returns; afterwards the ``in_process_main`` wrapper must have restored the
    root logger as it was before the call."""
    seen = []

    def record_the_root_logger(*args, **kwargs):
        root = logging.getLogger()
        handlers = [
            (
                type(handler),
                getattr(handler, "stream", None),
                handler.formatter._fmt if handler.formatter else None,
            )
            for handler in root.handlers
        ]
        seen.append((handlers, root.level))

    monkeypatch.setitem(vars(server.mcp), "run", record_the_root_logger)
    monkeypatch.setattr(sys, "argv", ["annealbridge-mcp"])
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level
    # capsys has replaced sys.stderr; the handler must write to this object.
    stderr = sys.stderr

    server.main()

    assert len(seen) == 1
    handlers, level = seen[0]
    assert [
        stream for _, stream, _ in handlers if stream in (sys.stdout, sys.__stdout__)
    ] == []
    assert [kind for kind, _, _ in handlers if kind.__name__ == "RichHandler"] == []
    assert handlers == [(logging.StreamHandler, stderr, SERVER_LOG_FORMAT)]
    assert level == logging.INFO
    assert root.handlers == handlers_before
    assert root.level == level_before
    assert capsys.readouterr().out == ""


@pytest.mark.usefixtures("in_process_main")
def test_unknown_variable_warning_is_logged_in_the_server_format(monkeypatch, capsys):
    """The log setup runs before ``load_settings()``, so the unknown-variable
    ``WARNING`` it logs already goes to stderr through the server's handler,
    in the server's format, before serving starts."""
    monkeypatch.setenv("ANNEALBRIDGE_NOT_A_REAL_SETTING", "1")
    stderr_when_serving = []

    def record_stderr(*args, **kwargs):
        stderr_when_serving.append(capsys.readouterr().err)

    monkeypatch.setitem(vars(server.mcp), "run", record_stderr)
    monkeypatch.setattr(sys, "argv", ["annealbridge-mcp"])

    server.main()

    assert len(stderr_when_serving) == 1
    assert (
        "WARNING annealbridge.config.settings: Ignoring unknown ANNEALBRIDGE_* "
        "variable(s): ANNEALBRIDGE_NOT_A_REAL_SETTING"
    ) in stderr_when_serving[0]
    assert capsys.readouterr().out == ""

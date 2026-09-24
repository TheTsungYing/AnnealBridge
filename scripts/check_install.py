"""Release acceptance probe: does an *installed* AnnealBridge actually work?

Run it with the Python of a clean virtual environment that has the built wheel
installed — never with the development environment. It answers the one
question the test suite cannot: whether what pip installs (the packaged
modules, the generated console scripts, the optional-extra boundary) behaves
like the source tree it was built from.

Two usages, matching the two environments CI builds:

    python scripts/check_install.py --examples examples          # core-only venv
    python scripts/check_install.py --examples examples --mcp    # "[mcp]" venv

Without ``--mcp`` the probe also asserts the extras are genuinely absent (no
``mcp``, no ``dwave-system``) and that ``annealbridge-mcp`` reports the missing
extra instead of raising. With ``--mcp`` it drives the installed
``annealbridge-mcp`` console script over stdio through a real MCP client.

Why this lives in ``scripts/`` and not in ``tests/``:

* Default ``pytest`` collects ``tests/``. This probe must *not* run against the
  editable development install — every check here would be meaningless or
  actively wrong there (check 1 asserts the package is not served from
  ``src/``, check 8 asserts the extras are missing).
* It verifies the *artifact*, not the source: it needs its own interpreter, its
  own working directory outside the checkout, and a scrubbed environment. That
  is a release step, not a unit of the test suite.

Standard library only, plus the installed ``annealbridge`` (and ``mcp`` under
``--mcp``) — the clean environment has nothing else, by design.

2026-09-11 install verification, gap 4.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.metadata as metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from contextlib import contextmanager
from pathlib import Path

EXAMPLE_FILES = ("knapsack.json", "integer_knapsack.json")
# The examples the wheel ships as package data, in the order
# ``annealbridge example`` lists them.
BUNDLED_EXAMPLES = [
    "knapsack",
    "integer_knapsack",
    "assignment",
    "tsp",
    "shift_scheduling",
]
MCP_TOOLS = [
    "get_optimization_capabilities",
    "recommend_backend",
    "solve_optimization",
    "validate_optimization_problem",
]

checks: list[str] = []


class CheckFailed(AssertionError):
    """A named check that did not hold. Carries the name so the top level can
    print ``FAIL <name>`` without guessing at context from a traceback."""

    def __init__(self, name: str, detail: str) -> None:
        super().__init__(f"{name}: {detail}")
        self.name = name
        self.detail = detail


def passed(name: str) -> None:
    checks.append(name)
    print("PASS", name, flush=True)


@contextmanager
def check(name: str):
    """Run one check. Report ``PASS name`` on success; turn any failure into a
    ``CheckFailed`` tagged with the same name."""
    try:
        yield
    except CheckFailed:
        raise
    except Exception as exc:  # noqa: BLE001 - every failure is a failed check
        raise CheckFailed(name, f"{type(exc).__name__}: {exc}") from exc
    passed(name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="check_install.py",
        description=(
            "Verify an installed AnnealBridge wheel from a clean virtual "
            "environment."
        ),
    )
    parser.add_argument(
        "--examples",
        required=True,
        metavar="DIR",
        help=(
            "directory holding the example problems "
            f"({' and '.join(EXAMPLE_FILES)})"
        ),
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help='also run the MCP checks (environment installed with the "[mcp]" extra)',
    )
    return parser.parse_args(argv)


def resolve_examples(directory: str) -> list[Path]:
    """Absolute paths of the example problems, before any ``chdir``."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise CheckFailed(
            "examples directory",
            f"{root} is not a directory; pass --examples with the directory "
            f"holding {' and '.join(EXAMPLE_FILES)}",
        )
    resolved = []
    for name in EXAMPLE_FILES:
        path = root / name
        if not path.is_file():
            raise CheckFailed(
                "examples directory", f"{path} is missing from --examples {root}"
            )
        resolved.append(path)
    return resolved


def scrub_environment() -> None:
    """Drop developer configuration and vendor credentials, then pin the two
    settings the probe depends on. Nothing about the environment is printed."""
    for key in list(os.environ):
        if key.startswith(("ANNEALBRIDGE_", "DWAVE_", "FUJITSU_")) or key == "PYTHONPATH":
            del os.environ[key]
    os.environ["ANNEALBRIDGE_ALLOW_REMOTE"] = "false"
    os.environ["ANNEALBRIDGE_ALLOW_REMOTE_RETRIES"] = "false"


def console_script(name: str) -> str:
    """Locate a console script generated by the install, without assuming a
    platform layout: ``sysconfig`` knows where this interpreter puts scripts,
    and ``shutil.which`` applies the platform's executable suffixes."""
    scripts_dir = sysconfig.get_path("scripts")
    found = shutil.which(name, path=scripts_dir)
    if found is None:
        raise CheckFailed(
            f"locate the {name} console script",
            f"not found in {scripts_dir}; install the wheel into the "
            "environment running this probe",
        )
    return found


def run(command: list[str], *, code: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", timeout=120
    )
    assert result.returncode == code, (
        f"{command[0]} {' '.join(command[1:])} exited {result.returncode}, "
        f"expected {code}; stderr: {result.stderr.strip()}"
    )
    return result


def run_bytes(
    command: list[str], *, stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    """``run`` without text decoding: stdout exactly as the process wrote it,
    so no newline translation hides a byte the command added or dropped."""
    result = subprocess.run(command, input=stdin, capture_output=True, timeout=120)
    assert result.returncode == 0, (
        f"{command[0]} {' '.join(command[1:])} exited {result.returncode}, "
        f"expected 0; stderr: {result.stderr.decode('utf-8', 'replace').strip()}"
    )
    return result


def verify_solution(result: dict, expected: float | None = None) -> None:
    assert result["status"] == "success", f"status was {result['status']}: {result}"
    assert result["solutions"], "no solutions returned"
    assert all(
        solution["hard_constraints_satisfied"] for solution in result["solutions"]
    ), "a returned solution violates a hard constraint"
    if expected is not None:
        actual = result["solutions"][0]["objective_value"]
        assert actual == expected, f"objective_value was {actual}, expected {expected}"


def run_common_checks(cli: str) -> str:
    """The checks that hold in every installed environment. Returns the
    installed package path for the report."""

    def json_cli(*args: str) -> dict:
        return json.loads(run([cli, *args]).stdout)

    import annealbridge

    package_path = annealbridge.__file__
    with check("imports come from the isolated installed package"):
        location = Path(package_path).resolve()
        assert sys.prefix != sys.base_prefix, (
            "this interpreter is not a virtual environment; run the probe with "
            "the Python of a clean venv that has the wheel installed"
        )
        assert location.is_relative_to(Path(sys.prefix).resolve()), (
            f"annealbridge is served from {location}, which is outside "
            f"{sys.prefix}"
        )
        # An editable install points back at the checkout's ``src/annealbridge``;
        # a real wheel install never has a ``src`` directory on its path.
        assert "src" not in location.parts, (
            f"{location} runs through a 'src' directory, so this looks like an "
            "editable install of the checkout rather than an installed wheel"
        )

    with check("CLI schema export"):
        assert "properties" in json_cli("export-schema")

    with check("CLI capabilities --json is UTF-8 (its descriptions are not ASCII)"):
        assert json_cli("capabilities", "--json")["backends"]

    with check("CLI validate"):
        assert json_cli("validate", "knapsack.json", "--json")["valid"]

    with check("CLI validate reads the problem from stdin"):
        piped = run_bytes(
            [cli, "validate", "-", "--json"],
            stdin=Path("knapsack.json").read_bytes(),
        )
        assert json.loads(piped.stdout)["valid"]

    with check("CLI example lists the shipped examples"):
        listed = run([cli, "example"]).stdout.splitlines()
        assert [line.split()[0] for line in listed] == BUNDLED_EXAMPLES, listed

    with check("CLI example prints the shipped file byte for byte"):
        printed = run_bytes([cli, "example", "knapsack"]).stdout
        assert printed == Path("knapsack.json").read_bytes(), (
            "annealbridge example knapsack differs from the shipped knapsack.json"
        )

    with check("CLI recommend"):
        recommendations = json_cli("recommend", "knapsack.json", "--json")
        assert recommendations["recommendations"][0]["usable"]

    with check("CLI exact binary optimum = 17"):
        verify_solution(json_cli("solve", "knapsack.json", "--json"), 17)

    with check("CLI exact integer + soft constraint optimum = 34"):
        verify_solution(json_cli("solve", "integer_knapsack.json", "--json"), 34)

    with check("CLI simulated annealing returns feasible solutions"):
        verify_solution(
            json_cli("solve", "knapsack.json", "--backend", "simulated_annealing", "--json")
        )

    return package_path


def run_core_only_checks() -> None:
    """What must be *absent*, and how the MCP script behaves without its extra."""
    with check("core install has neither the MCP nor the D-Wave SDK"):
        assert importlib.util.find_spec("mcp") is None, "the mcp package is installed"
        installed = {
            distribution.metadata["Name"].lower()
            for distribution in metadata.distributions()
        }
        assert "dwave-system" not in installed, "dwave-system is installed"

    with check("annealbridge-mcp reports the missing extra instead of raising"):
        result = run([console_script("annealbridge-mcp"), "--help"], code=2)
        assert "annealbridge[mcp]" in result.stderr, (
            "stderr does not name the extra to install: " f"{result.stderr.strip()}"
        )
        assert "Traceback" not in result.stderr, "stderr carries a traceback"
        assert result.stdout == "", f"stdout was not empty: {result.stdout!r}"


async def run_mcp_checks(problems: dict[str, dict]) -> None:
    from mcp import Client, StdioServerParameters

    params = StdioServerParameters(
        command=console_script("annealbridge-mcp"), args=[], env=dict(os.environ)
    )
    async with asyncio.timeout(120):
        async with Client(params) as client:

            async def call(
                name: str,
                problem: dict | None = None,
                *,
                arguments: dict | None = None,
            ) -> dict:
                if arguments is None:
                    arguments = {} if problem is None else {"problem": problem}
                result = await client.call_tool(name, arguments)
                assert not result.is_error, f"{name} returned a tool error: {result}"
                return result.structured_content

            with check("MCP installed script handshake, metadata and four tools"):
                tools = (await client.list_tools()).tools
                assert sorted(tool.name for tool in tools) == MCP_TOOLS
                assert client.server_info.version == metadata.version("annealbridge")
                assert "When to call each tool" in client.instructions

            with check("MCP prompts and resources"):
                listed_prompts = (await client.list_prompts()).prompts
                assert sorted(prompt.name for prompt in listed_prompts) == [
                    "assign",
                    "pick_subset",
                    "schedule_shifts",
                ]
                listed_resources = (await client.list_resources()).resources
                assert sorted(str(resource.uri) for resource in listed_resources) == [
                    "annealbridge://examples/assignment",
                    "annealbridge://examples/integer_knapsack",
                    "annealbridge://examples/knapsack",
                    "annealbridge://examples/shift_scheduling",
                    "annealbridge://examples/tsp",
                    "annealbridge://schema",
                ]
                # The examples are package data; the wheel must actually ship
                # them, and what it serves must parse to the file this probe
                # copied in (the byte-for-byte guard is a unit test).
                served = await client.read_resource("annealbridge://examples/knapsack")
                assert (
                    json.loads(served.contents[0].text) == problems["knapsack.json"]
                ), "the packaged knapsack example differs from the shipped one"
                schema = await client.read_resource("annealbridge://schema")
                assert "properties" in json.loads(schema.contents[0].text)

            with check("MCP capabilities"):
                # The schema only comes back when it is asked for; the plain
                # call must still carry the field, explicitly null.
                requested = await call(
                    "get_optimization_capabilities",
                    arguments={"include_schema": True},
                )
                assert requested["problem_json_schema"]["title"] == "OptimizationProblem"
                assert (await call("get_optimization_capabilities"))[
                    "problem_json_schema"
                ] is None

            knapsack = problems["knapsack.json"]
            with check("MCP validate"):
                assert (await call("validate_optimization_problem", knapsack))["valid"]

            with check("MCP recommend"):
                recommended = await call("recommend_backend", knapsack)
                assert recommended["recommendations"][0]["usable"]

            with check("MCP exact binary optimum = 17"):
                verify_solution(await call("solve_optimization", knapsack), 17)

            with check("MCP exact integer + soft constraint optimum = 34"):
                verify_solution(
                    await call("solve_optimization", problems["integer_knapsack.json"]),
                    34,
                )

            with check("MCP simulated annealing returns feasible solutions"):
                annealed = copy.deepcopy(knapsack)
                annealed["solver"] = {
                    "backend": "simulated_annealing",
                    "seed": 1234,
                    "num_reads": 100,
                }
                verify_solution(await call("solve_optimization", annealed))

            with check("MCP unknown field returned as structured invalid_problem"):
                unknown = copy.deepcopy(knapsack)
                unknown["unexpected_field"] = 1
                rejected = await call("solve_optimization", unknown)
                assert rejected["status"] == "invalid_problem", rejected["status"]
                assert "UNKNOWN_FIELD" in [
                    error["code"] for error in rejected["errors"]
                ], "UNKNOWN_FIELD is not among the reported errors"
                assert not rejected["solutions"], "an unknown field still returned solutions"

            with check("MCP semantic error returned as structured invalid_problem"):
                ghost = copy.deepcopy(knapsack)
                ghost["objective"]["linear_terms"][0]["variable"] = "ghost"
                assert (await call("solve_optimization", ghost))[
                    "status"
                ] == "invalid_problem"

            with check("MCP disabled remote rejected without fallback"):
                remote = copy.deepcopy(knapsack)
                remote["solver"] = {"backend": "dwave_qpu"}
                refused = await call("solve_optimization", remote)
                assert refused["status"] == "backend_unavailable", refused["status"]
                assert "REMOTE_DISABLED" in [
                    error["code"] for error in refused["errors"]
                ], "REMOTE_DISABLED is not among the reported errors"
                assert not refused["solutions"], "a disabled backend still returned solutions"


def report(package_path: str) -> None:
    payload = {
        "python": sys.version,
        "package_path": package_path,
        "checks": checks,
        "packages": {
            distribution.metadata["Name"]: distribution.version
            for distribution in metadata.distributions()
        },
    }
    print("REPORT_JSON " + json.dumps(payload), flush=True)
    print(f"{len(checks)} checks passed", flush=True)


def probe(args: argparse.Namespace) -> None:
    examples = resolve_examples(args.examples)
    origin = Path.cwd()
    # Work outside the checkout: no ``src/`` to import by accident, and no
    # ``examples/`` reachable by a relative path the probe did not create.
    workspace = Path(tempfile.mkdtemp(prefix="annealbridge-install-"))
    try:
        problems = {}
        for source in examples:
            shutil.copy2(source, workspace / source.name)
            problems[source.name] = json.loads(source.read_text(encoding="utf-8"))
        os.chdir(workspace)
        scrub_environment()

        package_path = run_common_checks(console_script("annealbridge"))
        if args.mcp:
            asyncio.run(run_mcp_checks(problems))
        else:
            run_core_only_checks()
        report(package_path)
    finally:
        os.chdir(origin)
        shutil.rmtree(workspace, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        probe(args)
    except CheckFailed as failure:
        print(f"FAIL {failure.name}", file=sys.stderr, flush=True)
        print(failure.detail, file=sys.stderr, flush=True)
        return 1
    except Exception as exc:  # noqa: BLE001 - one line, never a traceback
        print("FAIL install probe", file=sys.stderr, flush=True)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

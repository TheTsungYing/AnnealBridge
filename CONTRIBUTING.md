# Contributing to AnnealBridge

Thanks for taking the time. This page covers the development setup, how to run
the tests, the architectural rules the test suite enforces, and what a good
pull request looks like. Everyone taking part is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/OWNER/AnnealBridge.git
cd AnnealBridge
python -m venv .venv
# Windows: .venv\Scripts\activate  —  macOS / Linux: source .venv/bin/activate
pip install -e ".[all,dev]"
```

`[all]` pulls in the `mcp` SDK and `dwave-system`; `[dev]` adds `pytest`,
`anyio` and the MCP CLI used by the Inspector. The core package works with no
extras at all, and CI checks that too (see
[docs/testing.md](docs/testing.md)).

## Running the tests

```bash
pytest
```

The default run excludes the live remote tests (`-m "not remote"` is set in
`pyproject.toml`), so it never touches the network and never consumes vendor
quota. The suite is expected to pass with **no skip and no xfail**.

```bash
pytest -m remote        # CONSUMES REAL VENDOR QUOTA
```

Live tests skip individually unless their credential is present
(`DWAVE_API_TOKEN` for the three D-Wave tests, `FUJITSU_DA_API_KEY` for the
Fujitsu one). Do not run them casually.

## Checking the built package

`pytest` runs against the checkout. To reproduce what CI's *Built distribution*
job does — install the actual wheel into a clean environment and drive it
through its real console scripts — build the distribution and point the probe
at it:

```bash
pip install build
python -m build
python -m venv /tmp/ab-core && /tmp/ab-core/bin/python -m pip install dist/*.whl
/tmp/ab-core/bin/python -m pip check
/tmp/ab-core/bin/python scripts/check_install.py --examples examples
```

And the same for the `[mcp]` extra, which adds an MCP stdio handshake and a
tool call to the checks:

```bash
python -m venv /tmp/ab-mcp
/tmp/ab-mcp/bin/python -m pip install "dist/annealbridge-<version>-py3-none-any.whl[mcp]"
/tmp/ab-mcp/bin/python -m pip check
/tmp/ab-mcp/bin/python scripts/check_install.py --examples examples --mcp
```

These commands work as written on macOS, Linux and Windows Git Bash; in Windows
PowerShell the interpreter is `Scripts\python.exe` instead of `bin/python`.

`scripts/check_install.py` copies the example problems to a temporary directory
and runs everything from there, so a missing packaged file cannot be masked by
the checkout. It only uses the local solvers, leaves remote execution disabled
by policy, and consumes no vendor quota.

## Architecture rules (enforced by tests)

The package is a self-contained **core** plus thin **interfaces**. The
dependency direction is fixed:

```text
models ← validation ← penalty ← compiler ← solvers ← orchestration ← CLI / MCP
```

`tests/architecture/` fails the build if a change:

- imports `mcp`, `dwave.cloud`, `annealbridge.config` or
  `annealbridge.interfaces` from `models/`, `validation/`, `compiler/`,
  `penalty/`, `solvers/` or `orchestration/`;
- makes `validation/` import a higher layer, or `compiler/` import `penalty/`;
- hard-codes a backend name (`if backend == "..."`) in `orchestration/`,
  `validation/` or the interfaces;
- imports a third-party HTTP client anywhere in the core;
- requires a pipeline change to run a new backend (a fake extra backend is
  plugged in during the test and the service, validator and interfaces must
  work unchanged).

Read [docs/architecture.md](docs/architecture.md) before touching anything
below `interfaces/`.

## Design principles (the short version)

These are the lines the project does not cross. A pull request that crosses
one needs a very good reason in its description.

1. **The agent says *what*, the code decides *how*.** The problem JSON never
   contains a QUBO matrix, a penalty weight, a slack variable, or an integer
   encoding. Those are produced by the compiler.
2. **Never trust solver energy.** Feasibility and objective values are
   recomputed from the original problem for every candidate. Energy is kept
   for debugging only.
3. **Hard penalties and soft weights are different things.** The hard penalty
   is computed by the penalty strategy; the soft weight is supplied by the
   agent in objective units. They never share a field or derive from one
   another.
4. **Backends are replaceable parts.** The core knows only the `SolverBackend`
   protocol and what each backend declares about itself.
5. **No silent decisions.** An unavailable backend returns
   `backend_unavailable`, never a fallback. An over-limit parameter returns
   `resource_limit_exceeded`, never a clamped value.
6. **Each layer does one job.** No optimization logic in the CLI or MCP
   layer, no JSON parsing in a solver, no mutation in a validator.
7. **No speculative abstractions.** Only `ModelCompiler`, `SolverBackend`
   and `PenaltyStrategy` are interfaces; everything else is concrete code.
8. **Tests do not cheat.** No `skip` / `xfail` to hide a bug; scenario tests
   go through the full JSON → service → result path; MCP tests go through a
   real MCP client.

## Adding a backend

Implement the `SolverBackend` protocol in a new module under `solvers/`,
declare its `SolverCapabilities` (model types, limits, and the credential
environment variables / header names so redaction covers it automatically),
and register it in the registry. Add a mock-based test under
`tests/remote_mock/` (or `tests/unit/` for a local backend) and, if it is a
paid remote service, an opt-in live test under `tests/remote_live/`. The only
expected touch outside `solvers/` is adding the name (and any option block)
to `SolverPreferences` in `models/problem.py`; the service, validator and
interfaces must not change. If they have to, open an issue first. See
[docs/backends.md](docs/backends.md).

## Documentation

User-facing behaviour lives in `README.md` and the pages under `docs/`. If a
change adds or renames an error or warning code, an environment variable, a
CLI option, or an MCP tool field, update the matching page in the same pull
request and add a line to `CHANGELOG.md` under *Unreleased*.

## Pull requests

- Keep one logical change per pull request.
- Commit messages follow `type(scope): summary`, for example
  `fix(compiler): keep slack range exact for negative lower bounds`. Common
  types: `feat`, `fix`, `refactor`, `docs`, `test`, `perf`, `chore`.
- Fill in the pull request template; the checklist is short and every item
  matters.
- Never commit credentials, `dwave.conf`, `.env` files, or recorded vendor
  responses that contain account identifiers.

## Reporting security issues

See [SECURITY.md](SECURITY.md). Please do not file security problems as
public issues.

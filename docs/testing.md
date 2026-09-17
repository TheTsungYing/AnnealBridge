[← Back to README](../README.md) · [Documentation index](README.md)

# Testing and CI

## Running the suite

```bash
pytest
```

That is the whole default run. It never executes a live remote test, never
reaches beyond the local machine and never consumes vendor quota:
`pyproject.toml` sets `addopts = -m "not remote"`, so the opt-in live tests are
deselected before collection finishes. The one test module that makes an HTTP request or runs a listening server,
`tests/remote_mock/test_fujitsu_transport.py`, talks only to a loopback HTTP
server it starts on `127.0.0.1`, with the proxy variables cleared so the
request cannot be routed anywhere else.

The full default suite runs with **no skip and no xfail**. The only skips the
project allows anywhere are the opt-in live tests below.

## Live remote tests

```bash
pytest -m remote
```

**This consumes real vendor quota.** `-m remote` overrides the default
deselection, and the live tests then talk to real hardware.

Each live test still skips unless its own credential is present. Which
credential that is comes from the test module itself — a module declares
`REQUIRED_ENV` at import time, and modules that declare none fall back to
`DWAVE_API_TOKEN`:

| Live test | Credential |
| --- | --- |
| `tests/remote_live/test_dwave_qpu_live.py` | `DWAVE_API_TOKEN` |
| `tests/remote_live/test_leap_hybrid_live.py` | `DWAVE_API_TOKEN` |
| `tests/remote_live/test_leap_hybrid_cqm_live.py` | `DWAVE_API_TOKEN` |
| `tests/remote_live/test_fujitsu_da_live.py` | `FUJITSU_DA_API_KEY` |

The Fujitsu live test is a minimal knapsack with a 1 s time limit, the smallest
the schema allows. An unset variable — or one set to the empty string, which is
what an undefined CI secret expands to — counts as missing, and the test skips
rather than fails.

The rest of the suite is insulated from your machine: an autouse fixture
removes every declared vendor credential variable from the environment, so a
developer with `DWAVE_API_TOKEN` exported still sees the same availability,
recommendations and redaction results as CI. The live directory overrides that
fixture with a no-op — it is the one place that needs the real credentials.

## Test directories

| Directory | What it covers |
| --- | --- |
| `tests/unit/` | Models, validators, the bounds-aware size estimates (checked against brute-force enumeration), slack and integer encoding, both compilers including their integer paths, the `decode` step, the penalty strategy, the policy limits, backend routing, candidate arrays over integer rows, and each solver backend |
| `tests/scenarios/` | The full JSON → validate → compile → solve → re-validate → rank pipeline on the shipped examples, including the integer knapsack down all four paths (exact, simulated annealing, tabu, a fake CQM backend) reaching the same optimum |
| `tests/mcp/` | The four MCP tools driven through an in-memory MCP client, plus the tool list, the generated schemas, the thread offload and the stdio entry point (a few real subprocess checks; the rest of its argument and settings handling in-process) |
| `tests/remote_mock/` | The D-Wave backends against mocked samplers (with the contract all three Ocean backends share written once, in `ocean_contract.py`) and the Fujitsu backend against a scripted HTTP transport — request shape, polling, delete / cancel, every error mapping — plus the real `UrllibTransport` against a loopback server and the credential-leak suite |
| `tests/remote_live/` | Opt-in tests against real vendor hardware (see above) |
| `tests/architecture/` | The import boundaries, the no-backend-name rule, the no-third-party-HTTP rule and the fifth-backend rule, all described in [Architecture](architecture.md#enforced-boundaries) |
| `tests/golden/` | The recorded snapshot the golden test compares against, and the script that recorded it |
| `tests/fakes/` | Shared test doubles: a declared fake backend, a local CQM backend and a scripted Fujitsu transport |

## The golden test

Before integer variables were added, the compiled BQM and CQM, the size
estimates, the penalty scale and the full validation output of a fixed list of
`version: "1.0"` problems were recorded to a snapshot pinned in
`tests/golden/`. `tests/unit/test_golden_phase3a.py` replays that list and
asserts the current code still produces exactly those numbers.

That is what makes "every `1.0` document keeps its exact `1.0` behaviour" a
checked claim rather than an intention: any later change — integer variables,
bounds-aware estimates, the CQM integer path — is proven bit-for-bit neutral
for existing documents. The comparison goes through a JSON round-trip on both
sides so tuples and lists compare alike, and floats survive that round-trip
exactly. The test also asserts that the recorded problem list and the script's
list still describe the same problems, so a problem added on one side without
re-recording the other is a failure rather than a silent gap.

## Rules the suite holds itself to

- **No `skip`, no `xfail`.** Neither may be used to hide a real bug. The one
  exception is the live-test opt-in guard, which prevents accidental paid-quota
  consumption rather than hiding a failure.
- **Scenario tests run the whole pipeline.** A scenario starts from problem
  JSON and goes through `OptimizationService`; calling an internal function
  directly and labelling it an integration test does not count.
- **MCP tests go through a real client.** The four tools are exercised over an
  actual MCP client session, not by calling the Python functions underneath.

## Continuous integration

Two workflows live in `.github/workflows/`.

### `ci.yml` — every push and pull request

Runs on `push` and `pull_request`. Three jobs:

- **Test** — installs `.[all,dev]` on Python 3.11 and 3.12 (matrix, without
  fail-fast) and runs a plain `pytest`. No `-m` flag is passed, so the
  project's own `-m "not remote"` applies: unit, scenarios, architecture, MCP
  and remote-mock tests all run, and no vendor quota is touched. The workflow
  therefore needs neither `DWAVE_API_TOKEN` nor `FUJITSU_DA_API_KEY`. Before
  `pytest`, the job runs `ruff check .`, which checks only the F (Pyflakes)
  and I (import sorting) rules; there is no formatter check. A lint failure
  fails the job, but `pytest` still runs.
- **Minimal install** — installs the package with *no* extras and checks that
  the core still stands on its own: the solver registry imports and lists its
  backends, `annealbridge export-schema` works, and `annealbridge validate` /
  `annealbridge recommend` run against a shipped example. This is the guard
  that keeps the optional dependencies genuinely optional.
- **Built distribution** — the only job that tests what a `pip` user receives
  rather than the checkout. It builds the sdist and then the wheel from that
  sdist (`python -m build`), and on both `ubuntu-latest` and `windows-latest`
  creates two clean virtual environments: one with the plain wheel, one with
  `wheel[mcp]`. Each gets a `pip check`, then runs
  `scripts/check_install.py` — a probe that copies the example problems to a
  temporary directory outside the checkout and drives the installed `annealbridge`
  and `annealbridge-mcp` **console scripts** from there, so nothing in the
  repository can satisfy an import or a data file by accident. The core
  environment also asserts that `annealbridge-mcp --help` exits 2 with an
  install hint; the `[mcp]` environment adds a stdio handshake and a real tool
  call. Where Minimal install proves the extras are genuinely optional, this
  job proves the distribution itself is complete and usable once installed.

### `remote-live.yml` — manual only

Triggered by `workflow_dispatch` and nothing else. There is deliberately no
`push`, `pull_request` or `schedule` trigger, because this workflow really does
connect to D-Wave Leap and the Fujitsu Digital Annealer and spend quota.

Secret handling is deliberately narrow:

- `DWAVE_API_TOKEN` and `FUJITSU_DA_API_KEY` are injected **only into the
  single `pytest` step**, never at job or workflow level, keeping the exposure
  surface as small as it can be.
- A secret that has not been created — `FUJITSU_DA_API_KEY` on a repository
  with no Fujitsu account, say — is injected by GitHub as an empty string. The
  live-test guard and the backend both treat an empty value as missing, so the
  corresponding test skips instead of failing.
- A pull request from a fork cannot read this repository's secrets, and
  `workflow_dispatch` can only be started by someone with permission on the
  repository itself, so a fork cannot launch this workflow at all.

Neither workflow contains a `printenv`, an environment dump, an `echo` of a
secret or any other debug step that would print the environment. Keep it that
way when editing them.

See [CONTRIBUTING.md](../CONTRIBUTING.md) before opening a pull request.

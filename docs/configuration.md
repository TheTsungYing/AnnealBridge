[← Back to README](../README.md) · [Documentation index](README.md)

# Configuration

This page is the complete reference for AnnealBridge's configuration. Every
server setting is an environment variable with the `ANNEALBRIDGE_` prefix;
vendor credentials deliberately are **not**. The same variables configure the
CLI and the MCP server, because both build the service through the same
composition root.

Settings are read when the process wires up its service, turned into an
`ExecutionPolicy`, and enforced on every solve. Nothing here is read per
request, and no setting can be changed by a problem JSON.

## Server settings (`ANNEALBRIDGE_*`)

| Variable | Type | Default | Meaning |
| --- | --- | --- | --- |
| `ANNEALBRIDGE_ALLOW_REMOTE` | bool | `false` | Enable the remote backends (D-Wave and Fujitsu). While `false`, a request for one returns `backend_unavailable` / `REMOTE_DISABLED` and nothing touches the network |
| `ANNEALBRIDGE_ALLOW_REMOTE_RETRIES` | bool | `false` | Allow penalty retries on remote backends. Still bounded by `MAX_REMOTE_RETRIES` |
| `ANNEALBRIDGE_EXACT_MAX_VARIABLES` | int ≥ 1 | `24` | Compiled-variable ceiling for the `exact` backend, slack and integer-encoding bits included (`EXACT_VARIABLE_LIMIT`) |
| `ANNEALBRIDGE_MAX_QPU_READS` | int ≥ 1 | `1000` | Upper bound on `num_reads` for `dwave_qpu` (`QPU_READS_LIMIT`) |
| `ANNEALBRIDGE_MAX_QPU_ANNEALING_TIME_US` | float > 0 | `2000.0` | Upper bound on `dwave_qpu.annealing_time_us`, in microseconds (`QPU_ANNEALING_TIME_LIMIT`) |
| `ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS` | int ≥ 1 | `300` | Upper bound on the effective remote time limit; shared by `leap_hybrid_bqm`, `leap_hybrid_cqm` and `fujitsu_da` (`REMOTE_TIME_LIMIT`) |
| `ANNEALBRIDGE_MAX_CONCURRENT_SOLVES` | int ≥ 1 | `4` | Concurrent solves allowed (`CONCURRENCY_LIMIT`, retryable). Multiplied by `ANNEALBRIDGE_SA_WORKERS` it bounds how many sampling threads can run at once, so tune the two together under a container CPU quota — see [Recommended production settings](#recommended-production-settings) |
| `ANNEALBRIDGE_MAX_LOCAL_READS` | int ≥ 1 | `100000` | Upper bound on `num_reads` for `simulated_annealing` (`LOCAL_READS_LIMIT`) |
| `ANNEALBRIDGE_MAX_SWEEPS` | int ≥ 1 | `100000` | Upper bound on `num_sweeps` for `simulated_annealing` (`SWEEPS_LIMIT`) |
| `ANNEALBRIDGE_MAX_LOCAL_RETRIES` | int ≥ 0 | `10` | Upper bound on `max_retries` for local backends (`RETRY_LIMIT`) |
| `ANNEALBRIDGE_MAX_REMOTE_RETRIES` | int ≥ 0 | `3` | Upper bound on `max_retries` for remote backends, enforced even when remote retries are enabled (`RETRY_LIMIT`) |
| `ANNEALBRIDGE_MAX_TOP_K` | int ≥ 1 | `1000` | Upper bound on `top_k` (`TOP_K_LIMIT`) |
| `ANNEALBRIDGE_SA_WORKERS` | int ≥ 1 | unset (auto-detect) | Threads the `simulated_annealing` backend samples with. Changes wall time only, never a result: the same seed gives the same answer for any value. Unset uses the CPUs available to the process; a container CPU quota is not detected, so set it there. Its product with `ANNEALBRIDGE_MAX_CONCURRENT_SOLVES` is the ceiling on simultaneous sampling threads, so tune the two together — see [Recommended production settings](#recommended-production-settings) |
| `ANNEALBRIDGE_ENABLED_BACKENDS` | comma-separated names | unset | Registry names allowed to run. Unset or empty means every registered backend |
| `ANNEALBRIDGE_LIMITS` | JSON object | `{}` | Generic policy limits for backends that declare custom limit keys. Not needed by the built-in backends |
| `ANNEALBRIDGE_HTTP_HOST` | non-empty str, no whitespace | `127.0.0.1` | Default bind host for the MCP streamable-http transport. An empty or blank value is a configuration error, never a request to bind every interface |
| `ANNEALBRIDGE_HTTP_PORT` | int 1–65535 | `8000` | Default port for the MCP streamable-http transport. `0` is refused: an ephemeral port no host can be pointed at |

The two retry ceilings allow `0`, which still admits `max_retries: 0`. Every
other ceiling must be positive: a zero or negative limit would reject every
solve (or, for `MAX_CONCURRENT_SOLVES`, break the semaphore the service builds
from it), so such a value is a configuration error.

**Limits are enforced as errors, never silently clamped.** A request above a
ceiling is refused with `resource_limit_exceeded` and the code named above,
before anything is submitted — so an over-limit remote request costs zero
vendor calls and zero quota. The one number that is *raised* rather than
refused is a hybrid solver's time limit below the sampler's own minimum; that
floor comes from the vendor, not from this configuration, and the effective
value is reported in `metadata.effective_time_limit_seconds`.

`annealbridge capabilities` prints the limits each backend is actually checked
against under the current environment — the same values an agent reads from
`get_optimization_capabilities`. See [CLI](cli.md#capabilities).

### `ANNEALBRIDGE_ENABLED_BACKENDS`

A comma-separated allow-list of **registry names** — the same names that
appear in `solver.backend`, in `annealbridge capabilities`, and in
`get_optimization_capabilities`. Only the six built-in names are accepted
there: `solver.backend` is a closed schema, so a backend must be registered
under its own `capabilities.name` for any request to name it:

```bash
export ANNEALBRIDGE_ENABLED_BACKENDS="exact,simulated_annealing"
```

- Whitespace around each name is stripped and blank entries are dropped.
- Unset, or a value that reduces to nothing, means **every registered
  backend** — the variable is not a way to disable everything.
- A backend outside the list is refused with `backend_unavailable` /
  `BACKEND_DISABLED_BY_POLICY`, and is reported as not enabled in the
  capabilities view, so an agent is not steered towards it.

Note that a remote backend the policy will not call is reported as *not
enabled* regardless of this list: with `ANNEALBRIDGE_ALLOW_REMOTE=false`, all
four remote backends show `Enabled: no`.

### `ANNEALBRIDGE_LIMITS`

A JSON object of *generic* limit keys, for third-party backends that declare
custom limit keys in their `SolverCapabilities.parameter_limits`:

```bash
export ANNEALBRIDGE_LIMITS='{"iterations": 100000}'
```

- Every value must be a finite number greater than 0. A non-finite or
  non-positive value would either reject every solve or defeat the
  `value > limit` comparison, so it is rejected here.
- The six built-in backends need none of this: they use the dedicated
  variables above. (`fujitsu_da` shares
  `ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS` with the hybrid solvers and declares
  no limit key of its own.)
- A key that already has a dedicated variable is **rejected**, so every limit
  has exactly one source. The nine rejected keys, and the variable to use
  instead:

  | Rejected key | Configure with |
  | --- | --- |
  | `variables` | `ANNEALBRIDGE_EXACT_MAX_VARIABLES` |
  | `reads` | `ANNEALBRIDGE_MAX_QPU_READS` |
  | `annealing_time_us` | `ANNEALBRIDGE_MAX_QPU_ANNEALING_TIME_US` |
  | `time_seconds` | `ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS` |
  | `local_reads` | `ANNEALBRIDGE_MAX_LOCAL_READS` |
  | `sweeps` | `ANNEALBRIDGE_MAX_SWEEPS` |
  | `local_retries` | `ANNEALBRIDGE_MAX_LOCAL_RETRIES` |
  | `remote_retries` | `ANNEALBRIDGE_MAX_REMOTE_RETRIES` |
  | `top_k` | `ANNEALBRIDGE_MAX_TOP_K` |

If a registered backend declares a limit key the policy has no value for, the
service refuses to start and the failure is reported as a settings error — the
same way a bad value is. See [Adding a backend](backends.md#adding-a-backend).

### Unknown and invalid variables

The two failure modes are deliberately different:

- **An unknown `ANNEALBRIDGE_*` variable is not fatal.** A typo such as
  `ANNEALBRIDGE_MAX_QPU_READ`, or a leftover from an older release, is
  ignored: the setting keeps its default and the process logs one `WARNING`
  naming the variable(s) at startup. So a misspelt ceiling cannot silently
  look like it took effect, and a harmless leftover cannot stop the server
  from starting. The prefix and the suffix are compared case-insensitively,
  exactly as the settings lookup itself is.
- **An invalid *value* is fatal.** The process prints an operator-readable
  message and exits with code `2` (both `annealbridge` and
  `annealbridge-mcp`). The message names the variable and the reason —
  **never the value**, in case one holds a credential mis-set into an
  `ANNEALBRIDGE_*` variable.

## Vendor credentials

Vendor credentials are deliberately **not** `ANNEALBRIDGE_*` settings. They
never enter `ServerSettings`, the execution policy, or any model.

| Variable | Default | Meaning |
| --- | --- | --- |
| `FUJITSU_DA_API_KEY` | unset | Fujitsu Digital Annealer API key, required by `fujitsu_da`; sent only as the `X-Api-Key` request header |
| `FUJITSU_DA_URL` | `https://api.aispf.global.fujitsu.com/da` | Fujitsu DA base URL; must start with `https://`, or the backend is unavailable with `configuration_error` / `BACKEND_CONFIG_INVALID` |
| `DWAVE_API_TOKEN` | unset | Read by **Ocean**, not by AnnealBridge; one of the ways to give Leap credentials |

The two Fujitsu variables are read by the `fujitsu_da` backend alone, live on
every call, and are never cached.

D-Wave credentials belong to Ocean's own configuration — `dwave config create`
writes a config file, or set `DWAVE_API_TOKEN` — so AnnealBridge never has to
read, store, or pass a token itself. Full steps for both vendors are in
[Backends](backends.md#d-wave-setup).

Whatever the vendor, credential values are masked wherever they might be
echoed: results, error messages, logs and metadata all pass through redaction,
and a backend opts in purely by declaring its credential environment variables
and header names. See [Security](security.md).

## Configuring an MCP host

An MCP host launches the server as a subprocess, so the settings go in the
host's own `env` block rather than your shell. For Claude Desktop
(`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "annealbridge": {
      "command": "/path/to/venv/bin/annealbridge-mcp",
      "env": {
        "ANNEALBRIDGE_ALLOW_REMOTE": "false",
        "ANNEALBRIDGE_ENABLED_BACKENDS": "exact,simulated_annealing",
        "ANNEALBRIDGE_MAX_CONCURRENT_SOLVES": "2"
      }
    }
  }
}
```

Every value in an `env` block is a string, including booleans and numbers.
Restart the host after editing the file. See [MCP server](mcp.md) for the rest
of the host configuration.

## Recommended production settings

- **Keep `ANNEALBRIDGE_ALLOW_REMOTE=false` unless you need a remote
  backend.** It is the default, and it is the single switch that keeps the
  process off the network and off any vendor quota. Turn it on deliberately,
  for a deployment that has credentials and a budget.
- **Leave `ANNEALBRIDGE_ALLOW_REMOTE_RETRIES=false`** unless infeasible
  results on a remote backend are genuinely costing you more than the retries
  would. With it off, a `max_retries: 3` problem cannot turn into multiple
  billed submissions.
- **Set `ANNEALBRIDGE_ENABLED_BACKENDS` to an explicit allow-list.** Naming
  the backends a deployment is meant to use makes the intent legible, keeps an
  agent from being offered anything else, and survives a future backend being
  added to the registry.
- **Bind the HTTP transport to `127.0.0.1`.** That is the default for
  `ANNEALBRIDGE_HTTP_HOST` — an empty or blank value is refused rather than
  falling back to every interface — and the server has no authentication of
  any kind: do not bind it to `0.0.0.0` or expose it to a public network. If
  remote access is genuinely needed, put it behind an authenticating reverse
  proxy or on a private network. See [Security](security.md).
- **Tune the ceilings to the host.** `ANNEALBRIDGE_MAX_CONCURRENT_SOLVES`,
  `ANNEALBRIDGE_MAX_LOCAL_READS`, `ANNEALBRIDGE_MAX_SWEEPS` and
  `ANNEALBRIDGE_MAX_TOP_K` together bound how long one request can hold a
  concurrency slot; `ANNEALBRIDGE_EXACT_MAX_VARIABLES` bounds how much memory
  an exhaustive solve may ask for. `ANNEALBRIDGE_SA_WORKERS` is the CPU
  budget of one local annealing solve: up to
  `SA_WORKERS × MAX_CONCURRENT_SOLVES` threads can be sampling at once, so on
  a shared host keep that product near the core count (it only affects
  speed, never results).
- **Watch the startup log for the unknown-variable `WARNING`.** It is the only
  signal that a setting you thought you configured is still at its default.

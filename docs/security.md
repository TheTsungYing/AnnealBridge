[← Back to README](../README.md) · [Documentation index](README.md)

# Security model

AnnealBridge is middleware an AI agent calls, so its threat model is shaped by
two questions: what can a request make the server spend, and what can leave the
server. The answers below are properties of the code, not recommendations.

## Remote execution is off by default

`ANNEALBRIDGE_ALLOW_REMOTE` defaults to `false`. Until you opt in, a request
for `dwave_qpu`, `leap_hybrid_bqm`, `leap_hybrid_cqm` or `fujitsu_da` returns
`backend_unavailable` — it does not silently fall back to a local solver.
Nothing touches the network and no vendor quota is consumed.

The opt-in is separate from installing the extras and separate from having
credentials: all three have to be true before a remote backend runs. See
[Configuration](configuration.md) for the variables and
[Backends](backends.md) for what each remote backend needs.

## Retries and resource limits

- **Remote retries are off by default.** `ANNEALBRIDGE_ALLOW_REMOTE_RETRIES`
  defaults to `false`, so a problem carrying `max_retries: 3` cannot turn into
  several billed QPU submissions.
- **Opting in does not lift the ceiling.** `max_retries` remains bounded by
  `ANNEALBRIDGE_MAX_REMOTE_RETRIES` (default `3`), and a request above it is
  rejected with `resource_limit_exceeded` *before* any submission — zero
  vendor calls, zero quota.
- **The CQM path never retries.** It always runs exactly one attempt, because
  a native-constraint model has no hard penalty to turn up.
- **Over-limit values are errors, never clamps.** `MAX_QPU_READS`,
  `MAX_QPU_ANNEALING_TIME_US` and `MAX_REMOTE_TIME_SECONDS` are enforced by
  refusing the request. A value is never quietly reduced to the ceiling, so an
  agent can never mistake a downgraded run for the one it asked for.
- **Local parameters are bounded too.** `ANNEALBRIDGE_MAX_LOCAL_READS`,
  `ANNEALBRIDGE_MAX_SWEEPS`, `ANNEALBRIDGE_MAX_LOCAL_RETRIES` and
  `ANNEALBRIDGE_MAX_TOP_K` keep a single request from holding one of the
  concurrency slots (four by default) for a very long time. These are rejected
  rather than clamped as well. `ANNEALBRIDGE_SA_WORKERS` and
  `ANNEALBRIDGE_TABU_WORKERS` are the CPU budget of one local solve on their
  respective backend (threads per solve); they never change a result.
  `ANNEALBRIDGE_SB_MAX_VARIABLES` is the matching bound on *memory* for
  `simulated_bifurcation`, which holds the couplings as a dense `N × N`
  matrix: an over-limit problem is refused with `SB_VARIABLE_LIMIT` before
  anything is allocated, never clamped. The opt-in post-processing, which runs
  on this machine whatever the backend, is bounded by
  `ANNEALBRIDGE_MAX_POSTPROCESS_CANDIDATES` and by a per-attempt count of move
  evaluations, `ANNEALBRIDGE_MAX_POSTPROCESS_EVALUATIONS`: an over-limit
  request is refused with `POSTPROCESS_LIMIT` before solving, and a budget
  that runs out part-way stops the search with a `POSTPROCESS_LIMIT_REACHED`
  warning rather than running on.
- **The penalty ladder cannot run away.** A hard penalty that would have to
  double past the floating-point range stops with a structured
  `PENALTY_OVERFLOW` error instead of a solver error, and no backend is ever
  called with a non-finite model.

## Streamable HTTP has no authentication

In HTTP mode the server binds `127.0.0.1` by default and has **no
authentication or authorization of any kind**.

Do not expose it directly to a public network, and do not bind it to
`0.0.0.0`. If remote access is genuinely needed, put it behind an
authenticating reverse proxy or keep it on a private network. See
[MCP server](mcp.md) for the transport options.

## Credentials never appear in output

Solver metadata is filtered through a timing whitelist rather than returned
as-is, and results, error messages, logs and metadata all pass through
redaction, so an API token cannot leak into a tool response or a stack trace.
A dedicated credential-leak test suite covers this for every backend.

A backend opts into the protection purely by declaring its credential
environment variables and header names in `SolverCapabilities.credentials`.
The shared redaction module knows no vendor, so a new backend is protected the
moment it is registered — with no edit to the solver layer.

**How redaction works.** Each live credential value is replaced literally,
after stripping surrounding whitespace and also in its JSON-escaped and
URL-encoded forms, plus case-insensitive header and `token=` /
`Authorization:` patterns.

**Where its edges are**, stated plainly so you can reason about them:

- A value shorter than 8 characters is masked only by the patterns —
  replacing `1` everywhere would shred every number in a message.
- Any other transformation of a key cannot be matched literally: base64, a key
  the sender split across lines, or a key a vendor truncates inside its own
  error body.

Those cases are closed upstream instead:

- keys travel only in request headers, never in a URL or a body;
- response bodies are redacted *before* they are summarised or cut to length;
- the original exception is dropped, so no unredacted text survives in a
  traceback.

The D-Wave config-file token is re-read only when the config file or its
selector variables change, not on every log line.

## Fujitsu key handling

`FUJITSU_DA_API_KEY` follows the same rules, tightened by the fact that the
backend speaks the vendor's HTTPS API directly:

- It is read from the environment **inside the backend only**, on every call.
  It never enters `ServerSettings`, the execution policy, a model or metadata.
- It is sent solely as the `X-Api-Key` request header — never in a URL query
  string, a log line, an error message or an exception chain.
- Redaction masks the key's value wherever it might be echoed (a vendor error
  body, for instance) and masks `X-Api-Key` / `X-Access-Token` header lines, so
  even a dumped request cannot reveal it.
- The credential-leak suite runs every case against `fujitsu_da` as well as the
  D-Wave backends.

`FUJITSU_DA_URL` must start with `https://`; any other scheme makes the backend
unavailable with `configuration_error` / `BACKEND_CONFIG_INVALID` rather than
sending the key over an unencrypted connection.

D-Wave credentials are deliberately *not* AnnealBridge settings: they belong to
Ocean's own configuration (`dwave config create` or `DWAVE_API_TOKEN`), so
AnnealBridge never has to read, store or pass a Leap token itself.

## Fujitsu polling has a hard budget

A submitted job is polled for at most `time_limit_sec + 60` seconds. Past that
the backend stops waiting, sends a best-effort cancel and then a best-effort
delete — so the job does not keep occupying one of the account's slots — and
returns `REMOTE_TIMEOUT`.

Each individual HTTP request carries its own `request_timeout_seconds` (30 s by
default), so the real worst case is the polling budget plus up to 30 s each for
the submit, the last status request, the cancel and the delete.

One job cannot be released: a submit that itself times out. The vendor may have
created it, but no `job_id` ever reached us. The error says so and asks the
operator to check the account's job list.

## What is sent to a vendor

The Fujitsu request body carries only variable **indices** and coefficients —
the compiled QUBO as one `binary_polynomial`, plus the four solver options in
the `fujitsuDA3` block. The problem's `name`, any description and every other
business string stay on your machine.

For the D-Wave backends the submission is the compiled model itself, handed to
Ocean; the problem document is not sent, and the submission label is a fixed
string, not your problem's name.

In both cases the vendor sees a mathematical model, and the mapping from that
model back to your business meaning never leaves the process.

## Reporting a vulnerability

Please follow the process in [SECURITY.md](../SECURITY.md).

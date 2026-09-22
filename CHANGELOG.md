# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The MCP server now exposes three prompts and five resources beside its four
  tools. The prompts — `pick_subset`, `assign` and `schedule_shifts` — each
  take the user's request in their own words as one optional `request`
  argument and return a guide for turning that shape of request into a problem
  document (which decisions become variables, what the objective and the
  constraints are, which schema version to declare, which tools to call),
  ending in the smallest complete document of that shape; a host lists them in
  its input menu, which makes them the one place a user sees what the server
  is for without reading a tool description. They repeat what the server
  instructions already say and add no rule of their own, and a test validates
  and solves each embedded example. The resources are the four example
  documents (`annealbridge://examples/knapsack`, `integer_knapsack`,
  `assignment`, `tsp`) and the full problem schema
  (`annealbridge://schema`, the same text `annealbridge export-schema`
  prints), all `application/json`. The examples ship inside the package as
  data files, since the repository's `examples/` directory never reaches an
  installed wheel, and a test holds the packaged copies byte-for-byte equal to
  it. The server instructions now name both surfaces, and
  `scripts/check_install.py` checks an installed wheel lists them and serves
  the packaged example unchanged.
- `solve_optimization` reports progress. A host that sends a progress token
  with the call receives one MCP progress notification as each stage of each
  attempt starts — compile, solve, validate — carrying a message such as
  `attempt 2 of 3: solving on simulated_annealing` and the stage count as
  progress out of the total the attempt budget allows; without a token
  nothing is sent, and a notification the host can no longer receive is
  logged once and the rest skipped while the solve finishes regardless.
  Underneath, `OptimizationService.solve` grew a keyword-only `on_progress`
  callback, called with a new `orchestration.SolveProgress` (attempt number,
  attempt budget, stage, backend name, and the message above) on the solving
  thread and outside every timing window; an exception it raises is logged
  and dropped, never turned into `solver_error`. Without the argument the
  service behaves exactly as before, and the CLI passes none. `validate` and
  `recommend` report no progress.
- A successful `SolveResult` now carries a `message`: one deterministic
  English sentence an agent can relay as is, where the field used to be null
  unless something had gone wrong. It names the backend that produced the
  answer, whether optimality is proven, the rank-1 objective with the
  optimization direction, and how many distinct candidates the attempt saw,
  how many of those were feasible and how many are returned — every one a
  fact the result already carries in a field of its own, computed from the
  same values, so the sentence cannot disagree with what sits beside it.
  There are two shapes, one per kind of backend: `exact proved optimality:
  rank 1 has objective 17 (maximize); 10 of 16 distinct candidates were
  feasible, 5 returned.` when an exhaustive backend enumerated everything,
  and `simulated_annealing found 11 distinct candidates (10 feasible), 5
  returned: rank 1 has objective 17 (maximize); optimality is not proven.`
  for a heuristic. `with soft violation X` follows the objective when rank 1
  carries one, and the heuristic shape adds `on attempt N` when the answer
  came from a retry (an exhaustive backend is given a single attempt).
  It deliberately holds no timing and no configuration value or limit, so the
  same request produces the same sentence — timings stay in `elapsed_ms` and
  the per-attempt fields. The infeasible and failure paths keep the messages
  they had, and the `message` field description now documents all three.
- `OptimizationCapabilities` gained `annealbridge_version`, the installed
  package version, read from the same `package_version()` source as
  `SolveResult.annealbridge_version`, so a capabilities response names the
  build that produced it. Both interfaces build the same view: the MCP tool
  returns the field, and the CLI `capabilities` command carries it in the
  object it renders from, whose table still prints backends only.
- `recommend` orders the local heuristics by problem shape instead of by
  registry position, from two new `SolverCapabilities` declarations that are
  `False` unless a backend opts in: `strong_on_large_dense` (declared by
  `tabu` and `simulated_bifurcation`) and `weak_on_penalty_dominated`
  (declared by `simulated_bifurcation` alone). The first says the backend
  reaches the same energy as its peers in a fraction of the time on large
  dense unconstrained models; the second says it has a measured lower hit rate
  once hard constraints compile to penalties. A new `structure_fit` sort key,
  applied after the existing tiers and before registry order, scores a backend
  `0` when it declares the strength and the problem is large, dense and free
  of effective hard constraints, with the reason code `R_DENSE_STRENGTH`; `2`
  when it declares the weakness and the problem has at least one effective
  hard constraint, each of which becomes a penalty far above the objective,
  with `R_PENALTY_WEAKNESS`; and `1` otherwise. Both shapes are matched **only
  on the bqm path**, so a backend that compiles to cqm is never matched
  whatever it declares. Large means at least 500 estimated compiled variables,
  slack and integer encoding bits included; dense means the distinct variable
  pairs the model will couple — from the objective's quadratic terms and from
  the variable clique of every effective constraint, hard or soft, each pair
  counted once — reach at least half of `m(m−1)/2` for the `m` declared
  variables; and a hard constraint is effective when it has a non-zero
  coefficient and is not redundant over the declared bounds, since a redundant
  inequality such as `x <= 1` on a binary variable compiles to no penalty.
  Within the local heuristic tier a large dense problem with no effective hard
  constraint therefore ranks `tabu`, `simulated_bifurcation`,
  `simulated_annealing`; one with an effective hard constraint, such as the
  shipped knapsack, ranks `simulated_annealing`, `tabu`,
  `simulated_bifurcation`; a small unconstrained one matches neither shape and
  keeps registry order — and an `exact` the problem fits still ranks first
  overall in every case. Like every other dispatch in the service this reads a
  declaration, never a name, so `exact` and the remote backends are untouched,
  nothing becomes usable or unusable, and there is still no cost estimation,
  no benchmark and no history. The thresholds are calibrated on the
  measurements already recorded here and repeated in `docs/backends.md`, which
  gains a `Best for` column, a **When to choose it** paragraph per local
  heuristic, a section on this ordering — including what the conservative rule
  gives up on a large dense problem that does carry a hard constraint — and
  the two new fields in its "Adding a backend" checklist; `docs/mcp.md` and
  both READMEs describe the same thing from the agent's side.
- An eighth backend, `simulated_bifurcation`, implemented here on `numpy`:
  the classical-mechanics heuristic of Goto et al., "High-performance
  combinatorial optimization based on classical mechanics", *Science Advances*
  **7**, eabe7953 (2021). One integration step updates every variable of every
  read at once with a single dense matrix product, which is what makes it a
  matrix routine rather than a loop over reads. It is part of the core install
  — `numpy` is already a dependency — takes the BQM path, and honours
  `num_reads` (parallel trajectories), `num_sweeps` (integration steps) and
  `seed` (`0`–`4294967295`, the same range `tabu` declares). Both variants of
  the paper are implemented and selected by a `solver.simulated_bifurcation`
  option block: `mode: "discrete"` (dSB, the default) and `mode: "ballistic"`
  (bSB). Which one wins is problem-dependent, which is why the choice is
  exposed — on dense ±1 SK instances at 1000 variables dSB matched the best
  energy simulated annealing and tabu search found (−9112) in 3.9 s against
  their 33 s and 11.6 s, while bSB stopped three units short; on the shipped
  integer knapsack example it is the other way round, with dSB stalling at a
  local minimum three units above the optimum bSB reaches. Its weak spot is
  the small penalty-dominated QUBO that most of the shipped examples are: it
  hits their optimum in 1–5 % of its reads against about 7 % for the annealer,
  so the knapsack example needs roughly 2000 reads. Neither more steps nor a
  steepest-descent polish improved that when measured, so no polish is
  applied. It is registered fourth, after `tabu`, ranks with the other local
  heuristics in `recommend`, and is bounded by the existing
  `ANNEALBRIDGE_MAX_LOCAL_READS`, `ANNEALBRIDGE_MAX_SWEEPS` and
  `ANNEALBRIDGE_MAX_LOCAL_RETRIES` — it declares no time limit, because there
  is no wall clock anywhere in it. The result is a function of `(problem,
  num_reads, num_sweeps, seed, mode, device)`; the BLAS thread count is not
  part of that (measured bit-identical at 1 and 6 threads), but a different
  CPU instruction set or BLAS build may round the matrix products differently,
  which is a weaker promise than `tabu` makes and is documented as such.
- `ANNEALBRIDGE_SB_DEVICE` (`cpu`, the default, or `cuda`) chooses where the
  `simulated_bifurcation` dynamics run, with a new `gpu` extra
  (`pip install "annealbridge[gpu]"`) supplying PyTorch, imported lazily and
  only on the `cuda` setting. The extra is deliberately **not** part of `all`:
  torch is a large download, and on Windows the wheel PyPI serves is the
  CPU-only build, so a CUDA run needs `torch` reinstalled from PyTorch's own
  index — a second step a pip extra cannot express, and one the backends page
  spells out. **The backend never falls back to the CPU**: with `cuda` set, no
  torch reports `not_installed` and torch without a visible CUDA device
  reports `unavailable`, both through the ordinary capabilities view, because
  a silent fallback would hide a misconfigured server behind an answer that
  merely arrived more slowly. Measured on a GeForce GTX 1660 SUPER against the
  OpenBLAS CPU path on dense ±1 SK instances (1000 steps, 100 reads): 0.8 s
  against 2.3 s at 1000 variables, 6.4 s against 25.5 s at 5000, and 23 s for
  10 000 variables on the GPU at 740 MB of VRAM; the same request twice was
  bit-identical on the GPU, while the GPU's samples differ from the CPU's for
  the same request, as the device is part of the reproducibility contract.
- `ANNEALBRIDGE_SB_MAX_VARIABLES` (int ≥ 1, default `10000`) caps the compiled
  problem the `simulated_bifurcation` backend accepts, internal slack and
  integer-encoding variables included. It holds the couplings as a dense
  single-precision `N × N` matrix — `4 N²` bytes, about 400 MB at the default
  — so unlike the read and sweep ceilings it bounds memory rather than time,
  and like them it is a refusal, not a clamp: a larger problem is refused with
  the new `SB_VARIABLE_LIMIT` error code under `resource_limit_exceeded`,
  before compiling. The backend declares the cap in its capabilities (a new
  `compiled_variable_limit` declaration any backend may make), so the service
  checks it from the pre-compile estimate exactly as it checks the exhaustive
  ceiling, `recommend` lists the same refusal in `blocking`, and
  `capabilities` reports it as `max_variables`. It is a settings-only value,
  not a policy limit key, for the same reason the worker counts are: it sizes
  one backend's machine.
- A seventh backend, `tabu`, on `dwave.samplers.TabuSampler` — a multistart
  tabu search and a strong heuristic on dense QUBOs, where simulated annealing
  spends sweeps on moves tabu search rules out. It is part of the core install:
  no extra, no credential, no network. It takes the BQM path, honours
  `num_reads` and `seed`, and, having no notion of sweeps, declares
  `supports_num_sweeps=False` so a non-default `num_sweeps` raises
  `PARAMETER_IGNORED` rather than being swallowed by the sampler's `**kwargs`.
  Its seed range is `0`–`4294967295`, deliberately *not* the simulated
  annealer's `0`–`2147483647`: the two samplers in `dwave-samplers` disagree, so each
  backend declares its own rule instead of sharing a constant. It is registered
  third, after `simulated_annealing`, and bounded by the existing
  `ANNEALBRIDGE_MAX_LOCAL_READS` and `ANNEALBRIDGE_MAX_LOCAL_RETRIES` — no new
  limit key.
- The `tabu` backend fixes `timeout=None` and `num_restarts=0` rather than
  accepting the vendor defaults, and neither is reachable from a solver
  preference. The vendor's `timeout=20` is a 20 ms wall clock *per read*, which
  would make every answer a function of the machine's speed and its current
  load — at 2000 variables the first search is still being cut off mid-way and
  returns a worse energy than the untimed one. With the wall clock gone,
  `num_restarts=0` is what bounds the work instead: one plain search per read,
  costing a predictable *count* of variable updates (about 4 ms per read at 30
  variables, 12 ms at 200, 88 ms at 800, 330 ms at 2000, single-threaded).
  Restarts were not traded away for quality — on dense ±1 SK instances at 300
  and 600 variables, a fixed time budget spent on more restarts scored no
  better than the same budget spent on more reads — so diversification is left
  to `num_reads`, which the caller controls and policy caps. The result is
  therefore a function of `(problem, num_reads, seed)` alone.
- `ANNEALBRIDGE_TABU_WORKERS` (int ≥ 1, unset auto-detects the CPUs available
  to the process) sets how many read shards the `tabu` backend samples at once.
  It is a separate variable from `ANNEALBRIDGE_SA_WORKERS` because the two
  samplers are tuned independently, and like it, it changes wall time only —
  never a result.
- `tests/unit/test_registry_entry.py` pins what the MCP Registry entry depends
  on but nothing else would notice going missing: the
  `mcp-name: io.github.TheTsungYing/annealbridge` comment in `README.md` —
  present *and* followed by a boundary, since a name glued to a trailing
  character fails to match just as silently as a deleted line — the same name
  in `server.json`, its PyPI identifier being this distribution, and both of
  its version fields equalling `pyproject.toml`. The release workflow checks
  the versions too, but only once a release is under way; these run on every
  `pytest`, and the README marker had no check at all — dropping it let PyPI
  publish and failed only the registry job, after the version number was spent.

### Changed

- `get_optimization_capabilities` takes an `include_schema` boolean that
  defaults to `false`, and the default response no longer carries
  `problem_json_schema`: the field's type went from `object` to
  `object | null`, so the output schema a host sees changed with it. The
  schema was roughly three quarters of every response, so the plain call is
  now about a quarter of its former size, and an agent that only
  needs the backend list with its limits now pays for none of it, while
  `include_schema: true` returns exactly the schema it always did, identical
  to what `annealbridge export-schema` prints. `build_capabilities` grew a
  matching `include_schema` keyword argument defaulting to `true`, so the CLI
  `capabilities` command and every other direct caller are unchanged; with it
  false the cached schema is not even serialised, rather than serialised and
  dropped. The server instructions and the `validate_optimization_problem`
  description now say to pass `include_schema: true` when the schema itself is
  what is needed, and `scripts/check_install.py` asks for it the same way —
  which is also where the stale `Call order` assertion, left behind when
  those instructions' heading became `When to call each tool`, is fixed.
- The `structure_fit` key described under *Added* changes what `recommend`
  prints for problems that already existed: the shipped
  `examples/knapsack.json` now lists `simulated_bifurcation` last among the
  local heuristics and carries one extra reason code,
  `R_LOCAL_HEURISTIC, R_PENALTY_WEAKNESS`, where it previously carried
  `R_LOCAL_HEURISTIC` alone. No backend became usable or unusable, and
  `solve` is unaffected — `solver.backend` is still used exactly as given.
- The MCP server instructions now tell an agent how to turn a recommendation
  into the one backend a request carries, which was the one step the four tool
  descriptions left to guesswork. A backend the user named is used as given
  and never substituted; otherwise `recommend_backend` is called — the step is
  no longer described as being for when "the choice is not obvious" — and its
  reason codes are read, `R_DENSE_STRENGTH` and `R_PENALTY_WEAKNESS`
  included. When more than one local backend is usable and the user did not
  ask for an answer without being consulted, the agent presents the top
  entries with one-line reasons and **asks** which to run instead of deciding
  silently; either way the answer names the backend that ran and why. The
  `recommend_backend` and `solve_optimization` descriptions carry the matching
  sentences, and the instructions still name no configuration value and no
  limit — the reason codes they point at are categorical.
- The read sharding the simulated annealer grew — the fixed 25-read shard
  layout, the `SeedSequence`-derived per-shard seeds, the bounded fan-out and
  the in-order merge — moved out of `solvers/simulated_annealing.py` into a
  shared `solvers/sharding.py`, because `tabu` needs exactly the same rules and
  the reproducibility contract they carry (layout and seeds depend on
  `(num_reads, seed)` alone, so the answer is the same at any worker count) is
  one contract, not two that happen to agree. `simulated_annealing` behaves
  identically: same shard size, same derived seeds, same results for the same
  seed.
- The release workflow publishes to the MCP Registry as well as to PyPI. A new
  `publish-registry` job runs after `publish-pypi`, authenticates with GitHub
  OIDC (no stored credential) and publishes `server.json`. It first polls
  PyPI for the exact version being released and the
  `mcp-name: io.github.TheTsungYing/annealbridge` marker, because the registry
  proves ownership by reading that line out of the description PyPI renders
  from `README.md` and the JSON API lags an upload by a few seconds. The
  existing tag check now also refuses a `server.json` whose two version fields
  disagree with the tag, before anything irreversible happens. Recorded in the
  *Releasing* section of `CONTRIBUTING.md` and the *MCP Registry* section of
  `docs/mcp.md`.
- The MCP server instructions and tool descriptions now say which tool is worth
  a call in a given situation instead of prescribing the same four steps every
  time. On a local backend an agent may call `solve_optimization` directly —
  an invalid document comes back as `invalid_problem` carrying the same errors
  and `recommended_action` `validate_optimization_problem` would have given, so
  nothing is lost by finding out that way — while a remote backend or a large
  problem is still validated first, before quota is spent, and
  `get_optimization_capabilities` is for when the backend list with its limits
  or the full schema is actually needed rather than for every problem. On a
  host that asks the user to approve every tool call, a simple problem now
  costs one or two approvals instead of four. The instructions and the
  `solve_optimization` description also open with the everyday requests this
  server answers — which items to take within a budget or capacity, how to
  assign people or jobs to seats, shifts or machines, in which order to visit a
  handful of places, how to split things into groups or pick a subset meeting
  several requirements at once — so an agent recognizes them when the user
  never says "optimization". The **choosing a backend** rules (a named backend
  is used as given; otherwise `recommend_backend`, and more than one usable
  local backend means asking the user) and the four rules a first document
  breaks are unchanged, and the instructions still name no configuration value
  and no limit. Both READMEs gained a *what this is not for* section and the
  `--version` warm-up command in their MCP sections; `docs/mcp.md` matches, in
  its *Server instructions* and *Typical agent workflow* sections.

## [0.2.1] - 2026-09-16

Published to the [MCP Registry](https://registry.modelcontextprotocol.io) as
`io.github.TheTsungYing/annealbridge`. No optimization behaviour changed.

### Added

- `annealbridge mcp` runs the MCP server, exactly as the `annealbridge-mcp`
  console script does. Every argument is handed to the server's own parser,
  `--help` and `--version` included, so the two entry points answer
  identically, and a missing `[mcp]` extra prints the same one-line install
  hint on stderr and exits `2` through the same shim. It exists for the MCP
  Registry, which composes a package command as `<runtimeHint>
  <runtimeArguments> <identifier> <packageArguments>` and requires
  `identifier` to be the PyPI project name, so no console script whose name
  differs from the project is reachable. `server.main()` gained optional
  `argv` and `prog` parameters to support it; their defaults are the console
  script's previous behaviour. Documented in the new `mcp` section of
  `docs/cli.md` and the *Installation* section of `docs/mcp.md`.
- `server.json` in the repository root describes the server for the MCP
  Registry. A registry client resolves it to
  `uvx --from=annealbridge[mcp] annealbridge mcp`, covered by a real
  subprocess test. Its two version fields must equal the `pyproject.toml`
  version of the release being published.

### Changed

- `README.md` carries an `mcp-name: io.github.TheTsungYing/annealbridge` HTML
  comment. The MCP Registry proves ownership of a PyPI package by looking for
  that line in the description PyPI renders from the README, so it must
  survive every release; dropping it breaks the next registry publish.
  Documented in the new *MCP Registry* section of `docs/mcp.md`.

## [0.2.0] - 2026-09-16

### Added

- `annealbridge --version` and `annealbridge-mcp --version` print
  `annealbridge <version>` and exit `0`. Both are answered before any
  `ANNEALBRIDGE_*` setting is read, so they work while a setting holds an
  invalid value, and `annealbridge-mcp` logs no unknown-variable warning for
  them; `annealbridge --help` and `annealbridge-mcp --help` now list
  `--version` (subcommand help does not). Without the `mcp` extra,
  `annealbridge-mcp` still reports the missing extra and exits `2`.
  Documented in the new *Global options* section of `docs/cli.md` and the
  option table of `docs/mcp.md`.
- Each backend entry of `get_optimization_capabilities` gains `seed_min` and
  `seed_max`: the inclusive range of `solver.seed` the backend accepts, or
  `null` when it declares none. A backend declares them on
  `SolverCapabilities` — both together, only with `supports_seed=True` and
  with `seed_min <= seed_max`, or the declaration fails to construct. They
  are the backend's own rule, not a policy limit. `simulated_annealing`
  declares `0`–`2147483647`. Documented in the *BackendCapability* table of
  `docs/output-format.md`, the `get_optimization_capabilities` section of
  `docs/mcp.md` and *Declare `SolverCapabilities`* in `docs/backends.md`.

### Changed

- Both READMEs were restructured for a first-time reader: a PyPI badge, a
  Mermaid flowchart of the pipeline under the tagline (replacing the ASCII
  diagram), then *Quick start* (`pip install` and a self-contained Python
  knapsack that needs no file), *Use it from an AI agent (MCP)* (the `uvx`
  host entry, `claude mcp add` and the three tool calls behind one chat
  turn), *Use it from the command line*, *The problem JSON*, *How it works*,
  *Backends*, *Design guarantees* (the former *Features*), *Documentation*
  and *Development*. *Installation in depth*, *Security in one paragraph*
  and *Status* were folded into one-line pointers to `docs/mcp.md`,
  `docs/security.md` and `docs/limitations.md`; no technical statement
  changed.
- Dependency floors now match what the code actually calls:
  `pydantic-settings>=2.7` (`NoDecode`) and `dwave-system>=1.10`
  (`LeapHybridCQMSampler`). Older versions installed but failed on import or
  on the first hybrid CQM solve.
- The default test suite is strict: `xfail_strict` makes an xfail that starts
  passing a failure, and `filterwarnings = error` turns every warning into
  one. CI now runs on pushes to `main` and on pull requests only, cancels a
  superseded run on the same ref, and caches pip downloads between runs.
- A new *Upgrading* section in `docs/mcp.md` records that `uvx` reuses the
  environment it resolved on its first run, so a later release does not reach
  an existing host configuration by itself: `uv cache clean annealbridge` and
  a host restart do, `--refresh` re-resolves a single run at the cost of a
  network round trip, and pip or pipx installs upgrade the usual way.
  `annealbridge-mcp --version` and the `annealbridge_version` field of a
  result identify the version actually running. Both READMEs point at it from
  the MCP section.
- Documentation corrections: the `mcp dev` command in `docs/mcp.md` now points
  at `src/annealbridge/interfaces/mcp/__init__.py` (the previous target,
  `server.py`, loaded a second server instance that exposes no tool); the CLI
  examples in both READMEs show the `Elapsed` and `Optimality proven` lines;
  the capabilities docstring, `docs/mcp.md` and `docs/configuration.md` state
  that `solver.backend` accepts only the six built-in names and that a
  registry key must equal its `capabilities.name`; the `capabilities` example
  in `docs/cli.md` notes that it reflects an install without the `dwave`
  extra; and `docs/configuration.md` records that `ANNEALBRIDGE_SA_WORKERS ×
  ANNEALBRIDGE_MAX_CONCURRENT_SOLVES` bounds the concurrent sampling threads.
- Fujitsu DA result decoding drops an unreachable branch and gains tests for
  malformed vendor responses.
- `recommend()` now reports an exhaustive backend's `EXACT_VARIABLE_LIMIT`
  blocking entry in the same wording `solve()` uses — "Compiled problem has N
  variables (including internal), exceeding the exhaustive backend limit of
  L", where it previously said "Estimated compiled variables (N) exceed the
  exhaustive backend limit of L". The code, the `solver.backend` path and the
  condition that triggers it are unchanged.
- Internal tidy-up of `orchestration/` (no behaviour change): `ExecutionPolicy`
  gains `required_limit()`, which every solve-time check now reads its ceiling
  through instead of comparing against a possibly-`None` `limit()`; the
  `NO_COMPILER_FOR_MODEL_TYPE` and `EXACT_VARIABLE_LIMIT` messages are built
  by one function each in `orchestration/limits.py`; and the service's
  compile/solve/validate loop is split into `_prepare_attempts`,
  `_solve_attempt` and the pure `_infeasible_message`, with the last
  attempt's metadata assigned in exactly one place.
- Internal tidy-up of `validation/` and `orchestration/limits.py`, with a new
  `models/reflection.py` (no behaviour change): the backend-fit warnings are
  split into `_warn_exact_limits` and `_warn_embedding_density`, the decision
  behind `PARAMETER_IGNORED` is the pure `_ignored_parameters`, and the
  annotation reflection the validator and the limit checks each carried is
  now one set of primitives in `models/reflection.py`. Every warning code,
  message, path and order is unchanged.
- The Fujitsu DA backend's INFO line after a solve uses the format every
  other backend logs: "Backend fujitsu_da solved problem <name>: <n>
  variables, <m> samples (job_id=…, solve_time_us=…,
  effective_time_limit_seconds=…)", where it previously said "Backend
  fujitsu_da finished job <id>: <n> variables, <m> solutions (…)".
- `docs/backends.md` records the fixed cost of simulated-annealing sharding:
  every shard re-runs the sampler's default beta-range estimate, which with
  `ANNEALBRIDGE_SA_WORKERS=1` and 400 reads (16 shards) adds about 9–18 %
  wall time at the default 1000 sweeps and 52–73 % at 200 sweeps; more
  workers offset it. It also states that the D-Wave backends declare their
  credentials through `solvers.ocean.ocean_sampler_holder`.
- Internal tidy-up of `config/`, `interfaces/` and `orchestration/limits.py`
  (no behaviour change): `ServerSettings.to_policy()` copies the fields
  listed in `ExecutionPolicy.model_fields`, with a test pinning the two
  models' field names, types, defaults and bounds to each other; the
  `enabled_backends` / `allow_remote` gates are defined once in
  `policy_gate_errors`, shared by `gate_errors` and the capabilities view's
  `enabled`; `interfaces.composition.exit_on_settings_error` gives the CLI and
  the MCP server one settings-error message and exit `2`; and the CLI's
  `solve`, `validate` and `recommend` share `_run` and `Annotated` option
  aliases, with help and output unchanged word for word.
- Internal tidy-up of `solvers/` (no behaviour change): backends share
  `base.result_from_sampleset`, `base.log_solved` and `base.record_column`;
  the D-Wave backends obtain their sampler holder from
  `ocean.ocean_sampler_holder`, which also declares their credentials and the
  Ocean config-file token source; remote metadata is built by one function,
  `metadata.remote_metadata`, which still filters timing through the
  whitelist and which `sanitize_sampleset_info` now uses; and a comment
  explains why the simulated-annealing seed range copies the
  `dwave-samplers` rule, with a test pinning that rule and the single- and
  multi-shard paths to it.
- Development tooling adopts `ruff` (no runtime behaviour change): it is
  installed with the `[dev]` extra and enables only the F (Pyflakes) and I
  (import sorting) rules, configured under `[tool.ruff]` in `pyproject.toml`.
  CI's Test job runs `ruff check .` before `pytest`, and `pytest` still runs
  when that step fails. Imports were re-sorted to match, except in
  `annealbridge.interfaces.mcp`, whose server-before-tools order is kept so a
  core-only `annealbridge-mcp` still exits 2 with the install hint; one unused
  import was removed from a test and a shared test fixture was renamed. There
  is no formatter.
- A `solver.seed` outside the range the selected backend declares is now
  refused during validation: `validate` and `solve` report
  `INVALID_SOLVER_PREFERENCE` at `solver.seed` ("solver.seed must be an
  integer between 0 and 2147483647 on backend simulated_annealing, got -1"),
  so `validate` returns `valid: false` and `solve` returns `invalid_problem`
  without calling the backend. Previously `validate` reported the problem
  valid and `solve` returned `solver_error` with a sampling error — on a
  request of at most 25 reads the sampler's own message, which states the
  range as `0` to `2^32 - 1`. Only `simulated_annealing`
  declares a range; a backend that does not support seeding still raises
  only `SEED_IGNORED` for any seed, the problem schema accepts the same
  values (only the `seed` field description now mentions the range), and
  `version: "1.0"` and `"1.1"` documents are checked by the same rule — a
  seed the sampler rejected before is now refused earlier, and no seed it
  accepted is refused. Documented in
  `docs/problem-format.md`, `docs/backends.md`, `docs/errors.md` and
  `docs/mcp.md`.
- `recommend()` lists a validation error that only one backend's declaration
  raises — today a seed outside its range — in that backend's `blocking`: the
  entry is `usable: false` with `R_UNUSABLE` and
  `estimated_compiled_variables: null`, while the problem stays `valid` and
  the other backends are unaffected. Previously that backend was reported
  usable although a solve on it failed. Documented in the
  *BackendRecommendation* table of `docs/output-format.md`.
- The `INVALID_SOLVER_PREFERENCE` recommended action adds that a seed must
  lie within the seed range the selected backend declares in its
  capabilities. Documented in `docs/errors.md`.

### Fixed

- `get_optimization_capabilities` and `annealbridge capabilities` no longer
  fail as a whole when one backend's availability check raises: that backend
  is listed with `available: false` and the redacted `unavailable_reason`
  "availability check failed: unexpected <ExceptionClass>: <message>", and
  the other backends are listed as usual. Previously the MCP tool returned a
  generic tool error and the CLI command exited `1`. A backend reporting an
  availability category outside the known set is listed the same way, its
  redacted detail ("no reason reported" when it gave none) followed by
  "(unknown availability category '<category>')"; `solve` and `recommend`
  now redact that detail in their message too. The view now asks through the same
  guard `solve` and `recommend` use, whose results and messages are
  unchanged. Documented in `docs/output-format.md`, `docs/mcp.md` and
  `docs/cli.md`.
- `annealbridge-mcp` now also exits `2` with one `Error: ...` line on stderr,
  instead of exit `1` with a traceback, when every setting is valid but the
  service refuses to be wired from them — for example a registered backend
  declaring a limit key the policy has no value for. When the server is used
  without its entry point (`mcp dev`, an embedding host), the service is
  built on the first tool call; a settings error there now reaches the client
  as `Error executing tool <name>: <settings message>`, naming the variable
  but never its value, and the server logs it at `INFO` without a traceback.
  Previously the
  client got only `Error executing tool <name>` and the server logged the
  error with a traceback. Documented in `docs/mcp.md`.
- The MCP server's own log format now takes effect. Started through
  `annealbridge-mcp` or `python -m annealbridge.interfaces.mcp.server`, every
  record is one stderr line, `<asctime> <LEVEL> <logger name>: <message>`,
  the unknown-`ANNEALBRIDGE_*`-variable `WARNING` included. Previously the
  rich handler the MCP SDK installs at import left the server's logging setup
  without effect, so records came out in rich format, wrapped at 80 columns.
  stdout still carries only the stdio protocol, and uses that bypass the
  entry point (`mcp dev`, an embedding host, an in-memory `Client(mcp)`) keep
  the SDK's handler. Documented in `docs/mcp.md`.
- `metadata.timing_us` drops a whitelisted key whose value is NaN or
  ±infinity, or an integer too large for a float — including a Fujitsu DA
  millisecond string `"NaN"` or `"inf"`, or one that overflows when converted
  to microseconds — and keeps the
  others; `effective_time_limit_seconds` and `average_chain_break_fraction`
  are `null` (not reported) when not finite. Previously such values were
  serialised as `null` inside `timing_us`, against its `number` type, so an
  MCP client validating the tool's output schema rejected the result.
  Documented in the *Timing whitelist* paragraph of `docs/output-format.md`
  and the Fujitsu DA section of `docs/backends.md`.

### Performance

- Five hot paths do less repeated work; every output — biases, violations,
  `satisfied` flags, ranking, metadata — is bit-for-bit what it was. The BQM
  compiler's non-finite-bias check reads dimod's numpy vectors instead of
  iterating every bias in Python (about 20 ms → 0.04 ms on a model with
  eighteen thousand interactions, paid again on every retry); the batch
  validator computes each constraint residual once instead of twice; the
  INFO trace line derives the penalty scale from the hard penalty already
  computed instead of walking the problem a second time; the Ocean config
  parse behind `is_available()` is memoised on the same credential
  fingerprint the redaction cache uses (env token plus every config file's
  path, mtime and size — any change still shows on the next call), so one
  capabilities query parses the INI once rather than once per D-Wave
  backend; and a Leap hybrid attempt fetches its sampler twice instead of
  three times and asks `min_time_limit` once instead of twice, the
  pre-submission check's value being reused by the solve of the same
  attempt. The submitted `time_limit` is unchanged.
- Candidate deduplication and ranking (spec §25) do the same work with
  fewer sorts; `samples`, `energies`, `counts` and the ranked top-k are
  identical for every input. Integer (CQM and decoded-integer) rows are
  packed into a few `uint64` words -- one field per column, sized to that
  column's range -- instead of one sort key per column, so a 200-variable
  result sorts on a handful of keys rather than 201 (10.8 → 3 ms on 2000
  rows, 54 → 18 ms on 10000). Ranking builds the assignment tie-break key
  and runs the full sort only on the rows that can still place in the
  top-k (found with a partition on ranking score, then objective value,
  keeping every row tied at the k-th place), instead of on every feasible
  candidate. Assignments that fit one packed word -- up to 64 binary
  variables, so every exact-backend enumeration -- deduplicate with a
  single stable sort on the word plus a vectorised per-group minimum,
  rather than a two-key sort, and the first-seen order is recovered with
  a counting pass instead of another sort (a 2^20-row enumeration
  deduplicates in about 230 ms instead of 490).
- The BQM compiler gathers every linear, quadratic and offset contribution
  in Python and lets dimod build the model once from numpy vectors,
  instead of feeding each penalty term into a live model through
  `add_linear` / `add_quadratic` (about a microsecond per call). Every
  bias is the same chain of float additions in the same order, so the
  compiled model -- variable order, offset, each linear and quadratic bias
  -- is bit-for-bit what it was (checked against the previous compiler on
  the 40 golden problems, the examples and 900-odd random binary and
  integer problems, including fractional equality coefficients and soft
  weights); a 200-integer-variable, 25-constraint problem compiles in
  about 31 ms instead of 65. The estimates and the validator analyse each
  inequality once per pass: `count_slack_bits` and `constraint_bit_count`
  accept the caller's `analyze_inequality` result, so
  `estimate_compiled_variables`, `estimate_encoded_interactions` and the
  inequality warnings no longer analyse the same constraint twice (full
  validation of the same problem about 2.0 ms instead of 3.0), and
  `expand_square` indexes its pair loop instead of slicing a new list per
  variable. Every estimate and warning is unchanged.

### Security

- The vendor job id in the Fujitsu DA backend's two INFO lines (`submitted
  job` and the solve line) now passes through credential redaction.
- The problem name (the user-supplied `problem.name`) in the INFO line every
  backend logs after a successful solve now passes through credential
  redaction; the exact, simulated-annealing and D-Wave backends previously
  logged it verbatim.

## [0.1.0] - 2026-09-11

First public release.

### Added

- Structured `OptimizationProblem` JSON contract (schema versions `1.0` and
  `1.1`) with binary and bounded-integer variables, linear/quadratic
  objectives, and hard/soft linear constraints.
- Deterministic pipeline: validate → compile → solve → re-validate every
  candidate against the original problem → rank. Feasibility and objective
  values are never inferred from solver energy.
- BQM compiler (automatic hard-constraint penalties, binary slack for
  inequalities, binary encoding of integers) and CQM compiler (native
  constraints, native integers).
- Six solver backends behind one `SolverBackend` protocol: `exact`,
  `simulated_annealing`, `dwave_qpu`, `leap_hybrid_bqm`, `leap_hybrid_cqm`
  and `fujitsu_da` (Fujitsu Digital Annealer over its HTTPS API, no vendor
  SDK).
- Backend recommendation (`recommend_backend` / `annealbridge recommend`),
  advisory only.
- MCP server (`annealbridge-mcp`, stdio and streamable-http) exposing
  `get_optimization_capabilities`, `validate_optimization_problem`,
  `recommend_backend` and `solve_optimization`.
- CLI (`annealbridge`) with `solve`, `validate`, `recommend`, `capabilities`
  and `export-schema`.
- Policy layer driven by `ANNEALBRIDGE_*` environment variables: remote
  execution and remote retries off by default, every resource limit enforced
  as an error and never silently clamped, no silent fallback between
  backends.
- Credential redaction for every remote backend, declared per backend and
  covered by a credential-leak test suite.
- Architecture tests that enforce the layering (`models ← validation ←
  penalty ← compiler ← solvers ← orchestration ← CLI / MCP`) and prove that a
  new backend plugs into the pipeline without changing it.
- A `Release` workflow (`.github/workflows/release.yml`) publishes the built
  distribution to PyPI through Trusted Publishing when a `vX.Y.Z` tag is
  pushed; it refuses a tag that does not match the `pyproject.toml` version.
  Running the workflow by hand publishes the same artefact to TestPyPI.
- The READMEs and `docs/mcp.md` document `uvx` / `pipx` as the way to run
  `annealbridge-mcp` without managing a virtual environment, give the
  `uvx`-based `claude_desktop_config.json` entry first, and show Claude Code's
  `claude mcp add`. The links in `README.md` are absolute, so they survive on
  the PyPI project page.
- `SolveResult` now reports the service's own wall clock: `elapsed_ms` on
  the result (whatever the status) and `compile_ms` / `solve_ms` /
  `validate_ms` on every recorded attempt, for local and remote backends
  alike. These are independent of the vendor-reported `metadata.timing_us`
  and are the only result fields that vary between runs.
- `SolveResult.optimality_proven`: `true` only when an exhaustive backend
  enumerated every assignment, so rank 1 is the global optimum of the
  ranking score. The mirror of `infeasibility_proven`.
- `SolveAttempt.compiled_variables` and `compiled_interactions`: the actual
  size of the model each attempt ran (`CompiledProblem.num_interactions` is
  new too), as opposed to the estimate `validate` reports.
- `SolveResult.annealbridge_version`: the package version that produced the
  result, read through the new `annealbridge.version.package_version()`
  that the MCP server now shares.
- `annealbridge solve` prints `Elapsed:` in the header and `Optimality
  proven:` after a successful solve.
- The `simulated_annealing` backend samples its reads in shards of 25, up to
  `ANNEALBRIDGE_SA_WORKERS` of them concurrently (new setting; unset detects
  the CPUs available to the process). The shard layout and shard seeds depend
  on `num_reads` and `seed` only, so a result is identical for any worker
  count or machine — the setting changes wall time, never an answer. Measured
  on twelve cores: 256 variables × 1000 reads about 4× faster; on one core
  the per-shard overhead is 4–15 %. No new dependency: the sampler's C++ loop
  already releases the GIL and keeps its RNG thread-local.
- CI `package` job builds the sdist and wheel and installs the wheel — core
  and `[mcp]` — into clean virtual environments on Ubuntu and Windows, then
  runs the new `scripts/check_install.py` from outside the checkout through
  the real console scripts; the same probe reproduces the check locally
  (2026-09-11 install verification, gap 4).
- `Solution.sample_count`: how many rows of that attempt's raw solver output
  carried this business assignment, before deduplication. It is not a
  confidence measure — an exhaustive backend enumerates every business
  assignment once per combination of the slack and integer-encoding bits, so
  the count reflects the compiled model's internal variables rather than the
  solution.
- `SolveResult.infeasibility`: an `infeasible` result now says *why*. It
  carries the last attempt's `closest_candidate` — the candidate with the
  smallest total hard violation, ties going to the one seen first, with the
  validator's full per-constraint evaluations — and one
  `hard_violation_rates` entry per hard constraint, in problem order, giving
  how many of that attempt's deduplicated candidates the constraint rejected.
  Everything is recomputed from the original problem, never from solver
  energy, and the field stays `null` on every other status and when the
  attempt returned no samples at all. `annealbridge solve` prints the closest
  candidate and the rates after `Infeasibility proven:`.
- **Every output field now documents itself.** `SolveResult`,
  `ProblemValidationResult`, `BackendRecommendationResult` and
  `OptimizationCapabilities` — and every model nested in them — carry a
  `description` on each field, so the MCP `outputSchema` an agent receives
  explains what a field *means* instead of only its type: that `energy` is
  for debugging and never feeds feasibility, the objective or the ranking;
  that the `*_ms` timings are the service's own wall clock and unrelated to
  the vendor's `metadata.timing_us`; that `sampler_reported_feasible` is the
  vendor's claim and not a verdict; and what each of the seven `status`
  values means. The wording follows `docs/output-format.md`, and tests fail
  if any property in the four schemas loses its description. Input schemas
  were already documented; no field, default or serialized value changed.

### Changed

- **`SolveResult.metadata` is no longer `null` on the local backends.**
  `exact` and `simulated_annealing` now report a `SolverExecutionMetadata`
  like every other backend: `backend`, `remote: false`, the `model_type` the
  service stamps, and — on `simulated_annealing` — `num_reads_requested`,
  the reads asked of the sampler (the whole request, not a shard). A local
  run has no vendor side, so `timing_us` is empty and `solver_id`,
  `effective_time_limit_seconds`, `sampler_reported_feasible` and the two
  QPU fields stay `null`. A consumer that read `metadata is null` as "this
  ran locally" must read `metadata.remote` instead. `metadata` is still
  `null` whenever no attempt completed — the service invents nothing.
- **Seeded `simulated_annealing` results differ from 0.1.0 when
  `num_reads > 25`.** Reads beyond one shard are sampled under seeds derived
  from the request seed (`numpy.random.SeedSequence`), so the sample set for a
  given seed is a different — equally reproducible — draw than before. A
  request of at most 25 reads passes the seed straight through and returns
  exactly what it did. The reproducibility contract is now explicit: the
  result depends on `(problem, num_reads, num_sweeps, seed)` and on the
  `dwave-samplers` / `numpy` versions, and on nothing else. A seed outside
  the sampler's `0 ≤ seed < 2³¹` is a `solver_error` for any `num_reads`, as
  the single-call path already reported it.
- Soft constraints are now scored from their **exact** residual: the
  feasibility tolerance that decides `satisfied` no longer zeroes
  `violation_amount` / `weighted_penalty` for a soft constraint, and
  `soft_violation_score` sums `weight × residual²` over every soft constraint.
  This is what the BQM and CQM compilers already charge for, so the ranking
  can no longer prefer an assignment the solver was paying to avoid (a
  residual inside the tolerance under a huge weight). `satisfied`,
  `soft_violations` and every hard-constraint field are unchanged.
- A hard inequality whose minimal left-hand side overshoots the right-hand
  side by less than the feasibility tolerance (only reachable at magnitude
  ≥ 1e12) now compiles with zero slack bits instead of escaping as an
  uncaught `ValueError` from `validate_problem_full()` /
  `OptimizationService.solve()`. Constraints outside the tolerance are still
  rejected as `TRIVIALLY_INFEASIBLE`.
- `ANNEALBRIDGE_HTTP_HOST` must be a non-empty value without whitespace and
  `ANNEALBRIDGE_HTTP_PORT` must be in `1–65535`; invalid values are a
  `SettingsError` (exit code 2) instead of an empty host silently binding
  every interface or an out-of-range port raising at socket bind. The
  `annealbridge-mcp --host` / `--port` overrides obey the same rule.
- Removed the `Typing :: Typed` trove classifier. The wheel ships no
  `py.typed` and the project runs no type checker yet, so the classifier
  promised more than the package delivers; it will return together with
  `py.typed` once a type check is part of CI.
- Documentation: the README quick start now solves a problem file the reader
  saves from the page instead of `examples/knapsack.json`, which the wheel
  does not ship; every `examples/` path in the docs is marked as a repository
  checkout path (2026-09-11 install verification, gap 2).

### Fixed

- `annealbridge-mcp` on a core-only install (no `[mcp]` extra) now prints a
  one-line install hint on stderr and exits with code 2 instead of crashing
  with `ModuleNotFoundError: No module named 'mcp'`; the console script now
  enters through `annealbridge.interfaces.mcp_entrypoint`, which imports
  nothing from `mcp` at module level.
  `python -m annealbridge.interfaces.mcp.server` is unchanged.
- CQM compilation now refuses a model whose objective or native constraint
  carries a non-finite bias (a soft weight × coefficient product that
  overflowed) with `COMPILATION_FAILED`, matching the BQM path; no backend
  is called with such a model.
- A `PENALTY_OVERFLOW` failure after at least one completed attempt keeps
  that attempt's solver metadata (solver id, timing, usage), as the other
  failure paths already did.
- Remote backends (`fujitsu_da`, `dwave_qpu`, `leap_hybrid_bqm`,
  `leap_hybrid_cqm`) declare their credentials to the shared redaction in
  their own constructor, so a directly constructed backend masks its key in
  error messages even when no `SolverRegistry` was built.
- The Ocean config-file token cache is keyed by the `DWAVE_API_TOKEN` value
  too, so unsetting the environment token after it was cached no longer
  leaves the config-file token unmasked.
- `LazySampler` closes an Ocean sampler it stops caching (credential
  rotation or `REMOTE_AUTH_FAILED` invalidation), best effort and outside
  its lock, instead of leaking the underlying client's threads and session
  in a long-running MCP server.
- Documentation: the `SolveAttempt` table now says that `unique_samples` and
  `feasible_samples` are both counted **after** deduplication, and that
  fewer entries in `solutions` than `feasible_samples` means the list was
  truncated to `solver.top_k`. The previous wording ("how many satisfied
  every hard constraint") read as a count of raw solver rows.

[Unreleased]: https://github.com/TheTsungYing/AnnealBridge/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/TheTsungYing/AnnealBridge/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/TheTsungYing/AnnealBridge/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/TheTsungYing/AnnealBridge/releases/tag/v0.1.0

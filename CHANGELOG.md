# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Changed

- Both READMEs now open with *Install* (the `uvx` host entry, `claude mcp
  add`, the pip matrix) and *A conversation* (what a user asks, the three
  tool calls the agent makes, what comes back), followed by the command
  line and Python examples; *How it works*, *Features* and the rest moved
  below them. The extra install paths live under *Installation in depth*.
- Dependency floors now match what the code actually calls:
  `pydantic-settings>=2.7` (`NoDecode`) and `dwave-system>=1.10`
  (`LeapHybridCQMSampler`). Older versions installed but failed on import or
  on the first hybrid CQM solve.
- The default test suite is strict: `xfail_strict` makes an xfail that starts
  passing a failure, and `filterwarnings = error` turns every warning into
  one. CI now runs on pushes to `main` and on pull requests only, cancels a
  superseded run on the same ref, and caches pip downloads between runs.
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

[Unreleased]: https://github.com/TheTsungYing/AnnealBridge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/TheTsungYing/AnnealBridge/releases/tag/v0.1.0

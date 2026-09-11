# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CI `package` job builds the sdist and wheel and installs the wheel — core
  and `[mcp]` — into clean virtual environments on Ubuntu and Windows, then
  runs the new `scripts/check_install.py` from outside the checkout through
  the real console scripts; the same probe reproduces the check locally
  (2026-09-11 install verification, gap 4).

### Changed

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

## [0.1.0] - 2026-09-09

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

[Unreleased]: https://github.com/OWNER/AnnealBridge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/OWNER/AnnealBridge/releases/tag/v0.1.0

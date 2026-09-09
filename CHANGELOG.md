# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

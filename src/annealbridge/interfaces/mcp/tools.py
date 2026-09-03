"""The four MCP tools (Phase 2 spec §22–§24, 3a §24).

Each tool is a thin adapter: fetch the wired-up state, call into the core, and
return the core's Pydantic model so the SDK derives the structured output from
the return type annotation. No optimization logic here.
"""

import anyio

from annealbridge.models import OptimizationProblem, SolveResult
from annealbridge.validation import (
    BackendRecommendationResult,
    ProblemValidationResult,
)

from annealbridge.interfaces.mcp.models import (
    OptimizationCapabilities,
    build_capabilities,
)
from annealbridge.interfaces.mcp.server import get_state, mcp


@mcp.tool()
async def get_optimization_capabilities() -> OptimizationCapabilities:
    """Describe what this optimization server accepts and which solver backends
    are usable right now.

    Call this before formulating a problem to learn the supported problem
    schema (variable types, constraint operators, objective terms and the full
    JSON schema) and, per backend, whether it is installed/configured
    (available), whether server policy permits it (enabled), and its resource
    limits. This performs no solving and no network requests.
    """
    state = get_state()
    return build_capabilities(state.registry, state.policy)


@mcp.tool()
async def validate_optimization_problem(
    problem: OptimizationProblem,
) -> ProblemValidationResult:
    """Check a structured optimization problem without solving it.

    Returns semantic errors (each with a recommended_action), advisory
    warnings, and an estimate of the compiled variable count including slack
    bits. Call this before solve_optimization when planning to use a remote
    backend, so problems can be fixed before spending quota. Nothing is
    compiled or solved and no network requests are made.
    """
    return get_state().service.validate(problem)


@mcp.tool()
async def recommend_backend(problem: OptimizationProblem) -> BackendRecommendationResult:
    """Rank the solver backends of this server for a given problem, without solving it.

    Advisory only: solve_optimization always uses problem.solver.backend exactly as
    given and never substitutes a backend. Each entry reports whether the backend is
    usable right now (installed, credentialed, permitted by policy, within limits),
    the model type it would compile to, deterministic reason codes, blocking errors,
    and the same warnings validate_optimization_problem would give for that backend.
    Based only on the problem structure, backend capabilities and server policy —
    no cost estimation, no network requests, no quota consumed.
    """
    return get_state().service.recommend(problem)


@mcp.tool()
async def solve_optimization(problem: OptimizationProblem) -> SolveResult:
    """Solve a structured binary combinatorial optimization problem.

    Call this only after translating the user's request into explicit binary
    variables, an objective (linear/quadratic, minimize or maximize), and hard
    or soft linear constraints. Do not pass natural-language requirements.

    Inequality constraints (<=, >=) require integer coefficients and right-hand
    sides. Soft constraint weights are in objective units.

    Leave solver.penalty_multiplier at its default unless a previous result was
    infeasible on a remote backend; hard constraint penalties are managed by the
    server. Use get_optimization_capabilities to see which backends are enabled;
    call validate_optimization_problem first when planning to use a remote backend.
    Use recommend_backend to compare backends; the choice remains yours.

    Returns ranked feasible solutions with per-constraint evaluations, or a
    structured error with a recommended_action.
    """
    state = get_state()
    return await anyio.to_thread.run_sync(state.service.solve, problem)

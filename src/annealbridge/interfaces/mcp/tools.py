"""The four MCP tools (Phase 2 spec §22–§24, 3a §24).

Each tool is a thin adapter: fetch the wired-up state, call into the core, and
return the core's Pydantic model so the SDK derives the structured output from
the return type annotation. No optimization logic here.

Every core call that does real work (validate, recommend, solve) runs in a
worker thread through ``anyio.to_thread.run_sync`` (2026-09-09 review F-15):
they are synchronous and CPU-bound, and a large problem — ``recommend`` runs
the full validation once per registered backend — would otherwise freeze the
event loop and with it every other client of the server.
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

    supported_variable_types lists the variable types a problem may declare —
    "binary" and "integer" — and schema_versions lists every problem schema
    version this server accepts, newest last. Integer variables are only
    allowed when the problem carries "version": "1.1" at its top level;
    "version": "1.0" accepts binary variables only.
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
    compiled or solved and no network requests are made. solve_optimization
    reports the same warnings for the same backend, so skipping this call
    never hides them; calling it first only saves the solve.

    A field the schema does not declare is a tool error naming its path,
    never ignored: check the problem_json_schema from
    get_optimization_capabilities before inventing one.

    The estimate follows the model type the chosen backend compiles to. On a
    bqm backend it counts the slack bits of every inequality constraint plus
    the binary-encoding bits of every integer variable, so a wider
    lower_bound..upper_bound range costs more compiled variables. On a cqm
    backend integer variables are native and no encoding bits are counted.
    """
    return await anyio.to_thread.run_sync(get_state().service.validate, problem)


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

    A problem with integer variables adds reason codes: R_INTEGER_NATIVE when the
    backend compiles to cqm and takes integers as they are, R_INTEGER_ENCODED when
    it compiles to bqm and must binary-encode them, and R_INTEGER_BLOWUP when that
    encoding also raises an INTEGER_QUADRATIC_BLOWUP warning — a backend carrying
    that code is ranked after the ones without it.
    """
    return await anyio.to_thread.run_sync(get_state().service.recommend, problem)


@mcp.tool()
async def solve_optimization(problem: OptimizationProblem) -> SolveResult:
    """Solve a structured binary or bounded-integer combinatorial optimization
    problem.

    Call this only after translating the user's request into explicit binary or
    bounded-integer variables, an objective (linear/quadratic, minimize or
    maximize), and hard or soft linear constraints. Do not pass natural-language
    requirements.

    An integer variable is declared with "type": "integer" plus integer
    lower_bound and upper_bound (both required), and the problem must then carry
    "version": "1.1" at its top level. A backend that compiles to bqm encodes
    each integer in binary, so the compiled size grows with the range of the
    bounds; a backend that compiles to cqm takes integers natively. Integer
    values come back as ints inside their declared bounds.

    Inequality constraints (<=, >=) require integer coefficients and right-hand
    sides. Soft constraint weights are in objective units.

    Leave solver.penalty_multiplier at its default unless a previous result was
    infeasible on a remote backend; hard constraint penalties are managed by the
    server. Use get_optimization_capabilities to see which backends are enabled;
    call validate_optimization_problem first when planning to use a remote backend.
    Use recommend_backend to compare backends; the choice remains yours.

    Returns ranked feasible solutions with per-constraint evaluations, or a
    structured error with a recommended_action. Whatever the status, the
    result's warnings are the same advisory warnings
    validate_optimization_problem gives for this backend (an ignored seed or
    parameter, a wide integer range, a negligible soft weight, ...) followed
    by any raised during the run; read them before trusting a weaker answer
    than expected. A field the schema does not declare is a tool error naming
    its path, never ignored.
    """
    state = get_state()
    return await anyio.to_thread.run_sync(state.service.solve, problem)

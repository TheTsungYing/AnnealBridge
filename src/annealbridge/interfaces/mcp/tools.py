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

from typing import Annotated

import anyio
from pydantic import Field

from annealbridge.interfaces.mcp.models import (
    OptimizationCapabilities,
    build_capabilities,
)
from annealbridge.interfaces.mcp.server import get_state, mcp
from annealbridge.models import OptimizationProblem, SolveResult
from annealbridge.validation import (
    BackendRecommendationResult,
    ProblemValidationResult,
)


@mcp.tool()
async def get_optimization_capabilities(
    include_schema: Annotated[
        bool,
        Field(
            description=(
                "Whether to include the full OptimizationProblem JSON Schema "
                "as problem_json_schema. The default false keeps the response "
                "small; pass true only when the schema itself is needed, e.g. "
                "for integer variables or an unfamiliar field."
            )
        ),
    ] = False,
) -> OptimizationCapabilities:
    """Describe what this optimization server accepts and which solver backends
    are usable right now.

    Call this when you need the supported problem vocabulary (variable types,
    constraint operators, objective terms, accepted schema versions) or, per
    backend, whether it is installed/configured (available), whether server
    policy permits it (enabled), and its resource limits. It is not required
    before every problem: for a small binary problem on a local backend the
    example in the server instructions already shows the document shape. This
    performs no solving and no network requests.

    The full problem JSON schema is several times the size of everything
    else, so problem_json_schema is null unless include_schema is true. Ask
    for it when the document needs more than the example shows — integer
    variables, soft constraints, solver preferences — or before inventing a
    field; every field description is in it.

    supported_variable_types lists the variable types a problem may declare —
    "binary" and "integer" — and schema_versions lists every problem schema
    version this server accepts, newest last. Integer variables are only
    allowed when the problem carries "version": "1.1" at its top level;
    "version": "1.0" accepts binary variables only.
    """
    state = get_state()
    return build_capabilities(
        state.registry, state.policy, include_schema=include_schema
    )


@mcp.tool()
async def validate_optimization_problem(
    problem: OptimizationProblem,
) -> ProblemValidationResult:
    """Check a structured optimization problem without solving it.

    Returns semantic errors (each with a recommended_action), advisory
    warnings, and an estimate of the compiled variable count including slack
    bits. Call this before solve_optimization when planning to use a remote
    backend, so problems can be fixed before spending quota, or when the
    problem is large (many variables, wide integer ranges). On a local backend
    solve_optimization can be called directly: an invalid document returns
    status invalid_problem with the same errors and recommended_action.
    Nothing is compiled or solved and no network requests are made.
    solve_optimization reports the same warnings for the same backend, so
    skipping this call never hides them; calling it first only saves the
    solve.

    A field the schema does not declare is a tool error naming its path,
    never ignored: check the schema (get_optimization_capabilities with
    include_schema: true returns it as problem_json_schema) before inventing
    one.

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

    The entries are ordered best first, so the first usable entry is the default
    choice when the user named no backend. Among backends of the same kind, two
    reason codes come from a backend's declared, measured structural preference:
    R_DENSE_STRENGTH when it declares that it reaches the same energy as its
    peers in a fraction of the time on large dense unconstrained models and this
    problem is one, R_PENALTY_WEAKNESS when it declares a lower hit rate on
    models whose hard constraints compile to penalties and this problem has an
    effective hard constraint. Both shapes are bqm-path shapes, so a backend that
    compiles to cqm is never matched. A backend carrying the first is ranked
    ahead of its neighbours, one carrying the second behind them; neither changes
    which backends are usable.
    """
    return await anyio.to_thread.run_sync(get_state().service.recommend, problem)


@mcp.tool()
async def solve_optimization(problem: OptimizationProblem) -> SolveResult:
    """Solve a structured binary or bounded-integer combinatorial optimization
    problem.

    Use this when the user asks which items to take within a budget, weight or
    capacity, how to assign people or jobs to seats, shifts or machines, in
    which order to visit a handful of places, or how to split things into
    groups or pick a subset that meets several requirements at once — anything
    expressible as yes/no or bounded-count decisions with a linear or quadratic
    score and linear rules, even when the user never says "optimization".

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
    server. Call validate_optimization_problem first when planning to use a
    remote backend or when the problem is large; on a local backend an invalid
    document comes back as status invalid_problem with the same errors and
    recommended_action, so solving directly spends nothing. Use
    get_optimization_capabilities when you need the backend list or the full
    schema. Use recommend_backend to compare backends; the choice remains yours.

    Returns ranked feasible solutions with per-constraint evaluations, or a
    structured error with a recommended_action. When the status is infeasible,
    read infeasibility for the candidate that came closest to feasibility and
    the share of candidates each hard constraint rejected, so the answer can
    name the binding requirement instead of only reporting failure. Whatever
    the status, the result's warnings are the same advisory warnings
    validate_optimization_problem gives for this backend (an ignored seed or
    parameter, a wide integer range, a negligible soft weight, ...) followed
    by any raised during the run; read them before trusting a weaker answer
    than expected. A field the schema does not declare is a tool error naming
    its path, never ignored. If the user did not name a backend, say in the
    answer which backend ran and why.
    """
    state = get_state()
    return await anyio.to_thread.run_sync(state.service.solve, problem)

"""Command-line interface (spec §29).

Presentation layer only: parses arguments, loads the problem JSON, calls
:class:`OptimizationService`, and formats the result for the console.
No optimization logic lives here (spec §45).
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Optional, TypeVar, get_args

import typer
from pydantic import BaseModel, ValidationError

from annealbridge.config import SettingsError
from annealbridge.interfaces.capabilities import BackendCapability, build_capabilities
from annealbridge.interfaces.composition import (
    AppState,
    build_state,
    exit_on_settings_error,
)
from annealbridge.models import (
    OptimizationProblem,
    SolveError,
    SolveResult,
    SolverPreferences,
)
from annealbridge.orchestration import OptimizationService
from annealbridge.validation import (
    BackendRecommendationResult,
    ProblemValidationResult,
)
from annealbridge.version import package_version

app = typer.Typer(
    help="Optimization Tool Middleware CLI",
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _print_version(value: bool) -> None:
    """Eager ``--version``: print the installed version and exit 0.

    Runs while the options are parsed, before any command body, so it reads
    no ``ANNEALBRIDGE_*`` setting and answers even when they are invalid.
    """
    if value:
        typer.echo(f"annealbridge {package_version()}")
        raise typer.Exit()


@app.callback()
def _global_options(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    # Deliberately no docstring: the top-level help text stays the one given
    # to typer.Typer(help=...). The eager callback does all the work.
    pass


def _format_number(value: float) -> str:
    """Render integral floats without a trailing ``.0`` (e.g. ``17``, not ``17.0``)."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _load_problem(path: Path) -> OptimizationProblem:
    """Load and parse a problem JSON file, exiting with a friendly error on failure."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        typer.echo(f"Error: cannot read '{path}': {exc}", err=True)
        raise typer.Exit(code=2)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        typer.echo(f"Error: '{path}' is not valid JSON: {exc}", err=True)
        raise typer.Exit(code=2)
    try:
        return OptimizationProblem.model_validate(data)
    except ValidationError as exc:
        typer.echo(f"Error: '{path}' is not a valid optimization problem:", err=True)
        for err in exc.errors():
            location = ".".join(str(part) for part in err["loc"])
            if err["type"] == "extra_forbidden":
                # The input models refuse unknown keys (models/strict.py):
                # say so in the problem's own vocabulary and point at the
                # schema, instead of pydantic's generic wording.
                message = (
                    "unknown field, not in the problem schema (see "
                    "'annealbridge export-schema')"
                )
            else:
                message = err["msg"]
            typer.echo(f"  {location}: {message}", err=True)
        raise typer.Exit(code=2)


def _build_state() -> AppState:
    """Wire the service from the environment, exiting cleanly on bad settings."""
    try:
        return build_state()
    except SettingsError as exc:
        exit_on_settings_error(exc)


def _override_backend(problem: OptimizationProblem, backend: str) -> OptimizationProblem:
    """Return a copy of ``problem`` with ``solver.backend`` replaced by ``backend``."""
    try:
        preferences = SolverPreferences.model_validate(
            {**problem.solver.model_dump(), "backend": backend}
        )
    except ValidationError:
        known = get_args(SolverPreferences.model_fields["backend"].annotation)
        expected = ", ".join(f"'{name}'" for name in known)
        typer.echo(
            f"Error: unknown backend '{backend}' (expected one of {expected})",
            err=True,
        )
        raise typer.Exit(code=2)
    return problem.model_copy(update={"solver": preferences})


def _render_errors(items: Sequence[SolveError], lines: list[str], title: str) -> None:
    """Append an ``[CODE] path: message`` block with recommended actions.

    The single renderer for structured errors and warnings shared by all three
    commands: ``solve`` (:func:`_render_human`), ``validate``
    (:func:`_render_validation`) and ``recommend``
    (:func:`_render_recommendation`).
    """
    lines.append("")
    lines.append(f"{title} ({len(items)}):")
    for item in items:
        location = f" {item.path}" if item.path else ""
        lines.append(f"  [{item.code}]{location}: {item.message}")
        if item.recommended_action:
            lines.append(f"    recommended action: {item.recommended_action}")


def _render_human(problem: OptimizationProblem, result: SolveResult) -> str:
    """Format a SolveResult as the human-readable report from spec §29."""
    lines = [
        f"Problem:   {problem.name}",
        f"Backend:   {result.backend or '-'}",
        f"Status:    {result.status}",
        f"Attempts:  {len(result.attempts)}",
    ]
    if result.elapsed_ms is not None:
        lines.append(f"Elapsed:   {_format_number(round(result.elapsed_ms, 1))} ms")

    if result.status == "success":
        best = result.solutions[0]
        lines.append("")
        lines.append(f"Best solution (rank {best.rank})")
        lines.append(
            f"  objective ({result.objective_direction}):  "
            f"{_format_number(best.objective_value)}"
        )
        lines.append(
            f"  soft violation score:  {_format_number(best.soft_violation_score)}"
        )
        # Only a post-processing product is called out: "solver" is the
        # value every solve without solver.postprocess returns, so the
        # default report stays exactly as it was.
        if best.source != "solver":
            lines.append(f"  source:  {best.source} (post-processing)")
        for name in sorted(best.variables):
            lines.append(f"  {name} = {best.variables[name]}")

        hard = [e for e in best.constraint_evaluations if e.constraint_type == "hard"]
        soft = [e for e in best.constraint_evaluations if e.constraint_type == "soft"]
        hard_satisfied = sum(1 for e in hard if e.satisfied)
        soft_violated = sum(1 for e in soft if not e.satisfied)
        lines.append("")
        lines.append(f"Hard constraints: {hard_satisfied} / {len(hard)} satisfied")
        lines.append(f"Soft constraints: {soft_violated} violations")
        lines.append(
            f"Optimality proven: {'yes' if result.optimality_proven else 'no'}"
        )

    elif result.status in ("invalid_problem", "resource_limit_exceeded"):
        title = (
            "Validation errors"
            if result.status == "invalid_problem"
            else "Resource limit errors"
        )
        _render_errors(result.errors, lines, title)

    elif result.status == "infeasible":
        lines.append("")
        for attempt in result.attempts:
            lines.append(
                f"  attempt {attempt.attempt}: "
                f"penalty="
                f"{'-' if attempt.penalty is None else _format_number(attempt.penalty)}, "
                f"samples={attempt.samples_received}, "
                f"unique={attempt.unique_samples}, "
                f"feasible={attempt.feasible_samples}"
            )
            stats = attempt.postprocess
            if stats is not None:
                stopped = (
                    f", stopped at: {', '.join(stats.limit_reached)}"
                    if stats.limit_reached
                    else ""
                )
                lines.append(
                    f"    post-processing: selected={stats.candidates_selected}, "
                    f"repaired={stats.repair_succeeded}/{stats.repair_attempted}, "
                    f"improved={stats.local_search_improved}/"
                    f"{stats.local_search_started}{stopped}"
                )
        proven = "yes" if result.infeasibility_proven else "no"
        lines.append(f"Infeasibility proven: {proven}")
        if result.infeasibility is not None:
            closest = result.infeasibility.closest_candidate
            lines.append(
                "Closest candidate (hard violation total "
                f"{_format_number(closest.hard_violation_total)}):"
            )
            # Name-sorted, like the success block's variable listing.
            for name in sorted(closest.variables):
                lines.append(f"  {name} = {closest.variables[name]}")
            lines.append("Hard constraint violation rates:")
            for rate in result.infeasibility.hard_violation_rates:
                lines.append(
                    f"  {rate.constraint_id}: "
                    f"{rate.violated_candidates} / {rate.candidates} "
                    f"({_format_number(rate.violated_fraction * 100)}%)"
                )
        if result.message:
            lines.append(result.message)

    elif result.status in ("backend_unavailable", "configuration_error"):
        title = (
            "Backend unavailable"
            if result.status == "backend_unavailable"
            else "Configuration error"
        )
        _render_errors(result.errors, lines, title)

    elif result.status == "solver_error":
        if result.errors:
            _render_errors(result.errors, lines, "Solver error")
        else:
            lines.append("")
            lines.append(f"Solver error: {result.message or 'unknown error'}")

    if result.warnings:
        _render_errors(result.warnings, lines, "Warnings")

    return "\n".join(lines)


# The arguments ``solve``, ``validate`` and ``recommend`` share, declared
# once. Each ``--json`` option keeps its own declaration: its help names the
# result model that command prints.
ProblemFileArgument = Annotated[
    Path, typer.Argument(help="Path to an OptimizationProblem JSON file")
]
BackendOption = Annotated[
    Optional[str],
    typer.Option("--backend", help="Override solver.backend from the JSON"),
]

ResultT = TypeVar("ResultT", bound=BaseModel)


def _run(
    problem_file: Path,
    backend: Optional[str],
    json_output: bool,
    call: Callable[[OptimizationService, OptimizationProblem], ResultT],
    render: Callable[[OptimizationProblem, ResultT], str],
    failed: Callable[[ResultT], bool],
) -> None:
    """The shared body of ``solve``, ``validate`` and ``recommend``.

    Loads the problem, applies ``--backend`` when one is given, wires the
    service from the environment — the same composition root as the MCP
    server (spec §30) — and makes the command's one delegating ``call``.
    The result is printed as JSON or through the command's ``render``, and
    the process exits 1 when ``failed`` says so. No optimization logic
    lives here.
    """
    problem = _load_problem(problem_file)
    if backend is not None:
        problem = _override_backend(problem, backend)

    result = call(_build_state().service, problem)

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(render(problem, result))

    if failed(result):
        raise typer.Exit(code=1)


@app.command()
def solve(
    problem_file: ProblemFileArgument,
    backend: BackendOption = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the full SolveResult as JSON")
    ] = False,
) -> None:
    """Solve an optimization problem loaded from a JSON file."""
    _run(
        problem_file,
        backend,
        json_output,
        call=lambda service, problem: service.solve(problem),
        render=_render_human,
        failed=lambda result: result.status != "success",
    )


def _render_validation(
    problem: OptimizationProblem, result: ProblemValidationResult
) -> str:
    """Format a ProblemValidationResult as the human-readable report (3a §10)."""
    model_type = result.model_type or "unknown"
    lines = [
        f"Problem:   {problem.name}",
        f"Backend:   {problem.solver.backend}  (model type: {model_type})",
        f"Valid:     {'yes' if result.valid else 'no'}",
    ]
    if result.valid:
        lines.append(
            f"Estimated compiled variables: {result.estimated_compiled_variables}"
        )
        lines.append(f"Objective scale: {_format_number(result.objective_scale)}")

    for title, items in (("Errors", result.errors), ("Warnings", result.warnings)):
        if items:
            _render_errors(items, lines, title)
    return "\n".join(lines)


@app.command()
def validate(
    problem_file: ProblemFileArgument,
    backend: BackendOption = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the full ProblemValidationResult as JSON"),
    ] = False,
) -> None:
    """Validate an optimization problem without solving it."""
    # The same one-line delegation as MCP.
    _run(
        problem_file,
        backend,
        json_output,
        call=lambda service, problem: service.validate(problem),
        render=_render_validation,
        failed=lambda result: not result.valid,
    )


def _render_recommendation(
    problem: OptimizationProblem, result: BackendRecommendationResult
) -> str:
    """Format a BackendRecommendationResult as the table from 3a §25.

    Pure presentation: the order and every reason come from the service.
    """
    lines = [f"Problem:   {problem.name}"]
    if not result.valid:
        lines.append("Valid:     no")
        _render_errors(result.errors, lines, "Errors")
        return "\n".join(lines)

    lines.append(
        "Advisory:  recommendations only; `solve` uses solver.backend as given"
    )
    lines.append("")
    entries = result.recommendations
    name_width = max(len("Backend"), *(len(e.backend) for e in entries)) + 2
    lines.append(
        f"{'Rank':<6}{'Backend':<{name_width}}{'Usable':<8}{'Model':<7}Reasons"
    )
    for entry in entries:
        row = (
            f"{entry.rank:<6}"
            f"{entry.backend:<{name_width}}"
            f"{'yes' if entry.usable else 'no':<8}"
            f"{entry.model_type or '-':<7}"
            f"{', '.join(entry.reasons)}"
        )
        if entry.blocking:
            codes = ", ".join(error.code for error in entry.blocking)
            row = f"{row}   [{codes}]"
        lines.append(row.rstrip())
    return "\n".join(lines)


@app.command()
def recommend(
    problem_file: ProblemFileArgument,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the full BackendRecommendationResult as JSON"),
    ] = False,
) -> None:
    """Rank the backends for a problem without solving it.

    Advisory only: ``solve`` still uses solver.backend exactly as given.
    """
    # The same one-line delegation as MCP; no ranking logic lives here, and
    # there is no --backend: every backend is ranked.
    _run(
        problem_file,
        None,
        json_output,
        call=lambda service, problem: service.recommend(problem),
        render=_render_recommendation,
        failed=lambda result: not result.valid,
    )


def _format_limits(limits: dict[str, float | int]) -> str:
    """Render policy limits as ``key=value`` pairs (spec §30 table style)."""
    parts = []
    for key, value in limits.items():
        if key == "max_time_seconds":
            parts.append(f"max_time={_format_number(value)}s")
        else:
            parts.append(f"{key}={_format_number(value)}")
    return ", ".join(parts)


def _render_capabilities_table(backends: list[BackendCapability]) -> str:
    """Format the backend capability rows as the table from spec §30."""
    name_width = max(len("Backend"), *(len(b.name) for b in backends)) + 2
    header = (
        f"{'Backend':<{name_width}}{'Available':<11}{'Enabled':<9}"
        f"{'Remote':<8}Limits"
    )
    lines = [header]
    for b in backends:
        row = (
            f"{b.name:<{name_width}}"
            f"{'yes' if b.available else 'no':<11}"
            f"{'yes' if b.enabled else 'no':<9}"
            f"{'yes' if b.remote else 'no':<8}"
            f"{_format_limits(b.limits)}"
        )
        if b.unavailable_reason:
            row = f"{row}  ({b.unavailable_reason})"
        lines.append(row.rstrip())
    return "\n".join(lines)


@app.command()
def capabilities() -> None:
    """List backends with availability, policy status and limits."""
    state = _build_state()
    caps = build_capabilities(state.registry, state.policy)
    typer.echo(_render_capabilities_table(caps.backends))


@app.command("export-schema")
def export_schema() -> None:
    """Print the OptimizationProblem JSON schema."""
    typer.echo(json.dumps(OptimizationProblem.model_json_schema(), indent=2))


@app.command(
    "mcp",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
    add_help_option=False,
)
def mcp(ctx: typer.Context) -> None:
    """Run the MCP server (needs the "mcp" extra; same options as annealbridge-mcp)."""
    # Every argument is handed to the server's own parser, ``--help`` and
    # ``--version`` included, so this subcommand and ``annealbridge-mcp``
    # answer identically. The import is deferred and goes through
    # ``mcp_entrypoint``: a core-only install must still run every other CLI
    # command, and a missing extra must be one stderr line and exit 2 here too.
    from annealbridge.interfaces.mcp_entrypoint import run

    run(ctx.args, prog="annealbridge mcp")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

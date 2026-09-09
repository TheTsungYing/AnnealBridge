"""Command-line interface (spec §29).

Presentation layer only: parses arguments, loads the problem JSON, calls
:class:`OptimizationService`, and formats the result for the console.
No optimization logic lives here (spec §45).
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, get_args

import typer
from pydantic import ValidationError

from annealbridge.config import SettingsError
from annealbridge.interfaces.capabilities import BackendCapability, build_capabilities
from annealbridge.interfaces.composition import AppState, build_state
from annealbridge.models import (
    OptimizationProblem,
    SolveError,
    SolveResult,
    SolverPreferences,
)
from annealbridge.validation import (
    BackendRecommendationResult,
    ProblemValidationResult,
)

app = typer.Typer(
    help="Optimization Tool Middleware CLI",
    add_completion=False,
    pretty_exceptions_enable=False,
)


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
            typer.echo(f"  {location}: {err['msg']}", err=True)
        raise typer.Exit(code=2)


def _build_state() -> AppState:
    """Wire the service from the environment, exiting cleanly on bad settings."""
    try:
        return build_state()
    except SettingsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2)


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
        for name in sorted(best.variables):
            lines.append(f"  {name} = {best.variables[name]}")

        hard = [e for e in best.constraint_evaluations if e.constraint_type == "hard"]
        soft = [e for e in best.constraint_evaluations if e.constraint_type == "soft"]
        hard_satisfied = sum(1 for e in hard if e.satisfied)
        soft_violated = sum(1 for e in soft if not e.satisfied)
        lines.append("")
        lines.append(f"Hard constraints: {hard_satisfied} / {len(hard)} satisfied")
        lines.append(f"Soft constraints: {soft_violated} violations")

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
        proven = "yes" if result.infeasibility_proven else "no"
        lines.append(f"Infeasibility proven: {proven}")
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


@app.command()
def solve(
    problem_file: Path = typer.Argument(
        ..., help="Path to an OptimizationProblem JSON file"
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", help="Override solver.backend from the JSON"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the full SolveResult as JSON"
    ),
) -> None:
    """Solve an optimization problem loaded from a JSON file."""
    problem = _load_problem(problem_file)
    if backend is not None:
        problem = _override_backend(problem, backend)

    # Same composition root as the MCP server (spec §30).
    result = _build_state().service.solve(problem)

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(_render_human(problem, result))

    if result.status != "success":
        raise typer.Exit(code=1)


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
    problem_file: Path = typer.Argument(
        ..., help="Path to an OptimizationProblem JSON file"
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", help="Override solver.backend from the JSON"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the full ProblemValidationResult as JSON"
    ),
) -> None:
    """Validate an optimization problem without solving it (3a §10)."""
    problem = _load_problem(problem_file)
    if backend is not None:
        problem = _override_backend(problem, backend)

    # Same composition root and the same one-line delegation as MCP.
    result = _build_state().service.validate(problem)

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(_render_validation(problem, result))

    if not result.valid:
        raise typer.Exit(code=1)


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
    problem_file: Path = typer.Argument(
        ..., help="Path to an OptimizationProblem JSON file"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the full BackendRecommendationResult as JSON"
    ),
) -> None:
    """Rank the backends for a problem without solving it (3a §25).

    Advisory only: ``solve`` still uses solver.backend exactly as given.
    """
    problem = _load_problem(problem_file)

    # Same composition root and the same one-line delegation as MCP; no
    # ranking logic lives here.
    result = _build_state().service.recommend(problem)

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(_render_recommendation(problem, result))

    if not result.valid:
        raise typer.Exit(code=1)


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
    """List backends with availability, policy status and limits (spec §30)."""
    state = _build_state()
    caps = build_capabilities(state.registry, state.policy)
    typer.echo(_render_capabilities_table(caps.backends))


@app.command("export-schema")
def export_schema() -> None:
    """Print the OptimizationProblem JSON schema (spec §38)."""
    typer.echo(json.dumps(OptimizationProblem.model_json_schema(), indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()

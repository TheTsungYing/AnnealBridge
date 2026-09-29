"""Building blocks shared by the problem validator and the template expander.

Moved unchanged out of ``problem_validator`` (schema 1.3 spec §14.8) so the
expander (``validation/expansion.py``) can report the validator's own codes
with the validator's own wording and rules, without importing the
validator; ``problem_validator`` re-exports every name, so its callers and
tests are unaffected. Depends on ``annealbridge.models`` only.
"""

from collections import Counter

from annealbridge.models import Objective, SolveError, Variable, catalog_error

# 3b §7: integer bounds must lie within ±(2^31-1) so every encoded value,
# every product of two values and every float64 evaluation stays exact.
INTEGER_BOUND_LIMIT = 2**31 - 1

# Fixed categorical guidance per warning code (mirrors the error catalog's
# style; §13.2's RECOMMENDED_ACTIONS stays an error-code vocabulary). The
# text describes backends by capability, never by name (3a §13.2).
_WARNING_RECOMMENDED_ACTIONS: dict[str, str] = {
    "SOFT_WEIGHT_SMALL": (
        "The soft constraint's weight is tiny compared to the objective's "
        "scale, so it will barely influence solutions; raise the weight if "
        "the preference matters."
    ),
    "LARGE_SLACK_RANGE": (
        "This inequality needs many slack bits on a BQM backend, which "
        "enlarges the compiled model; tighten the bound or split the "
        "constraint if possible."
    ),
    "EXACT_NEAR_LIMIT": (
        "The estimated compiled size is close to the exhaustive backend's "
        "variable limit; consider a local heuristic backend before the "
        "problem grows."
    ),
    "EXACT_OVER_LIMIT": (
        "The estimated compiled size exceeds the exhaustive backend's "
        "variable limit; solving will fail with resource_limit_exceeded — "
        "reduce the problem or use a local heuristic backend."
    ),
    "DENSE_FOR_QPU": (
        "The problem is likely too dense or too large to minor-embed on a "
        "backend that requires embedding; consider a hybrid backend or a "
        "local heuristic backend."
    ),
    "SEED_IGNORED": (
        "This backend does not support seeding; remove solver.seed or use a "
        "backend that supports seeding if reproducibility is required."
    ),
    "PARAMETER_IGNORED": (
        "This parameter has no effect on the selected backend and will be "
        "ignored; remove it to avoid confusion."
    ),
    "DUPLICATE_TERM_MERGED": (
        "Duplicate objective terms are summed by the compiler; merge them in "
        "the input if the duplication is unintentional."
    ),
    "REDUNDANT_CONSTRAINT": (
        "This inequality is always satisfied and adds nothing to the model; "
        "remove it, or fix its bound if it was meant to restrict solutions."
    ),
    # 3b §9.3 / §19 (the bit threshold is LARGE_INTEGER_BITS_THRESHOLD and
    # is reported in the message, never in this fixed guidance).
    "LARGE_INTEGER_RANGE": (
        "An integer variable needs more encoding bits on a BQM backend than "
        "is comfortable; tighten its bounds, rescale its unit, or use a "
        "backend that accepts integer variables natively."
    ),
    "INTEGER_QUADRATIC_BLOWUP": (
        "Binary-encoding the integer variables produces many quadratic "
        "interactions on a BQM backend; prefer a backend that accepts "
        "integer variables natively (model type cqm), or reduce the ranges."
    ),
    # 2026-09-09 review (F-04 / F-12): the soft counterpart of
    # TRIVIALLY_INFEASIBLE. Legal (the weight is simply always paid), so a
    # warning; judged with the same tolerance, on both compiler paths alike.
    "SOFT_ALWAYS_VIOLATED": (
        "This soft constraint can never be satisfied within the variables' "
        "bounds: every solution pays its penalty and the weight only rewards "
        "the smallest violation; remove it, or fix its bound or coefficients "
        "if it was meant to be attainable."
    ),
    # Schema 1.2 spec §6.3: only a declared cardinality constraint gets the
    # slack-free pairwise encoding, so an equivalent linear one is pointed
    # at the declaration. The count is spelled out: this text holds no
    # digit (tests/unit/test_error_catalog.py).
    "CARDINALITY_FORM_AVAILABLE": (
        "Declare this constraint in cardinality_constraints with operator "
        '"<=" and rhs one to let a BQM backend encode it without a slack '
        "variable; the meaning is unchanged."
    ),
    # Schema 1.3 spec §14.10: the template expander's own advisories. Each
    # is one warning per template, its count in the message.
    "EMPTY_TEMPLATE_EXPANSION": (
        "The template produced nothing; check its where conditions and index "
        "sets, or remove it."
    ),
    "TEMPLATE_BOUNDARY_SKIPPED": (
        "Items whose shifted index runs past the end of a linear index set "
        "are left out, which relaxes a constraint template at the boundary; "
        "declare the set cyclic if it should wrap around, or ignore this if "
        "leaving them out is intended."
    ),
    "TEMPLATE_TERMS_MERGED": (
        "A generated linear constraint names the same variable more than "
        "once and the compiler sums the coefficients; check shifts on cyclic "
        "sets and term templates that reach the same variable."
    ),
    "UNUSED_TEMPLATE_VARIABLES": (
        "The listed generated variables appear in no objective term or "
        "constraint, so they were left out of the problem and of its "
        "solutions; narrow the family's index sets if they were not meant "
        "to exist, or reference them if they were."
    ),
}


def _warning(code: str, path: str | None, message: str) -> SolveError:
    return SolveError(
        code=code,
        path=path,
        message=message,
        retryable=False,
        recommended_action=_WARNING_RECOMMENDED_ACTIONS[code],
    )


def _error(*, code: str, path: str | None, message: str) -> SolveError:
    """Build a validation error with the catalog's fixed recommended_action."""
    return catalog_error(code, message, path=path)


def int_text(value: int) -> str:
    """``str(value)``, or, when Python refuses to convert an integer that
    long (its 4300-digit limit), a description of its length instead."""
    try:
        return str(value)
    except ValueError:
        return f"<an integer of about {int(value.bit_length() * 0.30103) + 1} digits>"


def _bound_text(value: int | None) -> object:
    """A bound as a message shows it: exactly as before whenever it can be
    written out, described by its length otherwise."""
    return value if value is None else int_text(value)


def _check_variable_bounds(variable: Variable, path: str, errors: list[SolveError]) -> None:
    """3b §9.1: integer variables need legal bounds, binary ones take none."""
    # Shown through _bound_text: an integer too long for str() would make
    # the message itself raise (schema 1.3 spec §14.16 item 8).
    lower, upper = _bound_text(variable.lower_bound), _bound_text(variable.upper_bound)
    lower_value, upper_value = variable.lower_bound, variable.upper_bound
    if variable.type == "binary":
        if lower_value is not None or upper_value is not None:
            errors.append(
                _error(
                    code="BOUNDS_ON_BINARY",
                    path=path,
                    message=(
                        f"Binary variable {variable.name} declares bounds "
                        f"[{lower}, {upper}]; binary variables are always 0/1"
                    ),
                )
            )
        return

    if lower_value is None or upper_value is None:
        errors.append(
            _error(
                code="INTEGER_BOUNDS_MISSING",
                path=path,
                message=(
                    f"Integer variable {variable.name} needs both lower_bound and "
                    f"upper_bound, got lower_bound={lower}, upper_bound={upper}"
                ),
            )
        )
        return
    if upper_value <= lower_value:
        errors.append(
            _error(
                code="INTEGER_BOUNDS_INVALID",
                path=path,
                message=(
                    f"Integer variable {variable.name} has upper_bound {upper} "
                    f"<= lower_bound {lower}; a variable needs at least two values"
                ),
            )
        )
    if abs(lower_value) > INTEGER_BOUND_LIMIT or abs(upper_value) > INTEGER_BOUND_LIMIT:
        errors.append(
            _error(
                code="INTEGER_RANGE_TOO_LARGE",
                path=path,
                message=(
                    f"Integer variable {variable.name} has bounds [{lower}, {upper}] "
                    f"outside the supported range of ±{INTEGER_BOUND_LIMIT}"
                ),
            )
        )


def _duplicate_linear_variables(objective: Objective) -> list[tuple[str, int]]:
    counts = Counter(term.variable for term in objective.linear_terms)
    return [(variable, count) for variable, count in counts.items() if count > 1]


def _duplicate_quadratic_pairs(
    objective: Objective,
) -> list[tuple[tuple[str, ...], int]]:
    counts = Counter(
        frozenset((term.variable1, term.variable2))
        for term in objective.quadratic_terms
    )
    return [
        (tuple(sorted(pair)), count) for pair, count in counts.items() if count > 1
    ]

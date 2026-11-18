"""A shared ``ValidationCache`` changes how often work is done, never the output.

2026-11-18 review R-06: ``validate_expanded(..., cache=...)`` lets a caller
that validates one expansion against several backends (routing) do the
backend-independent work once. The contract pinned here, through public
behaviour only:

* A -- every result equals the one a separate call without a cache gives,
  for every registered backend and the model types it supports, in registry
  order and in reverse order;
* B -- the log lines are the ones separate calls produce, one per report;
* C -- one validation analyses each inequality once, and a cache shared by
  every backend still analyses each inequality once; the error pass runs
  once per cache and once per ``recommend``;
* D -- a cache is bound to the first expansion it validates;
* E -- results never share their ``warnings`` / ``errors`` lists;
* F -- every ``recommend`` entry is what validating that backend alone gives;
* G -- ``problem_analysis=analyze_problem(p)`` never changes an estimate.
"""

import json
import logging

import pytest

from annealbridge.compiler import BQMCompiler, CQMCompiler
from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import ExecutionPolicy, routing
from annealbridge.solvers import SolverRegistry
from annealbridge.validation import (
    ValidationCache,
    estimates,
    expand_problem,
    problem_validator,
    validate_expanded,
    validate_problem_full,
)
from annealbridge.validation.estimates import (
    analyze_problem,
    compute_soft_energy_total,
    estimate_encoded_interactions,
    estimate_model_variables,
    variable_bounds,
)
from tests.builders import hard, lin, quad, soft
from tests.conftest import EXAMPLES_DIR

LOGGER = "annealbridge.validation.problem_validator"
MAX_COMPILED_VARIABLES = 10
# Above simulated_annealing's declared seed_max (2**31 - 1), within the
# range of the backends that declare 0..2**32 - 1; ignored by the others.
LARGE_SEED = 3_000_000_000


# --------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------


def general_payload(**solver) -> dict:
    """Binary problem with soft and hard ``<=`` / ``>=`` / ``==`` constraints.

    Triggers REDUNDANT_CONSTRAINT (``loose``), SOFT_WEIGHT_SMALL (``tiny``),
    SOFT_ALWAYS_VIOLATED (``never``), LARGE_SLACK_RANGE (``wide``) and,
    through the repeated ``x0`` term, DUPLICATE_TERM_MERGED.
    """
    return {
        "name": "validation cache",
        "variables": [{"name": f"x{index}"} for index in range(6)],
        "objective": {
            "direction": "minimize",
            "linear_terms": [
                lin("x0", 10),
                lin("x1", 5),
                lin("x2", 3),
                lin("x0", 2),
                lin("x5", -1),
            ],
            "quadratic_terms": [quad("x0", "x1", 2)],
        },
        "constraints": [
            hard("cap", "<=", 2, [lin("x0", 1), lin("x1", 1), lin("x2", 1)]),
            hard("floor", ">=", 1, [lin("x3", 1), lin("x4", 1)]),
            hard("pick", "==", 1, [lin("x0", 1), lin("x5", 1)]),
            hard(
                "wide",
                "<=",
                4000,
                [lin("x0", 5000), lin("x1", 3000), lin("x2", 2000)],
            ),
            soft("loose", "<=", 5, [lin("x0", 1), lin("x1", 1)], 1),
            soft("tiny", ">=", 1, [lin("x2", 1), lin("x3", 1)], 0.01),
            soft("never", ">=", 3, [lin("x4", 1), lin("x5", 1)], 2),
            soft("one", "==", 1, [lin("x3", 1), lin("x4", 1)], 1),
        ],
        "solver": solver,
    }


def general_problem(**solver) -> OptimizationProblem:
    return OptimizationProblem.model_validate(general_payload(**solver))


def integer_problem() -> OptimizationProblem:
    """Version 1.1: two bounded integers next to a binary."""
    return OptimizationProblem.model_validate(
        {
            "version": "1.1",
            "name": "validation cache integers",
            "variables": [
                {"name": "k", "type": "integer", "lower_bound": 0, "upper_bound": 10},
                {"name": "m", "type": "integer", "lower_bound": 0, "upper_bound": 5},
                {"name": "x"},
            ],
            "objective": {
                "direction": "minimize",
                "linear_terms": [lin("k", 1), lin("m", 2), lin("x", 1), lin("k", 1)],
                "quadratic_terms": [quad("k", "x", 1)],
            },
            "constraints": [
                hard("cap", "<=", 12, [lin("k", 1), lin("m", 1)]),
                soft("floor", ">=", 2, [lin("k", 3), lin("m", -1)], 1),
                hard("pin", "==", 3, [lin("k", 1), lin("x", 1)]),
            ],
        }
    )


def integer_knapsack() -> OptimizationProblem:
    """The version 1.1 example shipped in ``examples/``."""
    payload = json.loads((EXAMPLES_DIR / "integer_knapsack.json").read_text(encoding="utf-8"))
    return OptimizationProblem.model_validate(payload)


def cardinality_problem() -> OptimizationProblem:
    """Version 1.2: hard and soft cardinality constraints beside a constraint."""
    names = [f"x{index}" for index in range(5)]
    return OptimizationProblem.model_validate(
        {
            "version": "1.2",
            "name": "validation cache cardinality",
            "variables": [{"name": name} for name in names],
            "objective": {
                "direction": "minimize",
                "linear_terms": [lin(name, index + 1) for index, name in enumerate(names)],
            },
            "constraints": [
                hard("budget", "<=", 4, [lin("x0", 2), lin("x3", 3)]),
            ],
            "cardinality_constraints": [
                {
                    "id": "at_most_one",
                    "type": "hard",
                    "variables": ["x0", "x1", "x2"],
                    "operator": "<=",
                    "rhs": 1,
                },
                {
                    "id": "at_least_two",
                    "type": "soft",
                    "variables": ["x2", "x3", "x4"],
                    "operator": ">=",
                    "rhs": 2,
                    "weight": 1,
                },
                {
                    "id": "exactly_two",
                    "type": "hard",
                    "variables": names,
                    "operator": "==",
                    "rhs": 2,
                },
                {
                    "id": "always",
                    "type": "soft",
                    "variables": ["x1", "x4"],
                    "operator": "<=",
                    "rhs": 2,
                    "weight": 1,
                },
            ],
        }
    )


ITEMS = ["a", "b", "c", "d"]


def template_document() -> dict:
    """A 1.3 document whose generated entries get aggregated warnings.

    ``pair`` generates four fractional hard equalities (FRACTIONAL_HARD_EQUALITY
    on the heuristic BQM path), ``slack`` four redundant soft inequalities,
    and the two linear term templates repeat every ``x[i]``
    (DUPLICATE_TERM_MERGED). ``pick`` is a 0/1 parameter selecting the items
    the ``cap`` template constrains.
    """
    return {
        "version": "1.3",
        "name": "validation cache templates",
        "index_sets": [{"name": "item", "elements": ITEMS}],
        "parameters": [
            {
                "name": "pick",
                "indices": ["item"],
                "values": [{"key": ["b"], "value": 0}],
                "default": 1,
            }
        ],
        "variable_families": [
            {"name": "x", "indices": ["item"]},
            {"name": "y", "indices": ["item"]},
        ],
        "variables": [{"name": "u"}, {"name": "v"}],
        "objective": {
            "direction": "maximize",
            "linear_terms": [lin("u", 1), lin("v", 1), lin("u", 1)],
            "linear_term_templates": [
                {"for_each": ["i in item"], "coefficient": 1, "variable": "x[i]"},
                {"for_each": ["i in item"], "coefficient": 2, "variable": "x[i]"},
            ],
        },
        "constraints": [
            hard("explicit_half", "==", 0.5, [lin("u", 0.5)]),
            hard("explicit_cap", "<=", 1, [lin("u", 1), lin("v", 1)]),
        ],
        "constraint_templates": [
            {
                "id": "pair",
                "type": "hard",
                "for_each": ["i in item"],
                "terms": [
                    {"coefficient": 0.5, "variable": "x[i]"},
                    {"coefficient": 0.5, "variable": "y[i]"},
                ],
                "operator": "==",
                "rhs": 0.5,
            },
            {
                "id": "slack",
                "type": "soft",
                "for_each": ["i in item"],
                "terms": [
                    {"coefficient": 1, "variable": "x[i]"},
                    {"coefficient": 1, "variable": "y[i]"},
                ],
                "operator": "<=",
                "rhs": 5,
                "weight": 1,
            },
            {
                "id": "cap",
                "type": "hard",
                "for_each": ["i in item"],
                "where": ["pick[i] == 1"],
                "terms": [
                    {"coefficient": 1, "variable": "x[i]"},
                    {"coefficient": 1, "variable": "y[i]"},
                    {"coefficient": 1, "variable": "u"},
                ],
                "operator": "<=",
                "rhs": 2,
            },
        ],
    }


def template_problem(**solver) -> OptimizationProblem:
    data = template_document()
    if solver:
        data["solver"] = solver
    return OptimizationProblem.model_validate(data)


AGGREGATED = "generated entries from this template get this warning"

PROBLEMS = {
    "general": general_problem,
    "integer": integer_problem,
    "integer_knapsack": integer_knapsack,
    "cardinality": cardinality_problem,
    "template": template_problem,
    "seed": lambda: general_problem(seed=LARGE_SEED),
    "template_seed": lambda: template_problem(seed=LARGE_SEED),
    "soft_overflow": lambda: soft_overflow_problem(),
}

MORE_ERRORS = "more errors from"


def soft_overflow_problem() -> OptimizationProblem:
    """Valid on the CQM path; on the BQM path the soft penalty bound overflows."""
    return OptimizationProblem.model_validate(
        {
            "version": "1.0",
            "name": "soft overflow",
            "variables": [{"name": "a", "type": "binary"}, {"name": "b", "type": "binary"}],
            "objective": {"direction": "minimize", "linear_terms": [lin("a", 1)]},
            "constraints": [soft("s", "==", 1e160, [lin("a", 1e160), lin("b", 1)], 1e10)],
            "solver": {},
        }
    )


def capped_errors_problem(**solver) -> OptimizationProblem:
    """A template whose 25 generated constraints are all trivially infeasible:
    twenty errors are listed and five counted (schema 1.3 spec §14.11)."""
    return OptimizationProblem.model_validate(
        {
            "version": "1.3",
            "name": "capped errors",
            "index_sets": [{"name": "n", "elements": list(range(25))}],
            "variable_families": [{"name": "y", "indices": ["n"]}],
            "variables": [{"name": "u"}],
            "objective": {
                "direction": "minimize",
                "linear_terms": [lin("u", 1)],
                "linear_term_templates": [
                    {"for_each": ["k in n"], "coefficient": 1, "variable": "y[k]"}
                ],
            },
            "constraints": [],
            "constraint_templates": [
                {
                    "id": "never",
                    "type": "hard",
                    "for_each": ["k in n"],
                    "terms": [{"coefficient": 1, "variable": "y[k]"}],
                    "operator": ">=",
                    "rhs": 2,
                }
            ],
            "solver": solver,
        }
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def compilers() -> dict:
    return {c.model_type: c for c in (BQMCompiler(), CQMCompiler())}


def registry_calls(order: str) -> list[dict]:
    """Keyword arguments of one validation per (backend, supported model type)."""
    registry = SolverRegistry.default()
    names = list(registry.names())
    if order == "reversed":
        names.reverse()
    calls = []
    for name in names:
        caps = registry.get(name).capabilities
        for model_type in caps.supported_model_types:
            calls.append(
                {
                    "capabilities": caps,
                    "max_compiled_variables": MAX_COMPILED_VARIABLES,
                    "model_type": model_type,
                }
            )
    return calls


def inequality_count(problem: OptimizationProblem) -> int:
    return sum(
        1 for constraint in problem.all_constraints() if constraint.operator in ("<=", ">=")
    )


def counted(monkeypatch, module, name: str) -> list[int]:
    """Count calls of ``module.name`` without changing what it returns."""
    real = getattr(module, name)
    calls: list[int] = []

    def wrapper(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return calls


def warning_messages(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER and record.levelno == logging.WARNING
    ]


# --------------------------------------------------------------------------
# The fixtures trigger what they are meant to
# --------------------------------------------------------------------------


def test_general_problem_triggers_the_constraint_advisories():
    result = validate_problem_full(general_problem(), model_type="bqm")
    assert result.valid, result.errors
    codes = {warning.code for warning in result.warnings}
    assert {
        "REDUNDANT_CONSTRAINT",
        "SOFT_WEIGHT_SMALL",
        "SOFT_ALWAYS_VIOLATED",
        "LARGE_SLACK_RANGE",
        "DUPLICATE_TERM_MERGED",
    } <= codes


def test_template_problem_aggregates_generated_warnings():
    expansion = expand_problem(template_problem())
    assert not expansion.errors
    result = validate_expanded(expansion, model_type="bqm")
    assert result.valid, result.errors
    aggregated = {warning.code for warning in result.warnings if AGGREGATED in warning.message}
    assert {"FRACTIONAL_HARD_EQUALITY", "REDUNDANT_CONSTRAINT", "DUPLICATE_TERM_MERGED"} <= (
        aggregated
    )


@pytest.mark.parametrize("key", ["seed", "template_seed"])
def test_seed_problem_is_an_error_for_some_backends_and_a_warning_for_others(key):
    expansion = expand_problem(PROBLEMS[key]())
    results = [validate_expanded(expansion, **call) for call in registry_calls("registry")]
    assert any(not result.valid for result in results)
    assert any(
        result.valid and "SEED_IGNORED" in {warning.code for warning in result.warnings}
        for result in results
    )


# --------------------------------------------------------------------------
# A: a shared cache gives the results of separate calls
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["registry", "reversed"])
@pytest.mark.parametrize("key", list(PROBLEMS))
def test_shared_cache_equals_separate_calls(key, order):
    expansion = expand_problem(PROBLEMS[key]())
    assert not expansion.errors
    cache = ValidationCache()

    # The error pass first, as routing does.
    assert validate_expanded(expansion, errors_only=True, cache=cache) == validate_expanded(
        expansion, errors_only=True
    )
    # The baseline validates an expansion of its own, so nothing the cached
    # calls might leave on ``expansion`` can reach it.
    baseline = expand_problem(PROBLEMS[key]())
    for call in registry_calls(order):
        shared = validate_expanded(expansion, cache=cache, **call)
        separate = validate_expanded(baseline, **call)
        assert shared == separate, call["capabilities"].name
    # Without capabilities, on both compiler paths.
    for model_type in ("bqm", "cqm"):
        shared = validate_expanded(expansion, model_type=model_type, cache=cache)
        assert shared == validate_expanded(baseline, model_type=model_type)


@pytest.mark.parametrize("order", ["registry", "reversed"])
@pytest.mark.parametrize("seed", [None, LARGE_SEED])
def test_shared_cache_keeps_the_error_cap_of_a_failing_error_pass(order, seed):
    """The error pass itself fails, twenty errors listed and the rest counted:
    every result built from the shared pass carries the same cap and count."""
    solver = {} if seed is None else {"seed": seed}
    expansion = expand_problem(capped_errors_problem(**solver))
    baseline = expand_problem(capped_errors_problem(**solver))
    assert not expansion.errors
    cache = ValidationCache()

    for _ in range(2):
        first = validate_expanded(expansion, errors_only=True, cache=cache)
        assert first == validate_expanded(baseline, errors_only=True)
    assert len(first.errors) == 20
    assert MORE_ERRORS in first.errors[-1].message
    for call in registry_calls(order):
        shared = validate_expanded(expansion, cache=cache, **call)
        assert not shared.valid
        assert shared == validate_expanded(baseline, **call), call["capabilities"].name
    assert validate_expanded(expansion, errors_only=True, cache=cache) == first


def test_soft_overflow_is_an_error_on_the_bqm_path_only():
    expansion = expand_problem(soft_overflow_problem())
    assert not validate_expanded(expansion, model_type="bqm").valid
    assert validate_expanded(expansion, model_type="cqm").valid


def test_fractional_equality_separates_exhaustive_from_heuristic_backends():
    """The premise that makes A tell the two BQM cache entries apart."""
    expansion = expand_problem(template_problem())
    seen = set()
    for call in registry_calls("registry"):
        if call["model_type"] != "bqm":
            continue
        codes = {warning.code for warning in validate_expanded(expansion, **call).warnings}
        exhaustive = call["capabilities"].exhaustive
        assert ("FRACTIONAL_HARD_EQUALITY" in codes) is not exhaustive
        seen.add(exhaustive)
    assert seen == {True, False}


@pytest.mark.parametrize("key", list(PROBLEMS))
def test_shared_cache_results_do_not_depend_on_the_call_order(key):
    expansion = expand_problem(PROBLEMS[key]())
    forward_cache = ValidationCache()
    backward_cache = ValidationCache()
    forward = [
        validate_expanded(expansion, cache=forward_cache, **call)
        for call in registry_calls("registry")
    ]
    backward = [
        validate_expanded(expansion, cache=backward_cache, **call)
        for call in registry_calls("reversed")
    ]
    assert forward == list(reversed(backward))


def test_shared_cache_without_a_prior_error_pass_equals_separate_calls():
    expansion = expand_problem(general_problem())
    cache = ValidationCache()
    for call in registry_calls("reversed"):
        assert validate_expanded(expansion, cache=cache, **call) == validate_expanded(
            expansion, **call
        )


# --------------------------------------------------------------------------
# B: log lines
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["general", "template"])
def test_shared_cache_logs_what_separate_calls_log(key, caplog):
    expansion = expand_problem(PROBLEMS[key]())
    calls = registry_calls("registry")

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        validate_expanded(expansion, errors_only=True)
        for call in calls:
            validate_expanded(expansion, **call)
    separate = warning_messages(caplog)

    caplog.clear()
    cache = ValidationCache()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        validate_expanded(expansion, errors_only=True, cache=cache)
        for call in calls:
            validate_expanded(expansion, cache=cache, **call)
    shared = warning_messages(caplog)

    assert separate  # the duplicate objective term is logged
    assert any("appears" in message for message in separate)
    assert shared == separate


# --------------------------------------------------------------------------
# C: how often the work is done
# --------------------------------------------------------------------------


def embedding_capabilities():
    registry = SolverRegistry.default()
    return next(
        registry.get(name).capabilities
        for name in registry.names()
        if registry.get(name).capabilities.requires_embedding
    )


@pytest.mark.parametrize("key", ["general", "integer", "cardinality"])
def test_one_full_validation_analyses_each_inequality_once(key, monkeypatch):
    problem = PROBLEMS[key]()
    expected = inequality_count(problem)
    assert expected > 0
    # The validator analyses through ``estimates`` only; a direct import
    # would be a call this count cannot see.
    assert not hasattr(problem_validator, "analyze_inequality")
    calls = counted(monkeypatch, estimates, "analyze_inequality")

    result = validate_problem_full(problem, capabilities=embedding_capabilities())

    assert result.valid, result.errors
    assert len(calls) == expected


@pytest.mark.parametrize("key", ["general", "integer", "cardinality", "template"])
def test_a_cache_shared_by_every_backend_analyses_each_inequality_once(key, monkeypatch):
    expansion = expand_problem(PROBLEMS[key]())
    expected = inequality_count(expansion.problem)
    assert expected > 0
    calls = counted(monkeypatch, estimates, "analyze_inequality")

    cache = ValidationCache()
    validate_expanded(expansion, errors_only=True, cache=cache)
    for call in registry_calls("registry"):
        validate_expanded(expansion, cache=cache, **call)

    assert len(calls) == expected


@pytest.mark.parametrize("key", ["general", "template"])
def test_a_shared_cache_runs_the_error_pass_once(key, monkeypatch):
    expansion = expand_problem(PROBLEMS[key]())
    calls = counted(monkeypatch, problem_validator, "_collect")

    cache = ValidationCache()
    validate_expanded(expansion, errors_only=True, cache=cache)
    for call in registry_calls("registry"):
        validate_expanded(expansion, cache=cache, **call)

    assert len(calls) == 1


def test_separate_calls_each_run_the_error_pass(monkeypatch):
    expansion = expand_problem(general_problem())
    calls = counted(monkeypatch, problem_validator, "_collect")
    every = registry_calls("registry")

    for call in every:
        validate_expanded(expansion, **call)

    assert len(calls) == len(every)


@pytest.mark.parametrize("key", ["general", "template", "seed"])
def test_recommend_runs_the_error_pass_once(key, monkeypatch):
    calls = counted(monkeypatch, problem_validator, "_collect")

    result = routing.recommend(
        PROBLEMS[key](), SolverRegistry.default(), ExecutionPolicy(), compilers()
    )

    assert result.valid
    assert len(calls) == 1


# --------------------------------------------------------------------------
# D: a cache belongs to one expansion
# --------------------------------------------------------------------------


def test_a_cache_refuses_a_second_expansion():
    cache = ValidationCache()
    first = expand_problem(general_problem())
    second = expand_problem(integer_problem())
    validate_expanded(first, cache=cache)

    with pytest.raises(ValueError):
        validate_expanded(second, cache=cache)
    with pytest.raises(ValueError):
        validate_expanded(second, errors_only=True, cache=cache)


def test_a_cache_refuses_another_expansion_of_an_equal_problem():
    cache = ValidationCache()
    validate_expanded(expand_problem(template_problem()), cache=cache)

    with pytest.raises(ValueError):
        validate_expanded(expand_problem(template_problem()), cache=cache)


def test_a_cache_serves_the_same_expansion_repeatedly():
    expansion = expand_problem(general_problem())
    cache = ValidationCache()
    first = validate_expanded(expansion, model_type="bqm", cache=cache)
    for _ in range(3):
        assert validate_expanded(expansion, model_type="bqm", cache=cache) == first
    # Refusing another expansion leaves the cache usable for its own.
    with pytest.raises(ValueError):
        validate_expanded(expand_problem(integer_problem()), cache=cache)
    assert validate_expanded(expansion, model_type="bqm", cache=cache) == first


# --------------------------------------------------------------------------
# E: results do not share their lists
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["general", "template"])
def test_appending_to_a_results_warnings_does_not_reach_the_next_result(key):
    expansion = expand_problem(PROBLEMS[key]())
    cache = ValidationCache()

    first = validate_expanded(expansion, model_type="bqm", cache=cache)
    assert first.valid and first.warnings
    before = list(first.warnings)
    # OptimizationService.validate appends to a result's warnings like this.
    first.warnings.append(first.warnings[0])

    for call in registry_calls("registry"):
        again = validate_expanded(expansion, cache=cache, **call)
        assert again == validate_expanded(expansion, **call)
    assert validate_expanded(expansion, model_type="bqm", cache=cache).warnings == before


@pytest.mark.parametrize("key", ["seed", "template_seed"])
def test_appending_to_a_results_errors_does_not_reach_the_next_result(key):
    expansion = expand_problem(PROBLEMS[key]())
    rejecting = [
        call
        for call in registry_calls("registry")
        if not validate_expanded(expansion, **call).valid
    ]
    assert rejecting
    call = rejecting[0]
    cache = ValidationCache()

    pass_result = validate_expanded(expansion, errors_only=True, cache=cache)
    assert pass_result.errors == []
    pass_result.errors.append(validate_expanded(expansion, **call).errors[0])

    first = validate_expanded(expansion, cache=cache, **call)
    before = list(first.errors)
    first.errors.append(first.errors[0])

    assert validate_expanded(expansion, cache=cache, **call).errors == before
    assert validate_expanded(expansion, errors_only=True, cache=cache).errors == []
    for other in registry_calls("registry"):
        assert validate_expanded(expansion, cache=cache, **other) == validate_expanded(
            expansion, **other
        )


# --------------------------------------------------------------------------
# F: each recommend entry equals validating that backend alone
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["general", "integer", "cardinality", "template", "seed"])
def test_recommend_entries_equal_separate_validation(key):
    problem = PROBLEMS[key]()
    registry = SolverRegistry.default()
    policy = ExecutionPolicy()

    result = routing.recommend(problem, registry, policy, compilers())

    assert result.valid
    assert {entry.backend for entry in result.recommendations} == set(registry.names())
    expansion = expand_problem(problem)
    for entry in result.recommendations:
        alone = validate_expanded(
            expansion,
            capabilities=registry.get(entry.backend).capabilities,
            max_compiled_variables=int(policy.required_limit("variables")),
            model_type=entry.model_type,
        )
        assert entry.warnings == alone.warnings, entry.backend
        assert entry.estimated_compiled_variables == alone.estimated_compiled_variables


# --------------------------------------------------------------------------
# G: ProblemAnalysis never changes an estimate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key", ["general", "integer", "integer_knapsack", "cardinality", "template"]
)
def test_problem_analysis_gives_the_same_estimates(key):
    expansion = expand_problem(PROBLEMS[key]())
    problem = expansion.problem
    analysis = analyze_problem(problem)
    bounds = variable_bounds(problem)

    for model_type in ("bqm", "cqm"):
        assert estimate_model_variables(
            problem, model_type, problem_analysis=analysis
        ) == estimate_model_variables(problem, model_type)
    assert estimate_encoded_interactions(
        problem, problem_analysis=analysis
    ) == estimate_encoded_interactions(problem)
    assert compute_soft_energy_total(
        problem, bounds, problem_analysis=analysis
    ) == compute_soft_energy_total(problem, bounds)
    assert compute_soft_energy_total(problem, problem_analysis=analysis) == (
        compute_soft_energy_total(problem, bounds)
    )

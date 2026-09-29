"""Expansion is deterministic, byte for byte (validation/expansion).

Schema 1.3 spec 2026-09-25 §14.7 ("every order comes from an array; the
parameter tables are only looked up, never iterated, so the result does not
depend on dict / set iteration order or the hash seed") and §14.16 item 3:

* the same document expanded twice gives the same ``model_dump_json`` and
  the same error and warning lists;
* expanded in fresh interpreters under ``PYTHONHASHSEED`` 0, 1 and 12345,
  the output is the same bytes every time -- for a document that expands
  with every expansion warning, one that fails statically with errors from
  many sources (past the per-source ceiling), one that fails while
  generating, and one over the ceiling.
"""

import json
import os
import subprocess
import sys

import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.validation import expand_problem
from tests.conftest import EXAMPLES_DIR

# Runs in a fresh interpreter: the request on stdin, one JSON document out.
_EXPAND = r"""
import json
import sys

from annealbridge.models import OptimizationProblem
from annealbridge.validation import expand_problem

request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
result = expand_problem(
    OptimizationProblem.model_validate(request["document"]),
    max_template_bindings=request["limit"],
)
out = {
    "problem": None if result.problem is None else result.problem.model_dump_json(),
    "errors": [error.model_dump_json() for error in result.errors],
    "warnings": [warning.model_dump_json() for warning in result.warnings],
    "limit_exceeded": result.limit_exceeded,
}
sys.stdout.buffer.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))
"""

SEEDS = ("0", "1", "12345")
LIMIT = 250_000
WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
]


def _document(**fields) -> dict:
    data = {
        "version": "1.3",
        "name": "determinism",
        "variables": [{"name": "v"}],
        "objective": {"direction": "minimize", "linear_terms": []},
        "constraints": [],
    }
    data.update(fields)
    return data


def _with_warnings() -> dict:
    """String-keyed tables, pruning, boundary skips, merged and empty templates."""
    rows = [
        {"key": [a, b], "value": (i * 7 + j * 3) % 11 - 5}
        for i, a in enumerate(WORDS)
        for j, b in enumerate(WORDS)
        if (i + j) % 3
    ]
    return _document(
        index_sets=[
            {"name": "w", "elements": WORDS},
            {"name": "t", "elements": list(range(6)), "order": "linear"},
            {"name": "r", "elements": [f"r{k}" for k in range(5)], "order": "cyclic"},
        ],
        parameters=[
            {"name": "d", "indices": ["w", "w"], "values": rows, "default": 0},
            {"name": "on", "indices": ["w"], "values": [{"key": [a], "value": len(a) % 2} for a in WORDS]},
        ],
        variable_families=[
            {"name": "x", "indices": ["w", "t"]},
            {"name": "y", "indices": ["r"]},
            {"name": "unused", "indices": ["w"]},
        ],
        objective={
            "direction": "minimize",
            "linear_terms": [{"variable": "v", "coefficient": 1}],
            "linear_term_templates": [
                {
                    "for_each": ["a in w", "p in t"],
                    "where": ["on[a] == 1"],
                    "coefficient": "d[a,a]",
                    "variable": "x[a,p+2]",
                },
                {"for_each": ["q in r"], "coefficient": 2, "variable": "y[q-1]"},
            ],
            "quadratic_term_templates": [
                {
                    "for_each": ["a in w", "b in w", "p in t"],
                    "where": ["a < b", "d[a,b] != 0"],
                    "coefficient": "d[a,b]",
                    "variable1": "x[a,p]",
                    "variable2": "x[b,p+1]",
                }
            ],
        },
        constraint_templates=[
            {
                "id": "ring",
                "type": "hard",
                "for_each": ["q in r"],
                "terms": [
                    {"coefficient": 1, "variable": "y[q]"},
                    {"coefficient": -1, "variable": "y[q+5]"},
                ],
                "operator": "==",
                "rhs": 0,
            },
            {
                "id": "never",
                "type": "soft",
                "weight": 2,
                "for_each": ["a in w"],
                "where": ["on[a] == 7"],
                "terms": [{"for_each": ["p in t"], "coefficient": 1, "variable": "x[a,p]"}],
                "operator": "<=",
                "rhs": 1,
            },
        ],
        cardinality_constraint_templates=[
            {
                "id": "once",
                "type": "hard",
                "for_each": ["a in w"],
                "variables": [{"for_each": ["p in t"], "variable": "x[a,p]"}, "v"],
                "operator": "<=",
                "rhs": 1,
            }
        ],
    )


def _with_static_errors() -> dict:
    """Errors from many sources, two of them past the per-source ceiling."""
    return _document(
        index_sets=[
            {"name": "w", "elements": WORDS},
            {"name": "bad", "elements": [f"no {word}" for word in WORDS] + WORDS * 2},
            {"name": "t", "elements": [0, 1, 2]},
        ],
        parameters=[
            {"name": "d", "indices": ["w", "w"], "values": [{"key": [a, a], "value": 1} for a in WORDS]},
            {"name": "dup", "indices": ["w"], "values": [{"key": ["alpha"], "value": 1}] * 30},
            {"name": "w", "indices": ["t"], "default": 1},
        ],
        variable_families=[
            {"name": "x", "indices": ["w", "w"]},
            {"name": "__hidden", "indices": ["w"]},
            {"name": "n", "indices": ["t"], "type": "integer", "lower_bound": 5, "upper_bound": 1},
        ],
        objective={
            "direction": "minimize",
            "linear_terms": [],
            "linear_term_templates": [
                {"for_each": ["a in w"], "where": ["a = a"], "coefficient": "1", "variable": "x[a,zz]"},
            ],
        },
        constraint_templates=[
            {
                "id": "c",
                "type": "hard",
                "weight": 1,
                "for_each": ["a in w", "a in w"],
                "terms": [{"coefficient": 1, "variable": "x[a,a]"}],
                "operator": "<=",
                "rhs": 1,
            },
            {
                "id": "c",
                "type": "soft",
                "terms": [{"for_each": ["p in t"], "coefficient": 1, "variable": "n[p+1]"}],
                "operator": ">=",
                "rhs": 0.5,
            },
        ],
    )


def _with_generation_errors() -> dict:
    """PARAMETER_VALUE_MISSING only arises while generating."""
    return _document(
        index_sets=[{"name": "w", "elements": WORDS}],
        parameters=[
            {"name": "d", "indices": ["w", "w"], "values": [{"key": [a, a], "value": 1} for a in WORDS[::3]]}
        ],
        variable_families=[{"name": "x", "indices": ["w", "w"]}],
        objective={
            "direction": "minimize",
            "linear_terms": [],
            "linear_term_templates": [
                {"for_each": ["a in w", "b in w"], "coefficient": "d[a,b]", "variable": "x[a,b]"}
            ],
        },
    )


def _over_the_ceiling() -> dict:
    return _document(
        index_sets=[{"name": "w", "elements": WORDS}],
        variable_families=[{"name": "x", "indices": ["w", "w", "w", "w", "w"]}],
    )


def _tsp_template() -> dict:
    return json.loads((EXAMPLES_DIR / "tsp_template.json").read_text(encoding="utf-8"))


CASES = {
    "warnings": _with_warnings,
    "static errors": _with_static_errors,
    "generation errors": _with_generation_errors,
    "ceiling": _over_the_ceiling,
    "tsp_template": _tsp_template,
}


def _serialized(data: dict) -> dict:
    result = expand_problem(OptimizationProblem.model_validate(data), max_template_bindings=LIMIT)
    return {
        "problem": None if result.problem is None else result.problem.model_dump_json(),
        "errors": [error.model_dump_json() for error in result.errors],
        "warnings": [warning.model_dump_json() for warning in result.warnings],
        "limit_exceeded": result.limit_exceeded,
    }


def _bytes(out: dict) -> bytes:
    return json.dumps(out, ensure_ascii=False).encode("utf-8")


def _codes(items: list[str]) -> set[str]:
    return {json.loads(item)["code"] for item in items}


def _environment(seed: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in ("PYTHONIOENCODING", "PYTHONUTF8", "PYTHONHASHSEED")
    }
    env["PYTHONHASHSEED"] = seed
    return env


def _in_a_fresh_interpreter(data: dict, seed: str) -> bytes:
    request = json.dumps({"document": data, "limit": LIMIT}, ensure_ascii=False)
    completed = subprocess.run(
        [sys.executable, "-c", _EXPAND],
        input=request.encode("utf-8"),
        capture_output=True,
        timeout=120,
        env=_environment(seed),
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    return completed.stdout


class TestTheCasesCoverWhatTheyClaim:
    """Each document exercises the part of the expander it is named after."""

    def test_warnings(self):
        out = _serialized(_with_warnings())
        assert out["errors"] == [] and out["problem"] is not None
        assert _codes(out["warnings"]) == {
            "UNUSED_TEMPLATE_VARIABLES",
            "TEMPLATE_BOUNDARY_SKIPPED",
            "TEMPLATE_TERMS_MERGED",
            "EMPTY_TEMPLATE_EXPANSION",
        }

    def test_static_errors(self):
        out = _serialized(_with_static_errors())
        assert out["problem"] is None and out["warnings"] == []
        assert _codes(out["errors"]) == {
            "INDEX_SET_INVALID",
            "PARAMETER_TABLE_INVALID",
            "DUPLICATE_TEMPLATE_NAME",
            "RESERVED_VARIABLE_NAME",
            "INTEGER_BOUNDS_INVALID",
            "TEMPLATE_REFERENCE_INVALID",
            "HARD_CONSTRAINT_HAS_WEIGHT",
            "SOFT_CONSTRAINT_MISSING_WEIGHT",
            "NON_INTEGER_INEQUALITY",
        }
        suffixed = [json.loads(e)["message"] for e in out["errors"] if "more error" in e]
        assert len(suffixed) == 2

    def test_generation_errors(self):
        out = _serialized(_with_generation_errors())
        assert _codes(out["errors"]) == {"PARAMETER_VALUE_MISSING"}
        assert len(out["errors"]) == 20

    def test_ceiling(self):
        out = _serialized(_over_the_ceiling())
        assert out["limit_exceeded"] is True
        assert _codes(out["errors"]) == {"TEMPLATE_EXPANSION_LIMIT"}


class TestSameProcess:
    @pytest.mark.parametrize("case", list(CASES))
    def test_twice_the_same_bytes(self, case):
        assert _bytes(_serialized(CASES[case]())) == _bytes(_serialized(CASES[case]()))

    def test_the_same_problem_object_twice(self):
        source = OptimizationProblem.model_validate(_with_warnings())
        first = expand_problem(source)
        second = expand_problem(source)
        assert first.problem is not second.problem
        assert first.problem.model_dump_json() == second.problem.model_dump_json()
        assert first.warnings == second.warnings


class TestAcrossHashSeeds:
    """Fresh interpreters under different PYTHONHASHSEED values (§14.16 item 3)."""

    @pytest.mark.parametrize("case", ["warnings", "static errors", "generation errors", "ceiling"])
    def test_the_same_bytes(self, case):
        data = CASES[case]()
        outputs = {seed: _in_a_fresh_interpreter(data, seed) for seed in SEEDS}
        assert len(set(outputs.values())) == 1, {seed: len(out) for seed, out in outputs.items()}
        # ... and the very bytes this process produces.
        assert outputs[SEEDS[0]] == _bytes(_serialized(data))

"""Architecture test: constraints are read through ``all_constraints()`` only.

Schema 1.2 spec 2026-09-25 §5.2. ``OptimizationProblem.cardinality_constraints``
is a second constraint list; a reader that iterates ``problem.constraints``
directly would silently skip every cardinality constraint (an estimate too
small, a constraint never re-validated). So the rule is deny-by-default:
*every* read of an attribute named ``constraints`` anywhere in ``src`` is a
violation unless its (file, innermost function) pair is on the allowlist
below, each entry with the reason it may look at the raw list.

A second rule keeps the BQM encoding decision out of solution checking:
the modules that judge and post-process candidates never refer to
``uses_pairwise_penalty``, so re-validation is independent of how a
constraint was encoded (overview principle 2).
"""

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "annealbridge"

# (file relative to SRC_ROOT, innermost enclosing function) -> reason.
ALLOWED_CONSTRAINT_READS: dict[tuple[str, str], str] = {
    # The single entry point itself: linear list, then the lowered
    # cardinality constraints.
    ("models/problem.py", "all_constraints"): "the one reader everything else uses",
    # The validator reports paths into what the user wrote, so it has to
    # tell the two lists apart.
    ("validation/problem_validator.py", "_collect"): (
        "structural checks of the linear constraints, with constraints[i] paths; "
        "cardinality ones are checked by _check_cardinality_constraint"
    ),
    ("validation/problem_validator.py", "_constraint_paths"): (
        "pairs all_constraints() order with constraints[i] / "
        "cardinality_constraints[i] paths for the advisories"
    ),
    ("validation/problem_validator.py", "_check_constraint_ids"): (
        "reads only the ids of both lists, never lowered()"
    ),
    ("validation/problem_validator.py", "_warn_cardinality_form"): (
        "advises on linear constraints only: a declared one needs no advice"
    ),
    # dimod's ConstrainedQuadraticModel.constraints, not the problem's.
    ("compiler/cqm.py", "_check_finite"): "dimod cqm.constraints",
    ("compiler/cqm.py", "compile"): "dimod cqm.constraints",
}

# Modules that judge or post-process candidates: they must not know how a
# constraint was encoded.
ENCODING_BLIND_MODULES = [
    "validation/solution_validator.py",
    "orchestration/candidates.py",
    "orchestration/postprocess.py",
]
ENCODING_DECISION = "uses_pairwise_penalty"


def constraint_reads(tree: ast.AST) -> list[tuple[str, int]]:
    """``(innermost function or "<module>", line)`` of every ``.constraints``."""
    found: list[tuple[str, int]] = []

    def visit(node: ast.AST, function: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        if isinstance(node, ast.Attribute) and node.attr == "constraints":
            found.append((function, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(tree, "<module>")
    return found


def references(tree: ast.AST, name: str) -> list[int]:
    """Lines that import, name or access ``name`` in any form."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(alias.name.split(".")[-1] == name for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.Name) and node.id == name:
            lines.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == name:
            lines.append(node.lineno)
        elif isinstance(node, ast.Constant) and node.value == name:
            lines.append(node.lineno)
    return lines


def violations_in(relative: str, source: str) -> list[str]:
    """``file:line in function()`` for each read of ``source`` not allowlisted."""
    return [
        f"{relative}:{line} in {function}()"
        for function, line in constraint_reads(ast.parse(source, filename=relative))
        if (relative, function) not in ALLOWED_CONSTRAINT_READS
    ]


def source_files() -> list[Path]:
    files = sorted(SRC_ROOT.rglob("*.py"))
    assert files, f"no source files under {SRC_ROOT}"
    return files


def test_every_constraints_read_is_allowlisted():
    violations = []
    seen: set[tuple[str, str]] = set()
    for path in source_files():
        relative = path.relative_to(SRC_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        violations.extend(violations_in(relative, source))
        seen.update((relative, function) for function, _ in constraint_reads(ast.parse(source)))
    assert violations == [], (
        "read constraints through OptimizationProblem.all_constraints() "
        f"(schema 1.2 spec §5.2): {violations}"
    )
    # A stale allowlist entry hides nothing today but would let a future
    # direct read through unnoticed: every entry must still be in use.
    assert set(ALLOWED_CONSTRAINT_READS) <= seen, set(ALLOWED_CONSTRAINT_READS) - seen


def test_dimod_reads_in_the_cqm_compiler_are_on_the_model():
    """The compiler/cqm.py entries only ever read ``cqm.constraints``."""
    path = SRC_ROOT / "compiler" / "cqm.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    receivers = {
        ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "constraints"
    }
    assert receivers == {"cqm"}


def test_solution_checking_ignores_the_encoding_decision():
    offenders = {}
    for relative in ENCODING_BLIND_MODULES:
        path = SRC_ROOT / relative
        assert path.is_file(), f"missing module: {relative}"
        lines = references(ast.parse(path.read_text(encoding="utf-8")), ENCODING_DECISION)
        if lines:
            offenders[relative] = lines
    assert offenders == {}


def test_the_encoding_decision_names_no_backend():
    """spec §13 item 12: the predicate is about the model, not a backend."""
    from annealbridge.solvers import SolverRegistry

    path = SRC_ROOT / "validation" / "estimates.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    (function,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == ENCODING_DECISION
    ]
    body = ast.unparse(function)
    for backend in SolverRegistry.default().names():
        assert backend not in body, backend


# --------------------------------------------------------------------------
# Self-tests: the scanners do catch what they are meant to catch.
# --------------------------------------------------------------------------

SYNTHETIC = '''
def reader(problem):
    return [c.id for c in problem.constraints]

class Holder:
    def method(self, problem):
        def inner():
            return len(problem.constraints)
        return inner()

TOP_LEVEL = PROBLEM.constraints
fine = problem.all_constraints()
'''


def test_scanner_finds_direct_reads_with_their_function():
    reads = constraint_reads(ast.parse(SYNTHETIC))
    assert sorted(reads) == [("<module>", 11), ("inner", 8), ("reader", 3)]


def test_a_new_reader_is_a_violation():
    assert violations_in("orchestration/new_module.py", SYNTHETIC) == [
        "orchestration/new_module.py:3 in reader()",
        "orchestration/new_module.py:8 in inner()",
        "orchestration/new_module.py:11 in <module>()",
    ]


def test_the_allowlist_is_per_function_not_per_file():
    """A new helper in an allowlisted file is still a violation."""
    helper = "def _new_helper(problem):\n    return problem.constraints\n"
    assert violations_in("validation/problem_validator.py", helper) == [
        "validation/problem_validator.py:2 in _new_helper()"
    ]
    reader = "def all_constraints(self):\n    return list(self.constraints)\n"
    assert violations_in("models/problem.py", reader) == []
    assert violations_in("models/other.py", reader) == ["models/other.py:2 in all_constraints()"]


def test_reference_scanner_sees_every_form():
    source = (
        "from annealbridge.validation.estimates import uses_pairwise_penalty\n"
        "import annealbridge.validation.estimates as estimates\n"
        "estimates.uses_pairwise_penalty(c)\n"
        "f = uses_pairwise_penalty\n"
        "getattr(estimates, 'uses_pairwise_penalty')\n"
    )
    assert sorted(references(ast.parse(source), ENCODING_DECISION)) == [1, 3, 4, 5]
    assert references(ast.parse("x = all_constraints()\n"), ENCODING_DECISION) == []

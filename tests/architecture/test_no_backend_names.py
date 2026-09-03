"""Architecture test: no backend-name string constants in the core (3a §13.2).

The service, validator and interface adapters dispatch on capability
*flags*, never on a backend's name. This test walks the AST of the files
where such a comparison would hide and fails on any string constant whose
whole value is one of the shipped backend names. Docstrings (the leading
``Expr(Constant)`` of a module, class or function) are exempt; comments
never reach the AST. Substrings are deliberately not matched: a message
that *mentions* a backend is fine, a value that *is* a backend name is a
dispatch waiting to happen.

Where the names are allowed to appear: ``models/problem.py`` (the
``backend`` Literal and option-block fields), ``models/error_catalog.py``,
each backend's own module, ``solvers/registry.py``, README and tests.
"""

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "annealbridge"

BACKEND_NAMES = frozenset(
    {
        "exact",
        "simulated_annealing",
        "dwave_qpu",
        "leap_hybrid_bqm",
        "leap_hybrid_cqm",
    }
)

SCANNED_PACKAGES = ["orchestration", "validation"]
SCANNED_FILES = [
    "interfaces/capabilities.py",
    "interfaces/mcp/tools.py",
    "interfaces/cli/main.py",
]


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """ids of the Constant nodes that are docstrings."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                found.add(id(body[0].value))
    return found


def backend_name_constants(
    source: str, filename: str = "<string>"
) -> list[tuple[int, str]]:
    """``(lineno, value)`` of every non-docstring string constant that IS a backend name."""
    tree = ast.parse(source, filename=filename)
    docstrings = _docstring_nodes(tree)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in BACKEND_NAMES
            and id(node) not in docstrings
        ):
            hits.append((node.lineno, node.value))
    return hits


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for package in SCANNED_PACKAGES:
        package_dir = SRC_ROOT / package
        assert package_dir.is_dir(), f"missing package directory: {package_dir}"
        files.extend(sorted(package_dir.rglob("*.py")))
    for relative in SCANNED_FILES:
        path = SRC_ROOT / relative
        assert path.is_file(), f"missing scanned file: {path}"
        files.append(path)
    return files


def test_core_files_contain_no_backend_name_constants() -> None:
    violations: list[str] = []
    scanned = _scanned_files()
    for py_file in scanned:
        relative = py_file.relative_to(SRC_ROOT.parent.parent)
        for lineno, value in backend_name_constants(
            py_file.read_text(encoding="utf-8"), str(py_file)
        ):
            violations.append(f"{relative}:{lineno} -> {value!r}")

    assert scanned, "no files scanned — check SRC_ROOT resolution"
    assert not violations, (
        "backend name string constants found in the core (dispatch on "
        "capabilities instead):\n" + "\n".join(violations)
    )


class TestScannerItself:
    """The detector must catch what §13.2 forbids and ignore what it allows."""

    def test_flags_a_name_comparison(self):
        source = 'def f(backend):\n    if backend.name == "exact":\n        pass\n'
        assert backend_name_constants(source) == [(2, "exact")]

    def test_flags_names_inside_containers(self):
        source = 'NAMES = ("dwave_qpu", "leap_hybrid_cqm")\n'
        assert backend_name_constants(source) == [
            (1, "dwave_qpu"),
            (1, "leap_hybrid_cqm"),
        ]

    def test_ignores_docstrings_and_comments(self):
        source = (
            '"""exact"""\n'
            "# simulated_annealing\n"
            "class C:\n"
            '    """dwave_qpu"""\n'
            "    def m(self):\n"
            '        """leap_hybrid_bqm"""\n'
            "        return 1\n"
        )
        assert backend_name_constants(source) == []

    def test_ignores_substrings_and_mentions(self):
        source = 'MSG = "the exact backend enumerates"\nX = "exact_max_variables"\n'
        assert backend_name_constants(source) == []

    def test_a_string_statement_that_is_not_first_is_still_flagged(self):
        # Only the *first* statement of a body is a docstring.
        source = 'def f():\n    x = 1\n    "exact"\n'
        assert backend_name_constants(source) == [(3, "exact")]

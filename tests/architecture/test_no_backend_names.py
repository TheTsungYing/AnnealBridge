"""Architecture test: no backend-name string constants in the core (3a §13.2).

The service, validator and interface adapters dispatch on capability
*flags*, never on a backend's name. This test walks the AST of every module
in ``orchestration/``, ``validation/``, ``interfaces/`` and ``config/`` plus
``interrupt.py``, and fails on any string constant whose whole value is one
of the backend names registered by ``SolverRegistry.default()`` (the list is
read from the registry, so a new backend is covered without editing this
file). Docstrings (the leading
``Expr(Constant)`` of a module, class or function) are exempt; comments
never reach the AST. Substrings are deliberately not matched: a message
that *mentions* a backend is fine, a value that *is* a backend name is a
dispatch waiting to happen.

Where the names are allowed to appear: ``models/problem.py`` (the
``backend`` Literal and option-block fields), ``models/error_catalog.py``,
each backend's own module, ``solvers/registry.py``, README and tests.

One narrow exception inside the scanned tree: the example problem JSON that
the MCP server hands to clients (``interfaces/mcp/prompts.py`` and
``interfaces/mcp/server.py``) spells ``{"backend": "exact"}``. Only in those
two files, and only for the value of a dict literal whose key is the string
constant ``"backend"``, is a name allowed; comparisons, tuples, lists, sets
and the values of any other key in those files are still flagged.
"""

import ast
from pathlib import Path

from annealbridge.solvers import SolverRegistry

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "annealbridge"

BACKEND_NAMES = frozenset(SolverRegistry.default().names())

SCANNED_PACKAGES = ["orchestration", "validation", "interfaces", "config"]
SCANNED_FILES = [
    # Batch 6 (J): the interrupt every core layer polls; a name there
    # would be a per-backend stop rule.
    "interrupt.py",
]

# Files (posix, relative to SRC_ROOT) whose example problem JSON may name a
# backend as the value of a ``"backend"`` dict key. These dicts are sample
# problems shown to MCP clients (prompts and server instructions), not
# dispatch logic: nothing in the package branches on them. The exemption
# covers that one dict-value position only; see ``_example_backend_values``.
EXAMPLE_BACKEND_VALUE_FILES = frozenset(
    {
        "interfaces/mcp/prompts.py",
        "interfaces/mcp/server.py",
    }
)


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


def _example_backend_values(tree: ast.AST) -> set[int]:
    """ids of the value nodes paired with a ``"backend"`` key in a dict literal."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "backend":
                    found.add(id(value))
    return found


def backend_name_constants(
    source: str,
    filename: str = "<string>",
    *,
    allow_example_backend_values: bool = False,
) -> list[tuple[int, str]]:
    """``(lineno, value)`` of every non-docstring string constant that IS a backend name.

    With ``allow_example_backend_values`` the value of a ``"backend"`` key in
    a dict literal is not reported (example problem JSON); only the files in
    ``EXAMPLE_BACKEND_VALUE_FILES`` are scanned that way.
    """
    tree = ast.parse(source, filename=filename)
    docstrings = _docstring_nodes(tree)
    allowed = _example_backend_values(tree) if allow_example_backend_values else set()
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in BACKEND_NAMES
            and id(node) not in docstrings
            and id(node) not in allowed
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


def test_backend_names_come_from_a_populated_registry() -> None:
    """An empty registry would make the scan below pass vacuously."""
    assert BACKEND_NAMES
    assert "exact" in BACKEND_NAMES


def test_example_exemption_files_exist() -> None:
    for relative in EXAMPLE_BACKEND_VALUE_FILES:
        assert (SRC_ROOT / relative).is_file(), f"stale exemption: {relative}"


def test_core_files_contain_no_backend_name_constants() -> None:
    violations: list[str] = []
    scanned = _scanned_files()
    for py_file in scanned:
        relative = py_file.relative_to(SRC_ROOT.parent.parent)
        relative_to_src = py_file.relative_to(SRC_ROOT).as_posix()
        for lineno, value in backend_name_constants(
            py_file.read_text(encoding="utf-8"),
            str(py_file),
            allow_example_backend_values=(
                relative_to_src in EXAMPLE_BACKEND_VALUE_FILES
            ),
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

    def test_example_backend_value_is_exempt_only_when_enabled(self):
        source = 'EXAMPLE = {"solver": {"backend": "exact"}}\n'
        assert (
            backend_name_constants(source, allow_example_backend_values=True) == []
        )
        assert backend_name_constants(source) == [(1, "exact")]

    def test_exemption_does_not_cover_other_positions(self):
        source = (
            "def f(name):\n"
            '    if name == "exact":\n'
            "        pass\n"
            'NAMES = ("exact",)\n'
            'OTHER = {"other": "exact"}\n'
            'KEYED = {"exact": 1}\n'
            'LISTED = ["exact"]\n'
            'BOTH = {"backend": "exact", "fallback": "tabu"}\n'
        )
        found = backend_name_constants(source, allow_example_backend_values=True)
        assert sorted(found) == [
            (2, "exact"),
            (4, "exact"),
            (5, "exact"),
            (6, "exact"),
            (7, "exact"),
            (8, "tabu"),
        ]

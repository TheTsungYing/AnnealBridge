"""Architecture test: import boundaries for core packages (spec §4).

Core packages (models, validation, compiler, penalty, solvers, orchestration)
must not import ``mcp``, ``dwave.cloud``, ``annealbridge.config`` or
``annealbridge.interfaces``.  ``dwave.system`` is deliberately NOT banned:
remote solver backends are allowed to lazy-import it (spec §4).

One documented exception: ``solvers/ocean.py`` may lazy-import
``dwave.cloud.config`` *inside a function* — spec §19 requires the active
Ocean config token to be resolvable for redaction, and that config loader
lives in ``dwave.cloud``.  A module-level import there is still a violation.
Since the 2026-09-09 review F-10 the exemption belongs to ``ocean.py`` and
not to ``solvers/metadata.py``: the shared metadata module knows no vendor
at all, so reading the Ocean config moved to the one D-Wave-aware module.

The detector (``_imported_modules``) sees every import form: ``import a.b``,
absolute and relative ``from X import n`` (reporting both ``X`` and ``X.n``,
so ``from annealbridge import config`` and ``from .. import config`` are
caught), and dynamic ``importlib.import_module`` / ``import_module`` /
``__import__`` calls with a string-constant module name.  Relative names are
resolved against the package of the scanned file.
"""

import ast
import sys
from pathlib import Path

import pytest

CORE_PACKAGES = [
    "models",
    "validation",
    "compiler",
    "penalty",
    "solvers",
    "orchestration",
]

BANNED_IMPORTS = [
    "mcp",
    "dwave.cloud",
    "annealbridge.config",
    "annealbridge.interfaces",
]

# (file relative to SRC_ROOT, module prefix): allowed only inside a function.
LAZY_IMPORT_EXEMPTIONS = {
    ("solvers/ocean.py", "dwave.cloud.config"),
}

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "annealbridge"


def _is_banned(module: str) -> bool:
    return any(
        module == banned or module.startswith(banned + ".")
        for banned in BANNED_IMPORTS
    )


def _is_exempt(relative_to_src: str, module: str, in_function: bool) -> bool:
    if not in_function:
        return False
    return any(
        relative_to_src == path
        and (module == prefix or module.startswith(prefix + "."))
        for path, prefix in LAZY_IMPORT_EXEMPTIONS
    )


def _package_of(py_file: Path) -> str:
    """Dotted package that relative imports in ``py_file`` resolve against.

    For ``pkg/__init__.py`` that is ``pkg`` itself (the directory it sits in);
    for any other module ``pkg/mod.py`` it is the enclosing package ``pkg``.
    Both cases are therefore the file's parent directory under ``SRC_ROOT``.
    """
    relative = py_file.resolve().relative_to(SRC_ROOT)
    return ".".join(["annealbridge", *relative.parent.parts])


def _resolve_module(module: str | None, level: int, package: str) -> str:
    """Absolute module name for ``module`` at relative ``level`` within ``package``."""
    if not level:
        return module or ""
    parts = package.split(".")
    base = parts[: len(parts) - level + 1]
    return ".".join(base + ([module] if module else []))


def _import_from_module(node: ast.ImportFrom, package: str) -> str:
    """Absolute module named by a ``from ... import`` statement."""
    return _resolve_module(node.module, node.level, package)


# Call forms treated as dynamic imports when the first positional argument is
# a string constant: ``importlib.import_module(...)``, a bare
# ``import_module(...)`` (after ``from importlib import import_module``) and
# ``__import__(...)``.
_DYNAMIC_IMPORT_FUNCTIONS = {"import_module", "__import__"}


def _dynamic_import_target(node: ast.Call) -> str | None:
    """The literal module name of a dynamic import call, or ``None``."""
    func = node.func
    if isinstance(func, ast.Attribute):
        if not (
            func.attr == "import_module"
            and isinstance(func.value, ast.Name)
            and func.value.id == "importlib"
        ):
            return None
    elif not (isinstance(func, ast.Name) and func.id in _DYNAMIC_IMPORT_FUNCTIONS):
        return None
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _imported_modules(tree: ast.AST, package: str):
    """Yield ``(lineno, module_name, in_function)`` for every import in ``tree``.

    * ``import a.b`` reports ``a.b``.
    * ``from X import n`` (absolute, or relative resolved against ``package``)
      reports ``X`` and ``X.n`` for every alias, because ``n`` may itself be
      a submodule (``from annealbridge import config``). ``import *`` adds no
      ``X.*`` entry.
    * ``importlib.import_module("m")``, ``import_module("m")`` and
      ``__import__("m")`` report ``m`` when the first positional argument is a
      string constant; a leading ``.`` is resolved against ``package``. Calls
      with any other first argument are ignored.

    ``in_function`` is True for statements and calls inside a function body.
    """
    function_scoped: set[ast.AST] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom, ast.Call)):
                    function_scoped.add(child)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, node in function_scoped
        elif isinstance(node, ast.ImportFrom):
            module = _import_from_module(node, package)
            in_function = node in function_scoped
            yield node.lineno, module, in_function
            for alias in node.names:
                if alias.name != "*":
                    yield node.lineno, f"{module}.{alias.name}", in_function
        elif isinstance(node, ast.Call):
            target = _dynamic_import_target(node)
            if target is None:
                continue
            stripped = target.lstrip(".")
            level = len(target) - len(stripped)
            module = _resolve_module(stripped or None, level, package)
            yield node.lineno, module, node in function_scoped


def test_import_detector_recognises_every_form() -> None:
    """Guard ``_imported_modules`` so the rules below cannot silently go blind."""
    source = (
        "import mcp.server\n"  # 1
        "import dwave.cloud as cloud\n"  # 2
        "from a.b import c\n"  # 3
        "from annealbridge import config\n"  # 4
        "from .. import config\n"  # 5
        "from ..config.settings import Settings\n"  # 6
        "from . import x\n"  # 7
        "def f():\n"  # 8
        "    import annealbridge.interfaces.cli\n"  # 9
        "    importlib.import_module('mcp.server')\n"  # 10
        "import_module('annealbridge.interfaces')\n"  # 11
        "__import__('dwave.cloud')\n"  # 12
        "import_module('.config', 'annealbridge.solvers')\n"  # 13
        "import_module(name)\n"  # 14
        "__import__(f'{prefix}.x')\n"  # 15
        "from os.path import *\n"  # 16
        "importlib.import_module()\n"  # 17
    )
    found = sorted(
        _imported_modules(ast.parse(source), "annealbridge.solvers.sub")
    )
    assert found == [
        (1, "mcp.server", False),
        (2, "dwave.cloud", False),
        (3, "a.b", False),
        (3, "a.b.c", False),
        (4, "annealbridge", False),
        (4, "annealbridge.config", False),
        (5, "annealbridge.solvers", False),
        (5, "annealbridge.solvers.config", False),
        (6, "annealbridge.solvers.config.settings", False),
        (6, "annealbridge.solvers.config.settings.Settings", False),
        (7, "annealbridge.solvers.sub", False),
        (7, "annealbridge.solvers.sub.x", False),
        (9, "annealbridge.interfaces.cli", True),
        (10, "mcp.server", True),
        (11, "annealbridge.interfaces", False),
        (12, "dwave.cloud", False),
        (13, "annealbridge.solvers.sub.config", False),
        (16, "os.path", False),
    ]


def test_import_detector_resolves_relative_imports_from_the_top_package() -> None:
    """``from .. import config`` in a first-level module reaches the real config."""
    source = "from .. import config\nimport_module('..interfaces')\n"
    found = sorted(_imported_modules(ast.parse(source), "annealbridge.solvers"))
    assert found == [
        (1, "annealbridge", False),
        (1, "annealbridge.config", False),
        (2, "annealbridge.interfaces", False),
    ]
    assert all(_is_banned(module) for _, module, _ in found[1:])


def test_package_of_follows_the_file_location() -> None:
    assert _package_of(SRC_ROOT / "solvers" / "ocean.py") == "annealbridge.solvers"
    assert _package_of(SRC_ROOT / "solvers" / "__init__.py") == "annealbridge.solvers"
    assert _package_of(SRC_ROOT / "interrupt.py") == "annealbridge"
    assert _package_of(SRC_ROOT / "__init__.py") == "annealbridge"
    assert (
        _package_of(SRC_ROOT / "interfaces" / "mcp" / "tools.py")
        == "annealbridge.interfaces.mcp"
    )


def test_core_packages_do_not_import_banned_modules() -> None:
    scanned_files = 0
    violations: list[str] = []

    for package in CORE_PACKAGES:
        package_dir = SRC_ROOT / package
        assert package_dir.is_dir(), f"missing core package directory: {package_dir}"
        for py_file in sorted(package_dir.rglob("*.py")):
            scanned_files += 1
            relative_to_src = py_file.relative_to(SRC_ROOT).as_posix()
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for lineno, module, in_function in _imported_modules(
                tree, _package_of(py_file)
            ):
                if _is_banned(module) and not _is_exempt(
                    relative_to_src, module, in_function
                ):
                    relative = py_file.relative_to(SRC_ROOT.parent.parent)
                    violations.append(f"{relative}:{lineno} -> {module}")

    assert scanned_files > 0, "no core .py files scanned — check SRC_ROOT resolution"
    assert not violations, "banned imports found in core packages:\n" + "\n".join(
        violations
    )


# Dependency direction (README "Layering"): models ← validation ← penalty ←
# compiler ← solvers ← orchestration.  penalty may import validation and models
# only; compiler may not import penalty (it gets ``compute_objective_scale``
# straight from ``validation.estimates``).
VALIDATION_UPPER_LAYERS = [
    "annealbridge.penalty",
    "annealbridge.compiler",
    "annealbridge.solvers",
    "annealbridge.orchestration",
    "annealbridge.config",
    "annealbridge.interfaces",
]


def test_validation_does_not_import_upper_layers() -> None:
    """§4/§20: validation depends on models only, so estimates can't drift."""
    scanned_files = 0
    violations: list[str] = []

    for py_file in sorted((SRC_ROOT / "validation").rglob("*.py")):
        scanned_files += 1
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for lineno, module, _in_function in _imported_modules(
            tree, _package_of(py_file)
        ):
            if any(
                module == banned or module.startswith(banned + ".")
                for banned in VALIDATION_UPPER_LAYERS
            ):
                relative = py_file.relative_to(SRC_ROOT.parent.parent)
                violations.append(f"{relative}:{lineno} -> {module}")

    assert scanned_files > 0, "no validation .py files scanned"
    assert not violations, "validation imports upper layers:\n" + "\n".join(violations)


# 2026-09-09 review (F-16): the README architecture diagram used to draw
# penalty *after* compiler, which contradicted both the layering line and the
# code.  penalty is a sibling consumer of ``validation.estimates`` that
# orchestration applies on the bqm path; the compiler reads
# ``compute_objective_scale`` from ``validation.estimates`` directly, so the
# direction is now pinned by a test instead of by a comment.
#
# Only ``annealbridge.penalty`` is listed.  ``solvers`` cannot be added:
# ``compiler/base.py``, ``bqm.py`` and ``cqm.py`` import
# ``annealbridge.solvers.base`` inside an ``if TYPE_CHECKING:`` block, and
# ``_imported_modules`` reports those as ordinary module-level imports (the
# block is not a function scope).  Banning solvers would need an exemption
# mechanism, which is not worth it for this rule.
COMPILER_FORBIDDEN = ("annealbridge.penalty",)


def test_compiler_does_not_import_penalty() -> None:
    """§4: compiler sits below penalty, so it may never import it."""
    scanned_files = 0
    violations: list[str] = []

    for py_file in sorted((SRC_ROOT / "compiler").rglob("*.py")):
        scanned_files += 1
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for lineno, module, _in_function in _imported_modules(
            tree, _package_of(py_file)
        ):
            if any(
                module == banned or module.startswith(banned + ".")
                for banned in COMPILER_FORBIDDEN
            ):
                relative = py_file.relative_to(SRC_ROOT.parent.parent)
                violations.append(f"{relative}:{lineno} -> {module}")

    assert scanned_files > 0, "no compiler .py files scanned"
    assert not violations, (
        "compiler imports penalty (take compute_objective_scale from "
        "annealbridge.validation.estimates instead):\n" + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Phase 3a §13.3: orchestration never names a concrete backend or, outside
# the one place that builds the default compiler list, a concrete compiler.
# ---------------------------------------------------------------------------

# Concrete backend modules. ``annealbridge.solvers`` itself, ``.base``,
# ``.registry`` and ``.metadata`` remain importable: those are the seams.
CONCRETE_BACKEND_MODULES = [
    "annealbridge.solvers.exact",
    "annealbridge.solvers.simulated_annealing",
    "annealbridge.solvers.tabu",
    "annealbridge.solvers.simulated_bifurcation",
    "annealbridge.solvers.dwave_qpu",
    "annealbridge.solvers.leap_hybrid_bqm",
    "annealbridge.solvers.leap_hybrid_cqm",
    "annealbridge.solvers.fujitsu_da",
]

CONCRETE_COMPILER_MODULES = ["annealbridge.compiler.bqm", "annealbridge.compiler.cqm"]
CONCRETE_COMPILER_NAMES = {"BQMCompiler", "CQMCompiler"}

# The only orchestration file allowed to know the concrete compilers: it
# assembles the default ``compilers`` list there and nowhere else (§4).
COMPILER_IMPORT_ALLOWED_IN = "orchestration/optimizer.py"


def _orchestration_files() -> list[Path]:
    files = sorted((SRC_ROOT / "orchestration").rglob("*.py"))
    assert files, "no orchestration .py files scanned"
    return files


def test_orchestration_does_not_import_concrete_backends() -> None:
    violations: list[str] = []
    for py_file in _orchestration_files():
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for lineno, module, _in_function in _imported_modules(
            tree, _package_of(py_file)
        ):
            if any(
                module == banned or module.startswith(banned + ".")
                for banned in CONCRETE_BACKEND_MODULES
            ):
                relative = py_file.relative_to(SRC_ROOT.parent.parent)
                violations.append(f"{relative}:{lineno} -> {module}")
    assert not violations, (
        "orchestration imports a concrete backend (go through the registry "
        "/ SolverBackend protocol instead):\n" + "\n".join(violations)
    )


def _concrete_compiler_imports(tree: ast.AST):
    """Yield ``(lineno, description)`` for every concrete-compiler import.

    Catches both the module form (``annealbridge.compiler.bqm``, possibly as
    a ``from`` target) and the re-exported names
    (``from annealbridge.compiler import BQMCompiler``).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in CONCRETE_COMPILER_MODULES:
                    yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module in CONCRETE_COMPILER_MODULES:
                yield node.lineno, node.module
            elif node.module == "annealbridge.compiler":
                for alias in node.names:
                    if alias.name in CONCRETE_COMPILER_NAMES:
                        yield node.lineno, f"annealbridge.compiler.{alias.name}"
            elif node.module.startswith("annealbridge.compiler."):
                for alias in node.names:
                    if alias.name in CONCRETE_COMPILER_NAMES:
                        yield node.lineno, f"{node.module}.{alias.name}"


def test_concrete_compilers_only_imported_by_optimizer() -> None:
    violations: list[str] = []
    for py_file in _orchestration_files():
        relative_to_src = py_file.relative_to(SRC_ROOT).as_posix()
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for lineno, description in _concrete_compiler_imports(tree):
            if relative_to_src != COMPILER_IMPORT_ALLOWED_IN:
                relative = py_file.relative_to(SRC_ROOT.parent.parent)
                violations.append(f"{relative}:{lineno} -> {description}")
    assert not violations, (
        f"concrete compilers may only be imported by {COMPILER_IMPORT_ALLOWED_IN} "
        "(everything else uses the ModelCompiler protocol):\n" + "\n".join(violations)
    )


# Candidate processing and result wording were split out of optimizer.py.
# They work on the original problem and the raw solver output only, so they
# must not know the compiler package at all — not even ``compiler.base``.
# Post-processing (batch 4 G) works on the decoded business variables only
# and must never reach slack or encoding bits, so the same rule holds.
COMPILER_FREE_ORCHESTRATION_FILES = [
    "orchestration/candidates.py",
    "orchestration/messages.py",
    "orchestration/postprocess.py",
]
COMPILER_PACKAGE = "annealbridge.compiler"


def _is_compiler_module(module: str) -> bool:
    return module == COMPILER_PACKAGE or module.startswith(COMPILER_PACKAGE + ".")


def _compiler_package_imports(tree: ast.AST, package: str):
    """Yield ``(lineno, module)`` for every import reaching ``annealbridge.compiler``.

    Covers ``import annealbridge.compiler[...]``, ``from annealbridge.compiler[...]
    import ...``, ``from annealbridge import compiler`` and the relative forms
    (resolved against ``package``), at module level or inside a function.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_compiler_module(alias.name):
                    yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = _import_from_module(node, package)
            if _is_compiler_module(module):
                yield node.lineno, module
            else:
                for alias in node.names:
                    if _is_compiler_module(f"{module}.{alias.name}"):
                        yield node.lineno, f"{module}.{alias.name}"


def test_candidates_and_messages_do_not_import_the_compiler() -> None:
    violations: list[str] = []
    for relative_to_src in COMPILER_FREE_ORCHESTRATION_FILES:
        py_file = SRC_ROOT / relative_to_src
        assert py_file.is_file(), f"{relative_to_src} is missing; rule would be vacuous"
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for lineno, module in _compiler_package_imports(
            tree, "annealbridge.orchestration"
        ):
            relative = py_file.relative_to(SRC_ROOT.parent.parent)
            violations.append(f"{relative}:{lineno} -> {module}")
    assert not violations, (
        "candidate processing and result messages must not import the compiler "
        "package:\n" + "\n".join(violations)
    )


def test_compiler_package_import_detector_recognises_every_form() -> None:
    """Guard the detector itself so the rule above cannot silently go blind."""
    source = (
        "import annealbridge.compiler\n"
        "import annealbridge.compiler.base as base\n"
        "from annealbridge.compiler.base import ModelCompiler\n"
        "from annealbridge import compiler\n"
        "from .. import compiler\n"
        "from ..compiler.bqm import BQMCompiler\n"
        "def f():\n"
        "    from annealbridge.compiler import CQMCompiler\n"
        "from annealbridge import models\n"
        "from annealbridge.compilers_elsewhere import x\n"
        "from . import messages\n"
    )
    found = sorted(
        _compiler_package_imports(ast.parse(source), "annealbridge.orchestration")
    )
    assert found == [
        (1, "annealbridge.compiler"),
        (2, "annealbridge.compiler.base"),
        (3, "annealbridge.compiler.base"),
        (4, "annealbridge.compiler"),
        (5, "annealbridge.compiler"),
        (6, "annealbridge.compiler.bqm"),
        (8, "annealbridge.compiler"),
    ]


# ---------------------------------------------------------------------------
# Phase 3b §4 / §26.2: the Fujitsu DA backend speaks HTTPS through the
# standard library, so no new HTTP dependency may enter the package.
# ---------------------------------------------------------------------------

BANNED_HTTP_PACKAGES = {"requests", "httpx", "aiohttp"}


def test_no_third_party_http_package_is_imported_anywhere() -> None:
    """No module of ``annealbridge`` imports a third-party HTTP package.

    Checked over the whole package (not just ``solvers/fujitsu_da.py``) so a
    later remote backend cannot quietly add the dependency either; both
    module-level and function-scoped imports count.
    """
    scanned_files = 0
    violations: list[str] = []

    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        scanned_files += 1
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for lineno, module, _in_function in _imported_modules(
            tree, _package_of(py_file)
        ):
            if module.split(".")[0] in BANNED_HTTP_PACKAGES:
                relative = py_file.relative_to(SRC_ROOT.parent.parent)
                violations.append(f"{relative}:{lineno} -> {module}")

    assert scanned_files > 0, "no .py files scanned — check SRC_ROOT resolution"
    assert (SRC_ROOT / "solvers" / "fujitsu_da.py").is_file(), (
        "the Fujitsu DA backend module is missing; this rule would pass vacuously"
    )
    assert not violations, (
        "third-party HTTP packages are banned (use urllib.request behind the "
        "HttpTransport seam):\n" + "\n".join(violations)
    )


def test_compiler_import_detector_recognises_every_form() -> None:
    """Guard the detector itself so the rule above cannot silently go blind."""
    source = (
        "import annealbridge.compiler.bqm\n"
        "from annealbridge.compiler.cqm import CQMCompiler\n"
        "from annealbridge.compiler import BQMCompiler, ModelCompiler\n"
        "from annealbridge.compiler.base import ModelCompiler\n"
    )
    found = list(_concrete_compiler_imports(ast.parse(source)))
    assert found == [
        (1, "annealbridge.compiler.bqm"),
        (2, "annealbridge.compiler.cqm"),
        (3, "annealbridge.compiler.BQMCompiler"),
    ]


# ---------------------------------------------------------------------------
# Batch 6 (J), time-limit spec 2026-09-24 §6.1: ``annealbridge/interrupt.py``
# sits beside ``exceptions.py`` at the top of the package, below every core
# layer, so solvers and orchestration can both import it. That only holds
# while it imports nothing but the standard library -- no annealbridge
# module (not even a sibling), no third-party package.
# ---------------------------------------------------------------------------

STDLIB_ONLY_MODULES = ["interrupt.py"]


def _stdlib_violations(tree: ast.AST) -> list[tuple[int, str]]:
    """``(lineno, module)`` of every import that is not the standard library.

    A relative import is a violation too: it can only reach annealbridge.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in sys.stdlib_module_names:
                    found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                found.append((node.lineno, "." * node.level + module))
            elif module.split(".")[0] not in sys.stdlib_module_names:
                found.append((node.lineno, module))
    return found


@pytest.mark.parametrize("relative_to_src", STDLIB_ONLY_MODULES)
def test_stdlib_only_module_imports_only_the_standard_library(relative_to_src) -> None:
    py_file = SRC_ROOT / relative_to_src
    assert py_file.is_file(), f"{relative_to_src} is missing; rule would be vacuous"
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    violations = [
        f"{relative_to_src}:{lineno} -> {module}"
        for lineno, module in _stdlib_violations(tree)
    ]
    assert not violations, (
        "this module must import the standard library only, so every core "
        "layer can depend on it:\n" + "\n".join(violations)
    )


def test_stdlib_detector_recognises_every_form() -> None:
    """Guard the detector itself so the rule above cannot silently go blind."""
    source = (
        "import threading\n"
        "import time as clock\n"
        "from collections.abc import Callable\n"
        "import numpy\n"
        "from annealbridge.models import SolveError\n"
        "from . import exceptions\n"
        "from .exceptions import SolveCancelled\n"
        "def f():\n"
        "    import dimod\n"
        "    from os import path\n"
    )
    assert sorted(_stdlib_violations(ast.parse(source))) == [
        (4, "numpy"),
        (5, "annealbridge.models"),
        (6, "."),
        (7, ".exceptions"),
        (9, "dimod"),
    ]

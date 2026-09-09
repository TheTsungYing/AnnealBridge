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
"""

import ast
from pathlib import Path

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


def _imported_modules(tree: ast.AST):
    """Yield ``(lineno, module_name, in_function)`` for every absolute import."""
    function_scoped: set[ast.AST] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    function_scoped.add(child)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, node in function_scoped
        elif isinstance(node, ast.ImportFrom):
            # Relative imports (level > 0) resolve inside the package itself.
            if node.level == 0 and node.module is not None:
                yield node.lineno, node.module, node in function_scoped


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
            for lineno, module, in_function in _imported_modules(tree):
                if _is_banned(module) and not _is_exempt(
                    relative_to_src, module, in_function
                ):
                    relative = py_file.relative_to(SRC_ROOT.parent.parent)
                    violations.append(f"{relative}:{lineno} -> {module}")

    assert scanned_files > 0, "no core .py files scanned — check SRC_ROOT resolution"
    assert not violations, "banned imports found in core packages:\n" + "\n".join(
        violations
    )


# Layers above validation in the §4 dependency direction (penalty sits between
# validation and compiler: it may import validation, never the reverse).
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
        for lineno, module, _in_function in _imported_modules(tree):
            if any(
                module == banned or module.startswith(banned + ".")
                for banned in VALIDATION_UPPER_LAYERS
            ):
                relative = py_file.relative_to(SRC_ROOT.parent.parent)
                violations.append(f"{relative}:{lineno} -> {module}")

    assert scanned_files > 0, "no validation .py files scanned"
    assert not violations, "validation imports upper layers:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# Phase 3a §13.3: orchestration never names a concrete backend or, outside
# the one place that builds the default compiler list, a concrete compiler.
# ---------------------------------------------------------------------------

# Concrete backend modules. ``annealbridge.solvers`` itself, ``.base``,
# ``.registry`` and ``.metadata`` remain importable: those are the seams.
CONCRETE_BACKEND_MODULES = [
    "annealbridge.solvers.exact",
    "annealbridge.solvers.simulated_annealing",
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
        for lineno, module, _in_function in _imported_modules(tree):
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
        for lineno, module, _in_function in _imported_modules(tree):
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

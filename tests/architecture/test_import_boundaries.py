"""Architecture test: import boundaries for core packages (spec §4).

Core packages (models, validation, compiler, penalty, solvers, orchestration)
must not import ``mcp``, ``dwave.cloud``, ``annealbridge.config`` or
``annealbridge.interfaces``.  ``dwave.system`` is deliberately NOT banned:
remote solver backends are allowed to lazy-import it (spec §4).

One documented exception: ``solvers/metadata.py`` may lazy-import
``dwave.cloud.config`` *inside a function* — spec §19 requires ``redact()``
to resolve the token from the active Ocean config, and that config loader
lives in ``dwave.cloud``.  A module-level import there is still a violation.
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
    ("solvers/metadata.py", "dwave.cloud.config"),
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

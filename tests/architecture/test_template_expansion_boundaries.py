"""Architecture test: the template expander's dependencies (schema 1.3 spec §14.12).

``validation/expansion.py`` and ``validation/template_grammar.py`` turn a
schema 1.3 document into the ordinary problem everything else reads, so
they must not depend on anything that reads a problem downstream of them:
the expander may import the models (and, through them, the exception
module), the validator's shared building blocks in ``validation.issues``,
the grammar module and the standard library -- nothing else of the package
and no third-party module. The grammar module is stricter still: the
standard library only, so it can never reach for anything that evaluates.
"""

import ast
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "annealbridge"

ALLOWED = {
    "validation/expansion.py": (
        "annealbridge.models",
        "annealbridge.validation.issues",
        "annealbridge.validation.template_grammar",
    ),
    "validation/template_grammar.py": (),
}


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}:{node.lineno} uses a relative import"
            yield node.lineno, node.module


@pytest.mark.parametrize("relative", sorted(ALLOWED))
def test_imports_stay_within_the_allowed_modules(relative: str) -> None:
    allowed = ALLOWED[relative]
    violations = []
    for lineno, module in _imports(SRC_ROOT / relative):
        top = module.split(".")[0]
        if top in sys.stdlib_module_names:
            continue
        if any(module == prefix or module.startswith(prefix + ".") for prefix in allowed):
            continue
        violations.append(f"{relative}:{lineno} -> {module}")
    assert not violations, "disallowed imports:\n" + "\n".join(violations)


def test_the_grammar_never_evaluates() -> None:
    """No eval / exec / compile / __import__ anywhere in the two modules.

    Called by name, or as an attribute (``builtins.eval``); ``re.compile``
    is the one attribute call named ``compile`` allowed, since it only
    builds a regular expression. Neither module may name ``builtins`` or
    reach for these through ``getattr`` either.
    """
    for relative in ALLOWED:
        source = (SRC_ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                assert func.id not in {"eval", "exec", "compile", "__import__", "getattr"}, (
                    f"{relative}:{node.lineno} calls {func.id}"
                )
            elif isinstance(func, ast.Attribute):
                owner = func.value.id if isinstance(func.value, ast.Name) else None
                assert func.attr not in {"eval", "exec", "__import__"}, (
                    f"{relative}:{node.lineno} calls .{func.attr}"
                )
                if func.attr == "compile":
                    assert owner == "re", f"{relative}:{node.lineno} calls .compile"
        assert "builtins" not in source, relative

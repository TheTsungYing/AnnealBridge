"""Suite policy: the default suite neither skips nor xfails (2026-11-18 review R-24).

Two things keep that rule from depending on anyone's attention:

* the hooks at the bottom of ``tests/conftest.py`` fail a session in which a
  test without the ``remote`` marker was skipped or xfailed;
* that exception is only sound while every live test really carries the
  marker, so each ``tests/remote_live/test_*.py`` must set
  ``pytestmark = pytest.mark.remote`` at module level.

The hooks are exercised for real: each case below runs pytest in a child
process on a throwaway test file, with the repository's ``tests/conftest.py``
loaded as a plugin, and checks the exit status.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_LIVE_DIR = REPO_ROOT / "tests" / "remote_live"


def _sets_remote_pytestmark(tree: ast.Module) -> bool:
    """True when the module body holds ``pytestmark = pytest.mark.remote``."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        ):
            continue
        if ast.unparse(node.value) == "pytest.mark.remote":
            return True
    return False


def test_every_live_test_module_is_marked_remote() -> None:
    files = sorted(REMOTE_LIVE_DIR.glob("test_*.py"))
    assert files, f"no live test modules found in {REMOTE_LIVE_DIR}"
    unmarked = [
        path.name
        for path in files
        if not _sets_remote_pytestmark(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        )
    ]
    assert not unmarked, (
        "live test modules without `pytestmark = pytest.mark.remote` would run "
        "(and skip) in the default suite:\n" + "\n".join(unmarked)
    )


def test_the_pytestmark_detector_recognises_the_marker() -> None:
    """Guard the detector itself so the rule above cannot silently go blind."""
    marked = "import pytest\n\npytestmark = pytest.mark.remote\n"
    assert _sets_remote_pytestmark(ast.parse(marked))
    for source in (
        "import pytest\n",
        "import pytest\npytestmark = pytest.mark.slow\n",
        "import pytest\npytestmark = [pytest.mark.remote]\n",
        "import pytest\ndef f():\n    pytestmark = pytest.mark.remote\n",
    ):
        assert not _sets_remote_pytestmark(ast.parse(source)), source


_INI = "[pytest]\nmarkers =\n    remote: live remote tests\n"

# case name -> (source of the throwaway test file, expected pytest exit status)
_CASES = {
    "plain_skip": (
        "import pytest\n\n"
        "def test_ok():\n    pass\n\n"
        "def test_skipped():\n    pytest.skip('not today')\n",
        1,
    ),
    "skip_marker": (
        "import pytest\n\n"
        "@pytest.mark.skip(reason='not today')\n"
        "def test_skipped():\n    pass\n",
        1,
    ),
    "xfail": (
        "import pytest\n\n"
        "@pytest.mark.xfail(reason='known bug')\n"
        "def test_broken():\n    assert False\n",
        1,
    ),
    "module_level_skip": (
        "import pytest\n\n"
        "pytest.skip('whole module', allow_module_level=True)\n\n"
        "def test_never_collected():\n    pass\n",
        1,
    ),
    "remote_skip": (
        "import pytest\n\n"
        "pytestmark = pytest.mark.remote\n\n"
        "def test_live():\n    pytest.skip('no credential')\n",
        0,
    ),
    "all_passing": ("def test_ok():\n    pass\n", 0),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_forbidden_skip_or_xfail_fails_the_session(case, tmp_path) -> None:
    source, expected_exit = _CASES[case]
    (tmp_path / "pytest.ini").write_text(_INI, encoding="utf-8")
    test_file = tmp_path / "test_case.py"
    test_file.write_text(source, encoding="utf-8")

    # ``tests.conftest`` is importable because the child runs from the
    # repository root; ``-c`` keeps the repository's own addopts out of it.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.conftest",
            "-p",
            "no:cacheprovider",
            "-c",
            str(tmp_path / "pytest.ini"),
            str(test_file),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == expected_exit, output
    if expected_exit:
        assert "forbidden skip / xfail" in output, output
        assert "test_case.py" in output, output
    else:
        assert "forbidden skip / xfail" not in output, output

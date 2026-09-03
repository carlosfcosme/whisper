import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_check_no_weights_passes_on_this_tree():
    result = _run("check_no_weights.py")
    assert result.returncode == 0, result.stderr


def test_check_no_all_interfaces_passes_on_this_tree():
    result = _run("check_no_all_interfaces.py")
    assert result.returncode == 0, result.stderr

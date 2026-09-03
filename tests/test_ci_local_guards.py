"""Torch-free CI guard scripts: local fixtures, bind, gitignore."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / ".github" / "scripts"
GUARD_SCRIPTS = (
    "fail-remote-fixture-urls.py",
    "fail-bind-wildcard.py",
    "verify-gitignore.py",
)


def _run(name: str) -> str:
    script = SCRIPTS / name
    assert script.is_file(), script
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed.stdout


def test_ci_local_guard_scripts_pass():
    outputs = [_run(name) for name in GUARD_SCRIPTS]
    assert all("ok:" in stdout for stdout in outputs)

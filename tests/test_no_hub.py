"""CI guard: no Hub in tests, no committed weight files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.localhost_only

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD = REPO_ROOT / "scripts" / "ci_fail_hub_or_weights.py"


def test_workflow_runs_offline_guard():
    text = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text()
    assert "offline-guard:" in text
    assert "ci_fail_hub_or_weights.py" in text


def test_ci_guard_script_passes():
    proc = subprocess.run(
        [sys.executable, str(GUARD)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK:" in proc.stdout

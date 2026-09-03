"""Run the offline suite inside an isolated venv (no model/network fetch)."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_offline_venv_tests.sh"

OFFLINE_MODULES = (
    "tests/test_offline_fetch.py",
    "tests/test_loopback_bind.py",
    "tests/test_gitignore_weights.py",
)


def test_offline_suite_passes_inside_venv(tmp_path):
    env = os.environ.copy()
    env["OFFLINE_VENV_DIR"] = str(tmp_path / "offline-venv")
    env["OFFLINE_PYTHON"] = sys.executable
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            "offline venv tests failed:\n{0}\n{1}".format(result.stdout, result.stderr)
        )


def test_offline_venv_script_targets_required_modules():
    text = SCRIPT.read_text()
    for module in OFFLINE_MODULES:
        assert module in text, module
    assert "127.0.0.1:9" in text
    assert "HF_HUB_OFFLINE" in text
    assert "--no-index" in text
    assert "--no-deps" in text
    assert "--no-build-isolation" in text

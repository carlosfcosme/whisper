"""CI guard: start/serve must not bind all interfaces. No whisper import."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _checker():
    path = ROOT / "scripts" / "check_no_wildcard_bind.py"
    spec = importlib.util.spec_from_file_location("check_no_wildcard_bind", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


offline_check = _checker()

pytestmark = pytest.mark.localhost_only


def test_current_start_paths_are_loopback():
    assert offline_check.reasons_wildcard_bind() == []


def test_check_fails_when_start_script_has_wildcard():
    reasons = offline_check.reasons_wildcard_bind(
        {
            ".cursor/start.sh": "python3 -m whisper.serve --host 0.0.0.0 --port 80\n",
        }
    )
    assert reasons
    assert any("0.0.0.0" in r for r in reasons)


def test_check_fails_when_start_omits_loopback():
    reasons = offline_check.reasons_wildcard_bind(
        {".cursor/start.sh": "python3 -m whisper.serve --port 8765\n"}
    )
    assert any("127.0.0.1" in r for r in reasons)


def test_standalone_script_passes_on_this_tree():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_no_wildcard_bind.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

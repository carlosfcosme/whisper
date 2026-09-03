"""CI guard: committed checkpoints fail the check. No whisper import."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _checker():
    path = ROOT / "scripts" / "check_no_weights.py"
    spec = importlib.util.spec_from_file_location("check_no_weights", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


weight_check = _checker()

pytestmark = pytest.mark.localhost_only


def test_current_tree_has_no_committed_weights():
    assert weight_check.reasons_committed_weights() == []


def test_check_fails_on_tracked_pt():
    reasons = weight_check.reasons_committed_weights([("tiny.pt", 100)])
    assert any("tiny.pt" in r for r in reasons)


def test_check_fails_on_oversized_file():
    reasons = weight_check.reasons_committed_weights(
        [("notes/big.bin", weight_check.MAX_BYTES + 1)]
    )
    assert any("10 MiB" in r for r in reasons)


def test_small_fixture_is_allowed():
    jfk = ROOT / "tests" / "jfk.flac"
    assert jfk.is_file()
    assert jfk.stat().st_size <= weight_check.MAX_BYTES
    reasons = weight_check.reasons_committed_weights(
        [("tests/jfk.flac", jfk.stat().st_size)]
    )
    assert reasons == []


def test_standalone_script_passes_on_this_tree():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_no_weights.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

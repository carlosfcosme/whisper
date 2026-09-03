"""CI fails if checkpoints are committed."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _checker():
    path = ROOT / "scripts" / "check_no_weights.py"
    spec = importlib.util.spec_from_file_location("check_no_weights", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


weights_check = _checker()


def test_current_tree_has_no_committed_weights():
    assert weights_check.reasons_committed_weights() == []


def test_planted_pt_is_detected():
    reasons = weights_check.reasons_committed_weights(
        [("leaked.pt", 12), ("tests/jfk.flac", 1000)]
    )
    assert reasons == ["tracked weight file: leaked.pt"]


def test_oversized_file_is_detected():
    reasons = weights_check.reasons_committed_weights(
        [("blob.bin", weights_check.MAX_BYTES + 1)]
    )
    assert any("exceeds 10 MiB" in r for r in reasons)


def test_script_passes_on_this_tree():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_no_weights.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK:" in result.stdout

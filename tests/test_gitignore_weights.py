"""Weight/cache artifacts must be gitignored and untracked."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _checker():
    path = ROOT / "scripts" / "check_gitignore_weights.py"
    spec = importlib.util.spec_from_file_location("check_gitignore_weights", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gitignore_check = _checker()


def test_required_patterns_present():
    assert gitignore_check.reasons_gitignore_missing() == []


def test_missing_pattern_is_detected():
    reasons = gitignore_check.reasons_gitignore_missing("__pycache__/\n*.py[cod]\n")
    assert any("*.pt" in r for r in reasons)
    assert any("cache/" in r for r in reasons)
    assert any("weights/" in r for r in reasons)
    assert any("*.bin" in r for r in reasons)


def test_probes_are_ignored():
    assert gitignore_check.reasons_probes_not_ignored() == []


def test_no_tracked_weight_paths():
    assert gitignore_check.reasons_tracked_weight_paths() == []


def test_script_passes_on_this_tree():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_gitignore_weights.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK:" in result.stdout


def test_git_check_ignore_cache_and_pt():
    for probe in (
        "cache/tiny.pt",
        "weights/base.pt",
        ".cache/whisper/tiny.pt",
        "pytorch_model.bin",
        "model.onnx",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", probe],
            cwd=str(ROOT),
            check=False,
        )
        assert result.returncode == 0, probe


def test_ci_cache_must_not_store_weights():
    assert gitignore_check.reasons_ci_caches_weights() == []
    poisoned = """
      - uses: actions/cache@v4
        with:
          path: |
            ~/.cache/whisper
            weights/
            *.pt
"""
    reasons = gitignore_check.reasons_ci_caches_weights(poisoned)
    assert reasons, "checker must fail when Actions cache stores weights"

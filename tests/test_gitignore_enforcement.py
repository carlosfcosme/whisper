"""Fail if .gitignore does not keep weights and caches untracked."""

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    path = ROOT / "scripts" / "check_gitignore_caches.py"
    spec = importlib.util.spec_from_file_location("check_gitignore_caches", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gitignore_enforcement_script_passes():
    checker = _load_checker()
    assert checker.find_violations(ROOT) == []
    assert checker.main() == 0


def test_required_cache_and_weight_patterns_are_present():
    checker = _load_checker()
    text = (ROOT / ".gitignore").read_text()
    for pattern in checker.REQUIRED_PATTERNS:
        assert pattern in text


def test_git_refuses_to_add_weight_cache_paths():
    result = subprocess.run(
        ["git", "add", "-n", ".cache/whisper/tiny.pt", "tiny.pt"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    assert "ignored" in combined.lower() or result.returncode != 0

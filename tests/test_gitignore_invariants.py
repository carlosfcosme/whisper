"""Gitignore must keep weight and cache paths untracked."""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "check_gitignore_invariants.py"
    spec = importlib.util.spec_from_file_location("check_gitignore_invariants", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gitignore_declares_weight_and_cache_patterns():
    check = _load_script()
    assert check.missing_patterns(REPO_ROOT) == []


def test_weight_and_cache_paths_are_untracked():
    check = _load_script()
    assert check.tracked_weight_paths(REPO_ROOT) == []


def test_gitignore_matches_in_repo_weight_examples():
    check = _load_script()
    assert check.unignored_examples(REPO_ROOT) == []


def test_tracked_assets_are_not_gitignored():
    check = _load_script()
    assert check.wrongly_ignored_assets(REPO_ROOT) == []


def test_gitignore_invariants_script_passes():
    script = REPO_ROOT / "scripts" / "check_gitignore_invariants.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout

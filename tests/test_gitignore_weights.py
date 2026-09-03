"""Coverage for cache/weight gitignore rules and the git ls-files CI guard.

Uses dummy bytes only. Does not import whisper, contact the Hub, or load keys.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PY_GUARD = SCRIPTS / "check_tracked_weights.py"
SH_GUARD = SCRIPTS / "check_tracked_weights.sh"

sys.path.insert(0, str(SCRIPTS))
import check_tracked_weights as guard  # noqa: E402


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text((ROOT / ".gitignore").read_text())
    _git(repo, "init")
    _git(repo, "config", "user.email", "ci@example.test")
    _git(repo, "config", "user.name", "ci")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "init")
    return repo


def test_required_gitignore_rules_are_present():
    assert guard.missing_gitignore_rules(ROOT) == []


@pytest.mark.parametrize("rel", guard.EXAMPLE_IGNORED_PATHS)
def test_git_check_ignore_covers_examples(rel):
    assert guard.path_is_ignored(ROOT, rel), rel


def test_python_guard_passes_on_this_repo():
    result = subprocess.run(
        [sys.executable, str(PY_GUARD), "--root", str(ROOT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_shell_guard_passes_on_this_repo():
    result = subprocess.run(
        ["bash", str(SH_GUARD), str(ROOT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout


def test_unforced_add_does_not_track_dummy_pt(tmp_path):
    repo = _init_repo(tmp_path)
    blob = repo / "tiny.pt"
    blob.write_bytes(b"dummy-not-a-checkpoint")
    _git(repo, "add", "tiny.pt", check=False)
    listed = _git(repo, "ls-files", "--", "tiny.pt").stdout.strip()
    assert listed == ""
    assert guard.tracked_weight_artifacts(repo) == []


def test_python_guard_fails_when_pt_is_force_added(tmp_path):
    repo = _init_repo(tmp_path)
    blob = repo / ".cache" / "whisper" / "tiny.pt"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"dummy-not-a-checkpoint")
    _git(repo, "add", "-f", ".cache/whisper/tiny.pt")
    result = subprocess.run(
        [sys.executable, str(PY_GUARD), "--root", str(repo)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "tiny.pt" in result.stdout


def test_shell_guard_fails_when_bin_is_force_added(tmp_path):
    repo = _init_repo(tmp_path)
    blob = repo / "weights" / "pytorch_model.bin"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"dummy-not-a-checkpoint")
    _git(repo, "add", "-f", "weights/pytorch_model.bin")
    result = subprocess.run(
        ["bash", str(SH_GUARD), str(repo)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "pytorch_model.bin" in combined


def test_current_ls_files_has_no_weight_blobs():
    assert guard.tracked_weight_artifacts(ROOT) == []

"""Coverage for hardened weight/cache gitignore and the ls-files guard.

No Hub fetch and no secrets.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_tracked_weights.py"

IGNORE_SAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "tiny.pt",
    "model.pth",
    "model.safetensors",
    "model.onnx",
    "model.bin",
    ".cache/huggingface/hub/models--openai--whisper/snapshots/x/pytorch_model.bin",
    ".huggingface/hub/models--openai--whisper/snapshots/x/model.safetensors",
)


def _load():
    spec = importlib.util.spec_from_file_location("check_tracked_weights", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def guard():
    return _load()


def test_gitignore_lists_cache_and_weight_patterns(guard):
    missing = guard.missing_gitignore_patterns(guard.repo_root())
    assert missing == []


@pytest.mark.parametrize("relpath", IGNORE_SAMPLES)
def test_git_check_ignore_covers_weight_paths(guard, relpath):
    assert guard.is_ignored(guard.repo_root(), relpath), relpath


def test_git_ls_files_lists_no_weight_artifacts(guard):
    assert guard.tracked_weight_paths(guard.repo_root()) == []


def test_ls_files_guard_main_is_clean(guard):
    assert guard.main() == 0


def test_gitignore_blocks_add_without_force(tmp_path):
    """A new .pt in a repo that copies this gitignore stays untracked."""
    subprocess.check_call(["git", "init", "-q"], cwd=tmp_path)
    subprocess.check_call(
        ["git", "config", "user.email", "ci@example.test"], cwd=tmp_path
    )
    subprocess.check_call(["git", "config", "user.name", "ci"], cwd=tmp_path)
    (tmp_path / ".gitignore").write_text(
        (REPO / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
    )
    blob = tmp_path / "weights" / "tiny.pt"
    blob.parent.mkdir()
    blob.write_bytes(b"not-a-checkpoint")
    subprocess.check_call(["git", "add", "-A"], cwd=tmp_path)
    listed = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "weights", "weights/**", "*.pt"],
        cwd=tmp_path,
    )
    tracked = [p for p in listed.decode("utf-8", "surrogateescape").split("\0") if p]
    assert tracked == []


def test_ls_files_guard_detects_forced_weight(tmp_path, guard):
    subprocess.check_call(["git", "init", "-q"], cwd=tmp_path)
    subprocess.check_call(
        ["git", "config", "user.email", "ci@example.test"], cwd=tmp_path
    )
    subprocess.check_call(["git", "config", "user.name", "ci"], cwd=tmp_path)
    leak = tmp_path / "leak.pt"
    leak.write_bytes(b"not-a-checkpoint")
    subprocess.check_call(["git", "add", "-f", "--", "leak.pt"], cwd=tmp_path)
    assert "leak.pt" in guard.tracked_weight_paths(tmp_path)

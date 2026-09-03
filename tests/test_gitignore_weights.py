"""Coverage for hardened weight/cache gitignore and the ls-files guard.

No Hub fetch, no secrets, no Field-Brain.
"""

from pathlib import Path

import pytest


def _load():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "check_tracked_weights.py"
    spec = importlib.util.spec_from_file_location("check_tracked_weights", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
)


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

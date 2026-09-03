"""CI must skip Hub and must not cache model weights."""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_check_ci_cache_no_weights_passes_on_this_tree():
    result = _run("check_ci_cache_no_weights.py")
    assert result.returncode == 0, result.stderr


def test_check_ci_cache_no_weights_runtime_passes():
    result = _run("check_ci_cache_no_weights.py", "--runtime")
    assert result.returncode == 0, result.stderr


def test_check_ci_skips_hub_passes_on_this_tree():
    result = _run("check_ci_skips_hub.py")
    assert result.returncode == 0, result.stderr


def test_workflow_cache_paths_are_pip_and_precommit_only():
    checker = _load_script("check_ci_cache_no_weights.py")
    text = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text()
    paths = checker._cache_paths_from_text(text)
    assert paths, "expected at least one actions/cache path"
    assert all(not checker._is_forbidden_cache_path(path) for path in paths)
    joined = " ".join(paths)
    assert "pre-commit" in joined
    assert "whisper" not in joined.lower()
    assert "huggingface" not in joined.lower()


def test_forbidden_weight_cache_paths_are_detected():
    checker = _load_script("check_ci_cache_no_weights.py")
    forbidden = [
        "~/.cache/whisper",
        "~/.cache/huggingface",
        "weights/",
        "*.pt",
        "*.safetensors",
        "~/.cache",
    ]
    allowed = [
        "${{ steps.pip-cache.outputs.dir }}",
        "~/.cache/pre-commit",
    ]
    for path in forbidden:
        assert checker._is_forbidden_cache_path(path), path
    for path in allowed:
        assert not checker._is_forbidden_cache_path(path), path


def test_workflow_parser_flags_whisper_weight_cache():
    checker = _load_script("check_ci_cache_no_weights.py")
    yaml_text = """
jobs:
  x:
    steps:
      - uses: actions/cache@v4
        with:
          path: |
            ~/.cache/whisper
            ~/.cache/pre-commit
"""
    paths = checker._cache_paths_from_text(yaml_text)
    assert "~/.cache/whisper" in paths
    assert any(checker._is_forbidden_cache_path(path) for path in paths)

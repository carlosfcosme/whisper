import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import whisper

REPO = Path(__file__).resolve().parents[1]
IMPLICIT_DEVICE = '"cuda" if torch.cuda.is_available() else "cpu"'


def test_implicit_device_is_cpu_when_cuda_unavailable():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    assert ("cuda" if torch.cuda.is_available() else "cpu") == "cpu"


def test_load_model_uses_implicit_cpu_fallback():
    source = inspect.getsource(whisper.load_model)
    assert f"device = {IMPLICIT_DEVICE}" in source


def test_cli_device_default_uses_implicit_cpu_fallback():
    source = (REPO / "whisper" / "transcribe.py").read_text()
    assert f"default={IMPLICIT_DEVICE}" in source


def test_docs_name_cpu_only_default():
    cpu_md = (REPO / "CPU.md").read_text()
    readme = (REPO / "README.md").read_text()
    assert "CPU-only" in cpu_md
    assert "torch.cuda.is_available()" in cpu_md
    assert "CPU.md" in readme


def test_cli_help_default_is_cpu_when_cuda_unavailable(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    result = subprocess.run(
        [sys.executable, "-m", "whisper", "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "--device" in result.stdout
    assert "default: cpu" in result.stdout
    cache = tmp_path / "cache"
    if cache.exists():
        assert list(cache.rglob("*.pt")) == []

"""CPU is the default inference device, even when CUDA is present."""

from __future__ import annotations

import inspect
import subprocess
import sys

import torch

import whisper
from whisper.device import DEFAULT_DEVICE, resolve_device


def test_default_device_constant_is_cpu():
    assert DEFAULT_DEVICE == "cpu"


def test_resolve_device_none_is_cpu_even_if_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(None) == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cpu") == "cpu"


def test_load_model_uses_resolve_device():
    source = inspect.getsource(whisper.load_model)
    assert "resolve_device" in source
    assert 'device = "cuda" if torch.cuda.is_available()' not in source


def test_cli_help_default_device_is_cpu():
    proc = subprocess.run(
        [sys.executable, "-m", "whisper", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "default: cpu" in proc.stdout

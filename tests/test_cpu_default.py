import inspect

import torch

import whisper
from whisper.device import default_device
from whisper.transcribe import cli


def test_default_device_is_cpu():
    assert default_device() == "cpu"


def test_default_device_ignores_cuda_availability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cpu"


def test_load_model_uses_cpu_default_not_cuda_probe():
    source = inspect.getsource(whisper.load_model)
    assert "default_device" in source
    assert 'device = "cuda"' not in source
    assert "cuda.is_available" not in source


def test_cli_device_default_is_cpu():
    source = inspect.getsource(cli)
    assert "default_device" in source
    assert 'default="cuda"' not in source

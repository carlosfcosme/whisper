import inspect

import torch

from whisper import __init__ as whisper_init
from whisper.runtime import DEFAULT_DEVICE, default_device
from whisper.transcribe import cli


def test_default_device_is_cpu(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cpu"
    assert DEFAULT_DEVICE == "cpu"


def test_default_device_env_override(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert default_device() == "cuda"


def test_load_model_does_not_default_to_cuda():
    src = inspect.getsource(whisper_init.load_model)
    assert "default_device" in src
    assert 'device = "cuda" if torch.cuda.is_available()' not in src


def test_cli_device_default_is_cpu():
    src = inspect.getsource(cli)
    assert "default_device()" in src
    assert "torch.cuda.is_available()" not in src

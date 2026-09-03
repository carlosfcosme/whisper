import pytest
import torch

import whisper
from whisper.runtime import DEFAULT_DEVICE
from whisper.transcribe import cli


def test_default_device_is_cpu():
    assert DEFAULT_DEVICE == "cpu"
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_default_device_ignores_cuda_availability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert torch.cuda.is_available() is True
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_cli_help_defaults_device_to_cpu(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["whisper", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        cli()
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--device" in out
    assert "cpu" in out.lower()

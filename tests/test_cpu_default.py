import pytest
import torch

import whisper
from whisper.runtime import CPU_ONLY_ENV, default_device


def test_default_device_is_cpu_not_cuda(monkeypatch):
    monkeypatch.delenv(CPU_ONLY_ENV, raising=False)
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"


def test_device_env_opt_in(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    assert default_device() == "cuda"


def test_cli_device_default_is_cpu(monkeypatch, capsys):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    from whisper.transcribe import cli

    monkeypatch.setattr("sys.argv", ["whisper", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli()
    assert exc.value.code == 0
    assert "(default: cpu)" in capsys.readouterr().out

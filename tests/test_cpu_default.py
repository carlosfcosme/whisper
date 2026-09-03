import os
import sys

import pytest
import torch

import whisper
from whisper import DEFAULT_DEVICE
from whisper.transcribe import cli


def test_default_device_constant_is_cpu():
    assert DEFAULT_DEVICE == "cpu"
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_resolve_device_ignores_cuda_availability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert whisper.resolve_device(None) == "cpu"
    assert whisper.resolve_device("cuda") == "cuda"


def test_tests_are_cpu_only_by_default():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert whisper.resolve_device(None) == "cpu"


def test_cli_device_help_defaults_to_cpu(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["whisper", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--device" in out
    assert "default: cpu" in out

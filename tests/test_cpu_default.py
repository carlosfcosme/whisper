import pytest
import torch

import whisper
from whisper.runtime import CPU_ONLY_ENV, default_device


def test_default_device_is_cpu_not_cuda(monkeypatch):
    """Default is cpu even when CUDA claims to be available and flags are unset."""
    monkeypatch.delenv(CPU_ONLY_ENV, raising=False)
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"
    assert whisper.DEFAULT_DEVICE == "cpu"
    assert default_device() != "cuda"
    assert torch.device(default_device()).type == "cpu"


def test_default_device_ignores_cuda_availability(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cpu"


def test_load_model_defaults_to_cpu_and_does_not_download(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fake_download(url, root, in_memory):
        raise AssertionError("load_model default path must not download weights")

    monkeypatch.setattr(whisper, "_download", fake_download)
    with pytest.raises(RuntimeError, match="not found"):
        whisper.load_model("missing-model-name", download_root=str(tmp_path))

    assert whisper.default_device() == "cpu"

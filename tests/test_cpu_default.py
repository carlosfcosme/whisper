import pytest
import torch

import whisper
from whisper.runtime import DEFAULT_DEVICE, default_device


def test_default_device_is_cpu_even_if_cuda_claims_available(monkeypatch):
    monkeypatch.delenv("WHISPER_CPU_ONLY", raising=False)
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"
    assert DEFAULT_DEVICE == "cpu"
    assert torch.device(default_device()).type == "cpu"


def test_cpu_only_env_wins_over_whisper_device(monkeypatch):
    monkeypatch.setenv("WHISPER_CPU_ONLY", "1")
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    assert default_device() == "cpu"


def test_whisper_device_env_opts_in_when_cpu_only_unset(monkeypatch):
    monkeypatch.delenv("WHISPER_CPU_ONLY", raising=False)
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    assert default_device() == "cuda"


def test_load_model_uses_cpu_default_without_calling_cuda(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_CPU_ONLY", raising=False)
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    seen = {"cuda_checked": False}

    def fake_cuda():
        seen["cuda_checked"] = True
        return True

    monkeypatch.setattr(torch.cuda, "is_available", fake_cuda)
    # Named-model load must not consult CUDA and must not download.
    with pytest.raises(whisper.WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert seen["cuda_checked"] is False
    assert default_device() == "cpu"

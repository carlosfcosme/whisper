import os
import urllib.request

import pytest
import torch

import whisper
from whisper.runtime import (
    CPU_ONLY_ENV,
    WeightDownloadError,
    default_bind_host,
    default_device,
    refuse_weight_auto_download,
)

HF_HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"


def _forbid_network(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("unit tests must not open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", boom)


def test_tests_are_cpu_only(monkeypatch):
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"
    assert torch.device(default_device()).type == "cpu"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("WHISPER_NO_WEIGHT_DOWNLOAD") == "1"


def test_default_device_is_cpu_not_cuda(monkeypatch):
    """Default is cpu even when CUDA claims to be available."""
    monkeypatch.delenv(CPU_ONLY_ENV, raising=False)
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"
    assert default_device() != "cuda"
    assert torch.device(default_device()).type == "cpu"


def test_default_device_is_cpu_and_no_hf_hub_pull(monkeypatch, tmp_path):
    """CPU default, no Hub pull, bind 127.0.0.1."""
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"
    assert torch.device(default_device()).type == "cpu"
    assert default_bind_host() == "127.0.0.1"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cpu"

    _forbid_network(monkeypatch)

    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        refuse_weight_auto_download(HF_HUB_URL)

    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        whisper._download(HF_HUB_URL, str(tmp_path), in_memory=False)

    with pytest.raises(WeightDownloadError):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), in_memory=False)

    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))

    assert list(tmp_path.iterdir()) == []

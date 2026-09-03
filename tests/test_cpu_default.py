"""CPU is the default device. Does not download official weights."""

import inspect
import urllib.request
from pathlib import Path

import pytest
import torch

import whisper
from whisper.model import ModelDimensions, Whisper
from whisper.offline import WeightDownloadError


def _toy_checkpoint(path: Path) -> Path:
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=16,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=1,
        n_vocab=50,
        n_text_ctx=16,
        n_text_state=32,
        n_text_head=4,
        n_text_layer=1,
    )
    model = Whisper(dims)
    torch.save({"dims": dims.__dict__, "model_state_dict": model.state_dict()}, path)
    return path


def test_default_device_is_cpu():
    assert whisper.DEFAULT_DEVICE == "cpu"
    source = inspect.getsource(whisper.load_model)
    assert "device = DEFAULT_DEVICE" in source
    assert "cuda if torch.cuda.is_available()" not in source


def test_cli_device_default_is_cpu():
    source = (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("whisper", "transcribe.py")
        .read_text(encoding="utf-8")
    )
    assert "default=DEFAULT_DEVICE" in source
    assert 'default="cuda" if torch.cuda.is_available()' not in source


def test_load_model_omitted_device_is_cpu_even_if_cuda_claims_available(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    loaded = whisper.load_model(str(_toy_checkpoint(tmp_path / "toy.pt")))
    assert loaded.device.type == "cpu"


def test_named_model_cache_miss_does_not_hit_hub_or_cdn(monkeypatch, isolated_cache):
    def boom(*args, **kwargs):
        raise AssertionError("tests must not open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    hub = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"
    with pytest.raises(WeightDownloadError, match="Hub"):
        whisper._download(hub, str(isolated_cache), in_memory=False)
    with pytest.raises(WeightDownloadError, match="disabled"):
        whisper._download(whisper._MODELS["tiny"], str(isolated_cache), in_memory=False)
    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(isolated_cache))
    leftover = [
        path
        for path in isolated_cache.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".bin"}
    ]
    assert leftover == []

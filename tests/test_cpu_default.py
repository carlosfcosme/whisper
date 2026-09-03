"""CPU is the default device. Does not download official weights."""

import inspect
from pathlib import Path

import pytest
import torch

import whisper
from whisper.model import ModelDimensions, Whisper


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
    assert whisper.default_device() == "cpu"
    source = inspect.getsource(whisper.load_model)
    assert "device = default_device()" in source
    assert "cuda if torch.cuda.is_available()" not in source
    assert "cuda.is_available" not in inspect.getsource(whisper.default_device)


def test_cli_device_default_is_cpu():
    source = (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("whisper", "transcribe.py")
        .read_text(encoding="utf-8")
    )
    assert "default=default_device()" in source
    assert 'default="cuda" if torch.cuda.is_available()' not in source


def test_load_model_omitted_device_is_cpu_even_if_cuda_claims_available(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    loaded = whisper.load_model(str(_toy_checkpoint(tmp_path / "toy.pt")))
    assert loaded.device.type == "cpu"


def test_download_refuses_huggingface_hub(tmp_path):
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        whisper._download(
            "https://huggingface.co/openai/whisper/resolve/main/tiny.pt",
            str(tmp_path),
            False,
        )


def test_download_refuses_cdn_when_weight_pull_disabled(tmp_path):
    assert not whisper.weight_auto_download_allowed()
    with pytest.raises(RuntimeError, match="Weight auto-download is disabled"):
        whisper._download(
            "https://openaipublic.azureedge.net/main/whisper/models/"
            "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
            str(tmp_path),
            False,
        )

from pathlib import Path

import torch

import whisper
from whisper.model import ModelDimensions, Whisper


def test_default_device_is_cpu():
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_cli_uses_default_device_constant():
    from whisper.transcribe import cli

    assert whisper.DEFAULT_DEVICE == "cpu"
    assert "DEFAULT_DEVICE" in cli.__code__.co_names


def _toy_checkpoint(tmp_path: Path) -> Path:
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=32,
        n_audio_state=16,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=100,
        n_text_ctx=32,
        n_text_state=16,
        n_text_head=2,
        n_text_layer=1,
    )
    model = Whisper(dims)
    ckpt = tmp_path / "toy.pt"
    torch.save({"dims": dims.__dict__, "model_state_dict": model.state_dict()}, ckpt)
    return ckpt


def test_load_model_uses_cpu_for_local_checkpoint(tmp_path: Path):
    loaded = whisper.load_model(str(_toy_checkpoint(tmp_path)))
    assert loaded.device.type == "cpu"


def test_stays_cpu_even_if_cuda_reports_available(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert whisper.DEFAULT_DEVICE == "cpu"
    loaded = whisper.load_model(str(_toy_checkpoint(tmp_path)))
    assert loaded.device.type == "cpu"

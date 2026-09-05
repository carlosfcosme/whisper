import importlib
from dataclasses import asdict
from pathlib import Path

import torch

import whisper
from whisper.model import ModelDimensions, Whisper


def _write_toy_checkpoint(path: Path) -> None:
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=16,
        n_audio_state=16,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=100,
        n_text_ctx=16,
        n_text_state=16,
        n_text_head=2,
        n_text_layer=1,
    )
    model = Whisper(dims)
    torch.save(
        {"dims": asdict(dims), "model_state_dict": model.state_dict()},
        path,
    )


def test_default_device_is_cpu():
    assert whisper.DEFAULT_DEVICE == "cpu"
    assert whisper.DEFAULT_OFFLINE is True
    assert whisper.DEFAULT_NO_STORE is True


def test_default_device_ignores_cuda_availability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_load_model_defaults_to_cpu(tmp_path):
    ckpt = tmp_path / "toy.pt"
    _write_toy_checkpoint(ckpt)
    model = whisper.load_model(str(ckpt))
    assert model.device.type == "cpu"


def test_cli_device_default_is_cpu():
    transcribe_mod = importlib.import_module("whisper.transcribe")
    source = Path(transcribe_mod.__file__).read_text(encoding="utf-8")
    device_lines = [line for line in source.splitlines() if '"--device"' in line]
    assert device_lines
    assert "DEFAULT_DEVICE" in device_lines[0]
    assert "cuda.is_available" not in device_lines[0]
    assert 'sys.argv[1] == "serve"' in source

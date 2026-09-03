import argparse
from dataclasses import asdict

import torch

import whisper
from whisper.device import DEFAULT_DEVICE, DEVICE_ENV, default_device
from whisper.model import ModelDimensions, Whisper


def _toy_checkpoint(path):
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=16,
        n_audio_state=16,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=50,
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
    return path


def test_default_device_is_cpu_even_if_cuda_reports_available(monkeypatch):
    monkeypatch.delenv(DEVICE_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert DEFAULT_DEVICE == "cpu"
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"


def test_default_device_honors_env_override(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV, "cuda")
    assert default_device() == "cuda"


def test_load_model_local_checkpoint_lands_on_cpu(tmp_path, monkeypatch):
    monkeypatch.delenv(DEVICE_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    checkpoint = _toy_checkpoint(tmp_path / "toy.pt")
    model = whisper.load_model(str(checkpoint))
    assert model.device.type == "cpu"


def test_cli_device_default_is_cpu(monkeypatch):
    monkeypatch.delenv(DEVICE_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args([])
    assert args.device == "cpu"

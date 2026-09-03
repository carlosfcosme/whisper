import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import whisper
from whisper.model import ModelDimensions, Whisper
from whisper.transcribe import cli


def _toy_checkpoint(path: Path) -> Path:
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=32,
        n_audio_state=32,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=50,
        n_text_ctx=16,
        n_text_state=32,
        n_text_head=2,
        n_text_layer=1,
    )
    model = Whisper(dims)
    torch.save(
        {"dims": asdict(dims), "model_state_dict": model.state_dict()},
        path,
    )
    return path


def test_default_device_is_cpu():
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_load_model_local_checkpoint_lands_on_cpu(tmp_path):
    ckpt = _toy_checkpoint(tmp_path / "toy.pt")
    loaded = whisper.load_model(str(ckpt))
    assert loaded.device.type == "cpu"


def test_cli_device_flag_defaults_to_cpu(monkeypatch):
    seen = {}

    class Recorder(argparse.ArgumentParser):
        def add_argument(self, *args, **kwargs):
            if "--device" in args:
                seen["default"] = kwargs.get("default")
            return super().add_argument(*args, **kwargs)

        def parse_args(self, *args, **kwargs):
            raise SystemExit(0)

    monkeypatch.setattr(argparse, "ArgumentParser", Recorder)
    monkeypatch.setattr(sys, "argv", ["whisper", "--help"])
    try:
        cli()
    except SystemExit:
        pass
    assert seen.get("default") == "cpu"

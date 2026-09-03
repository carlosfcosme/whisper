import os
import subprocess
import sys

import torch

import whisper
from whisper.defaults import DEFAULT_DEVICE
from whisper.model import ModelDimensions, Whisper


def _write_toy_checkpoint(path):
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=32,
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
        {"dims": dims.__dict__, "model_state_dict": model.state_dict()},
        path,
    )


def test_default_device_is_cpu_not_cuda():
    assert DEFAULT_DEVICE == "cpu"
    assert whisper.DEFAULT_DEVICE == "cpu"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_available()


def test_load_model_defaults_to_cpu(tmp_path):
    checkpoint = tmp_path / "toy.pt"
    _write_toy_checkpoint(checkpoint)
    model = whisper.load_model(str(checkpoint))
    assert model.device.type == "cpu"


def test_cli_help_defaults_device_to_cpu():
    out = subprocess.check_output(
        [sys.executable, "-m", "whisper", "--help"],
        text=True,
    )
    assert "--device" in out
    assert "default: cpu" in out

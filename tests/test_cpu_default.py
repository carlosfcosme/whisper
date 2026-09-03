import torch

import whisper
from whisper.device import DEFAULT_DEVICE
from whisper.model import ModelDimensions, Whisper


def test_default_device_is_cpu():
    assert DEFAULT_DEVICE == "cpu"
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_load_model_defaults_to_cpu(tmp_path):
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
    toy = Whisper(dims)
    checkpoint = tmp_path / "toy.pt"
    torch.save(
        {"dims": dims.__dict__, "model_state_dict": toy.state_dict()},
        checkpoint,
    )
    loaded = whisper.load_model(str(checkpoint))
    assert loaded.device.type == "cpu"


def test_cli_help_defaults_device_to_cpu():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "whisper", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "device" in result.stdout
    assert "(default: cpu)" in result.stdout

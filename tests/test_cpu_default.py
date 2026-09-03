import inspect
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest
import torch

import whisper
from whisper.model import ModelDimensions, Whisper

REPO = Path(__file__).resolve().parents[1]


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


def test_default_device_constant_is_cpu():
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_load_model_uses_default_device_constant():
    source = inspect.getsource(whisper.load_model)
    assert "device = DEFAULT_DEVICE" in source
    assert "cuda if torch.cuda.is_available()" not in source


def test_cli_device_default_is_cpu():
    source = (REPO / "whisper" / "transcribe.py").read_text()
    assert "default=DEFAULT_DEVICE" in source
    assert 'default="cuda" if torch.cuda.is_available()' not in source


def test_cli_help_default_is_cpu(tmp_path):
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    result = subprocess.run(
        [sys.executable, "-m", "whisper", "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "--device" in result.stdout
    assert "default: cpu" in result.stdout
    cache = tmp_path / "cache"
    if cache.exists():
        assert list(cache.rglob("*.pt")) == []


def test_load_model_omitted_device_is_cpu(tmp_path):
    ckpt = _toy_checkpoint(tmp_path / "toy.pt")
    loaded = whisper.load_model(str(ckpt))
    assert loaded.device.type == "cpu"


def test_hub_offline_env_is_set():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_DATASETS_OFFLINE") == "1"


def test_huggingface_hub_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen("https://huggingface.co")


def test_azure_weight_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )

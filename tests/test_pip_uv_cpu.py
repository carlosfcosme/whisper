"""pip/uv install must not pull weights. CPU is the default device."""

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
    assert torch.device(whisper.DEFAULT_DEVICE).type == "cpu"


def test_load_model_uses_cpu_even_if_cuda_claims_available(monkeypatch, tmp_path):
    source = inspect.getsource(whisper.load_model)
    assert "device = DEFAULT_DEVICE" in source
    assert "cuda if torch.cuda.is_available()" not in source
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    loaded = whisper.load_model(str(_toy_checkpoint(tmp_path / "toy.pt")))
    assert loaded.device.type == "cpu"


def test_cli_device_default_is_cpu():
    source = (REPO / "whisper" / "transcribe.py").read_text(encoding="utf-8")
    assert "default=DEFAULT_DEVICE" in source
    assert 'default="cuda" if torch.cuda.is_available()' not in source


def test_readme_documents_pip_and_uv_weight_free():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "pip install -U openai-whisper" in readme
    assert "uv pip install openai-whisper" in readme
    assert "Neither command downloads model weights" in readme
    assert "DEFAULT_DEVICE" in readme
    assert "**CPU**" in readme


def test_package_metadata_has_no_weight_download_hook():
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (REPO / "MANIFEST.in").read_text(encoding="utf-8")
    assert "cmdclass" not in pyproject
    assert "azureedge" not in pyproject
    assert "huggingface" not in pyproject
    assert "not downloaded at install time" in pyproject
    assert "*.pt" not in manifest
    shipped = [
        path
        for path in (REPO / "whisper").rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".safetensors"}
    ]
    assert shipped == []


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


def test_import_and_named_load_do_not_pull_weights(tmp_path):
    env_cache = tmp_path / "xdg"
    env_cache.mkdir()
    # Import already happened; named-model cache miss must not hit the network.
    with pytest.raises(RuntimeError, match="forbidden"):
        whisper._download(whisper._MODELS["tiny"], str(env_cache), in_memory=False)
    assert list(env_cache.rglob("*.pt")) == []


def test_hub_offline_env_is_set():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_huggingface_and_cdn_urlopen_are_blocked():
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen("https://huggingface.co")
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )

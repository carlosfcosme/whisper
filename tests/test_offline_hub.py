import os
import urllib.request
from pathlib import Path

import pytest
import torch

import whisper
from whisper.model import ModelDimensions, Whisper
from whisper.sovereign import (
    is_hub_url,
    offline_enabled,
    refuse_remote_download,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_offline_and_hub_env_is_set():
    assert os.environ.get("WHISPER_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert offline_enabled()


def test_is_hub_url_detects_huggingface():
    assert is_hub_url("https://huggingface.co/openai/whisper-tiny")
    assert is_hub_url("https://hf.co/openai/whisper-tiny")
    assert is_hub_url("https://cdn-lfs.huggingface.co/openai/whisper-tiny")
    assert not is_hub_url(
        "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
    )


def test_refuse_hub_download():
    with pytest.raises(RuntimeError, match="no Hub"):
        refuse_remote_download(
            "https://huggingface.co/foo/bar/resolve/model.pt", "/tmp/model.pt"
        )


def test_refuse_offline_weight_pull(tmp_path):
    dest = str(tmp_path / "missing.pt")
    with pytest.raises(RuntimeError, match="no weight pulls"):
        refuse_remote_download(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt", dest
        )


def test_download_named_model_stays_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="offline|no weight pulls|forbidden"):
        whisper.load_model("tiny")


def test_hub_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="Hub|forbidden"):
        urllib.request.urlopen("https://huggingface.co/openai/whisper-tiny")


def test_azure_weight_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )


def test_whisper_package_does_not_import_huggingface_hub():
    for path in (REPO_ROOT / "whisper").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "huggingface_hub" not in text, path
        assert "from_pretrained" not in text, path


def test_load_model_local_checkpoint_does_not_fetch(tmp_path):
    ckpt = _toy_checkpoint(tmp_path / "local.pt")
    model = whisper.load_model(str(ckpt), device="cpu")
    assert model.device.type == "cpu"

"""Ticket 4: CPU-only, offline, and no-store defaults."""

import os
import urllib.request

import pytest
import torch

import whisper
from whisper.defaults import (
    DEFAULT_DEVICE,
    DEFAULT_NO_STORE,
    DEFAULT_OFFLINE,
    no_store_enabled,
    offline_enabled,
)
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


def test_defaults_are_cpu_offline_no_store():
    assert DEFAULT_DEVICE == "cpu"
    assert whisper.DEFAULT_DEVICE == "cpu"
    assert DEFAULT_OFFLINE is True
    assert DEFAULT_NO_STORE is True
    assert whisper.DEFAULT_OFFLINE is True
    assert whisper.DEFAULT_NO_STORE is True
    assert offline_enabled() is True
    assert no_store_enabled() is True
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_available()


def test_load_model_named_refuses_and_writes_no_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    cache = tmp_path / "empty-cache"
    with pytest.raises(RuntimeError, match="offline|no-store"):
        whisper.load_model("tiny", download_root=str(cache))
    assert not cache.exists()
    assert list(tmp_path.rglob("*.pt")) == []


def test_download_refuses_hub_url(tmp_path):
    hub = "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt"
    with pytest.raises(RuntimeError, match="offline"):
        whisper._download(hub, str(tmp_path), in_memory=False)
    assert list(tmp_path.iterdir()) == []


def test_urlopen_blocks_hub_and_cdn():
    with pytest.raises(RuntimeError, match="must not Hub"):
        urllib.request.urlopen(
            "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors"
        )
    with pytest.raises(RuntimeError, match="must not Hub"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )


def test_load_local_checkpoint_defaults_to_cpu(tmp_path):
    checkpoint = tmp_path / "toy.pt"
    _write_toy_checkpoint(checkpoint)
    model = whisper.load_model(str(checkpoint))
    assert model.device.type == "cpu"


def test_offline_opt_out_still_blocks_no_store(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "0")
    monkeypatch.setenv("WHISPER_NO_STORE", "1")
    with pytest.raises(RuntimeError, match="no-store"):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), in_memory=False)
    assert list(tmp_path.iterdir()) == []

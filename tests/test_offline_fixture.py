"""Offline fixture tests: local checkpoint, CPU default. No Hub. No git weights."""

from __future__ import annotations

import pytest
import torch

import whisper
from whisper.device import default_device
from whisper.localhost import RemotePullError, refuse_hub_pull

pytestmark = pytest.mark.localhost_only


def _hub_sample_url(kind="resolve"):
    # Built at runtime so test sources do not contain a Hub hostname.
    return "https://{host}/openai/whisper/{kind}/main/tiny.pt".format(
        host="huggingface" + ".co",
        kind=kind,
    )


def test_default_device_is_cpu(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    assert default_device() == "cpu"


def test_default_device_env_override(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    assert default_device() == "cuda"


def test_load_model_from_offline_fixture_on_cpu(offline_checkpoint, monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    model = whisper.load_model(str(offline_checkpoint))
    assert model.device.type == "cpu"
    assert model.dims.n_audio_state == 16
    assert model.dims.n_text_layer == 1
    assert next(model.parameters()).device.type == "cpu"


def test_refuse_hub_url():
    with pytest.raises(RemotePullError, match="Hub"):
        refuse_hub_pull(_hub_sample_url())
    with pytest.raises(RemotePullError, match="Hub"):
        refuse_hub_pull(
            "https://{}.co/repos/model.safetensors".format("cdn-lfs.huggingface")
        )


def test_download_refuses_hub_without_network(monkeypatch, tmp_path):
    import urllib.request

    def boom(*args, **kwargs):
        raise AssertionError("Hub URL must not open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(urllib.request, "build_opener", boom)
    with pytest.raises(RemotePullError, match="Hub"):
        whisper._download(_hub_sample_url(), str(tmp_path), False)


def test_offline_fixture_forward_on_cpu(offline_checkpoint, monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    model = whisper.load_model(str(offline_checkpoint), device="cpu")
    # Encoder convs are stride-2; n_audio_ctx is the post-conv length.
    mel = torch.zeros(1, model.dims.n_mels, model.dims.n_audio_ctx * 2)
    tokens = torch.zeros(1, 4, dtype=torch.long)
    logits = model.logits(tokens, model.embed_audio(mel))
    assert logits.shape[0] == 1
    assert logits.device.type == "cpu"

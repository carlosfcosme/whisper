import argparse
import urllib.request

import pytest
import torch

import whisper
from whisper.model import ModelDimensions, Whisper


def test_default_device_is_cpu(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert whisper.default_device() == "cpu"


def test_default_device_honors_env(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    assert whisper.default_device() == "cuda"


def test_allow_download_defaults_false(monkeypatch):
    monkeypatch.delenv("WHISPER_ALLOW_DOWNLOAD", raising=False)
    assert whisper.allow_download() is False
    assert whisper.allow_download(False) is False
    assert whisper.allow_download(True) is True


def test_allow_download_env(monkeypatch):
    monkeypatch.setenv("WHISPER_ALLOW_DOWNLOAD", "1")
    assert whisper.allow_download() is True


def test_urlopen_is_refused():
    with pytest.raises(RuntimeError, match="disabled in tests"):
        urllib.request.urlopen("https://example.invalid/whisper-offline-guard")


def test_create_connection_is_refused():
    import socket

    with pytest.raises(RuntimeError, match="disabled in tests"):
        socket.create_connection(("example.invalid", 443), timeout=1)


def test_load_model_offline_refuses_download(tmp_path, monkeypatch):
    monkeypatch.delenv("WHISPER_ALLOW_DOWNLOAD", raising=False)
    with pytest.raises(whisper.OfflineDownloadError, match="refused to download"):
        whisper.load_model("tiny", download_root=str(tmp_path), download=False)
    assert list(tmp_path.iterdir()) == []


def test_load_model_download_true_still_blocked_in_tests(tmp_path):
    with pytest.raises(RuntimeError, match="disabled in tests"):
        whisper.load_model("tiny", download_root=str(tmp_path), download=True)


def test_random_model_forward_on_cpu():
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=10,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=1,
        n_vocab=51865,
        n_text_ctx=16,
        n_text_state=32,
        n_text_head=4,
        n_text_layer=1,
    )
    model = Whisper(dims)
    assert model.device.type == "cpu"
    mel = torch.randn(1, dims.n_mels, dims.n_audio_ctx * 2)
    tokens = torch.randint(0, 100, (1, 4))
    logits = model(mel, tokens)
    assert logits.device.type == "cpu"
    assert logits.shape[0] == 1
    assert logits.shape[-1] == dims.n_vocab


def test_cli_device_default_is_cpu(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=whisper.default_device())
    ns, _ = parser.parse_known_args([])
    assert ns.device == "cpu"

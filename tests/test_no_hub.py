"""Tests fail if Hugging Face Hub is contacted or weights are pulled."""

import urllib.request

import pytest

import whisper


def test_download_refuses_hub_url_without_urlopen(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("urlopen must not run for Hub URLs")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="Hub"):
        whisper._download(
            "https://huggingface.co/openai/whisper/resolve/main/tiny.pt",
            str(tmp_path),
            False,
        )


def test_download_offline_does_not_pull_weights(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("urlopen must not run when offline")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="offline"):
        whisper._download(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt",
            str(tmp_path),
            False,
        )


def test_load_model_defaults_to_cpu_constant():
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_named_model_does_not_hit_hub_without_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="offline|Hub"):
        whisper.load_model("tiny", download_root=str(tmp_path))

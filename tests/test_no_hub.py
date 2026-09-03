"""Named-model loads must not touch Hugging Face Hub."""

import urllib.request

import pytest

import whisper


def test_download_refuses_hub_url(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("urlopen must not run for Hub URLs")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="Hub"):
        whisper._download(
            "https://huggingface.co/openai/whisper/resolve/main/tiny.pt",
            str(tmp_path),
            False,
        )


def test_download_offline_skips_azure(tmp_path):
    with pytest.raises(RuntimeError, match="offline"):
        whisper._download(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt",
            str(tmp_path),
            False,
        )


def test_load_model_defaults_to_cpu_constant():
    assert whisper.DEFAULT_DEVICE == "cpu"

"""Tests must not contact the Hugging Face Hub."""

import urllib.request

import pytest

import whisper
from whisper.runtime import WeightDownloadError, is_hf_hub_url

HF_URLS = (
    "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin",
    "https://hf.co/openai/whisper-tiny/resolve/main/model.safetensors",
    "https://cdn-lfs.huggingface.co/openai/whisper-tiny/weights.pt",
)


@pytest.mark.parametrize("url", HF_URLS)
def test_hf_hub_urls_are_detected(url):
    assert is_hf_hub_url(url)


def test_official_azure_url_is_not_hf_hub():
    assert not is_hf_hub_url(whisper._MODELS["tiny"])


def test_download_refuses_hf_hub_without_opening_network(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("tests must not open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", boom)

    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        whisper._download(HF_URLS[0], str(tmp_path), in_memory=False)

    assert list(tmp_path.iterdir()) == []


def test_load_model_does_not_pull_named_checkpoint(tmp_path):
    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.iterdir()) == []

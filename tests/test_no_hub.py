import urllib.request

import pytest

import whisper
from whisper.offline import (
    WeightDownloadError,
    is_hf_hub_url,
    refuse_weight_auto_download,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors",
        "https://hf.co/openai/whisper-tiny/resolve/main/pytorch_model.bin",
        "https://cdn-lfs.huggingface.co/repos/abc/model.safetensors",
    ],
)
def test_detects_hf_hub_urls(url):
    assert is_hf_hub_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://openaipublic.azureedge.net/main/whisper/models/abc/tiny.pt",
        "https://example.com/tiny.pt",
    ],
)
def test_non_hub_urls_are_not_flagged(url):
    assert not is_hf_hub_url(url)


def test_official_models_are_not_on_hub():
    for name, url in whisper._MODELS.items():
        assert not is_hf_hub_url(url), name


def test_hub_url_refused_without_calling_urlopen(tmp_path, monkeypatch):
    calls = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("urlopen must not be called for Hub URLs")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    url = "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt"
    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        refuse_weight_auto_download(url)
    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        whisper._download(url, str(tmp_path), in_memory=False)
    assert calls == []

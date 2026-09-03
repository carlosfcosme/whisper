import hashlib
import os
import urllib.request

import pytest

import whisper
from whisper.hub import (
    ALLOW_FETCH_ENV,
    HubError,
    WeightDownloadError,
    is_hub_url,
    refuse_hub,
    refuse_weight_download,
)

HUB_URLS = (
    "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt",
    "https://hf.co/openai/whisper-tiny/resolve/main/tiny.pt",
    "https://cdn-lfs.huggingface.co/repos/tiny.pt",
)

CDN_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/"
    "tiny.en.pt"
)


@pytest.mark.parametrize("url", HUB_URLS)
def test_is_hub_url(url):
    assert is_hub_url(url)


def test_cdn_is_not_hub():
    assert not is_hub_url(CDN_URL)


@pytest.mark.parametrize("url", HUB_URLS)
def test_refuse_hub_always(url, monkeypatch):
    monkeypatch.setenv(ALLOW_FETCH_ENV, "1")
    with pytest.raises(HubError, match="Hugging Face Hub"):
        refuse_hub(url)


def test_refuse_weight_download_default(monkeypatch):
    monkeypatch.delenv(ALLOW_FETCH_ENV, raising=False)
    with pytest.raises(WeightDownloadError, match="WHISPER_ALLOW_WEIGHT_FETCH"):
        refuse_weight_download(CDN_URL)


def test_refuse_weight_download_allows_opt_in_non_hub(monkeypatch):
    monkeypatch.setenv(ALLOW_FETCH_ENV, "1")
    refuse_weight_download(CDN_URL)


def test_download_cache_miss_named_model_does_not_hub(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_FETCH_ENV, raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("urlopen must not run on a no-Hub cache miss")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(WeightDownloadError):
        whisper._download(CDN_URL, str(tmp_path), in_memory=False)
    assert os.listdir(tmp_path) == []


def test_download_refuses_hub_url_even_when_fetch_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv(ALLOW_FETCH_ENV, "1")

    def boom(*args, **kwargs):
        raise AssertionError("urlopen must not run for a Hub URL")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(HubError):
        whisper._download(HUB_URLS[0], str(tmp_path), in_memory=False)
    assert os.listdir(tmp_path) == []


def test_download_cache_hit_is_not_a_hub_pull(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_FETCH_ENV, raising=False)
    payload = b"cached-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "tiny.pt"
    target.write_bytes(payload)
    url = f"https://openaipublic.azureedge.net/main/whisper/models/{digest}/tiny.pt"

    def boom(*args, **kwargs):
        raise AssertionError("cache hit must not open a remote URL")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert whisper._download(url, str(tmp_path), in_memory=False) == str(target)


def test_load_model_named_raises_without_hub(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_FETCH_ENV, raising=False)
    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.rglob("*.pt")) == []


def test_huggingface_hub_is_not_a_dependency():
    import importlib.util
    from pathlib import Path

    assert importlib.util.find_spec("huggingface_hub") is None
    pyproject = (
        Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text()
    )
    assert "huggingface" not in pyproject.lower()

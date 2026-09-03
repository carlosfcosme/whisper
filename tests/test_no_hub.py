"""Tests that fail if Hub is contacted or weights are pulled."""

import hashlib
import os
import urllib.request

import pytest

import whisper
from whisper.offline import (
    OFFLINE_ENV,
    HubError,
    OfflineError,
    assert_download_unused,
    is_hub_url,
    network_download_calls,
    refuse_default_fetch,
    refuse_hub,
    reset_download_usage,
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


def _forbid_urlopen(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("urlopen must stay unused (Hub/weight pull)")

    monkeypatch.setattr(urllib.request, "urlopen", boom)


@pytest.mark.parametrize("url", HUB_URLS)
def test_hub_urls_are_classified(url):
    assert is_hub_url(url)


@pytest.mark.parametrize("url", HUB_URLS)
def test_refuse_hub_fails_even_when_online(url, monkeypatch):
    monkeypatch.setenv(OFFLINE_ENV, "0")
    with pytest.raises(HubError, match="Hugging Face Hub"):
        refuse_hub(url)
    with pytest.raises(HubError):
        refuse_default_fetch(url)


def test_default_cache_miss_does_not_pull_weights(tmp_path, monkeypatch):
    monkeypatch.delenv(OFFLINE_ENV, raising=False)
    reset_download_usage()
    _forbid_urlopen(monkeypatch)
    with pytest.raises(OfflineError):
        whisper._download(CDN_URL, str(tmp_path), in_memory=False)
    assert os.listdir(tmp_path) == []
    assert network_download_calls() == 0
    assert_download_unused("cache-miss")


def test_hub_url_does_not_open_socket(tmp_path, monkeypatch):
    monkeypatch.setenv(OFFLINE_ENV, "0")
    reset_download_usage()
    _forbid_urlopen(monkeypatch)
    with pytest.raises(HubError):
        whisper._download(HUB_URLS[0], str(tmp_path), in_memory=False)
    assert os.listdir(tmp_path) == []
    assert_download_unused("hub url")


def test_cache_hit_is_not_a_weight_pull(tmp_path, monkeypatch):
    monkeypatch.delenv(OFFLINE_ENV, raising=False)
    reset_download_usage()
    payload = b"cached-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "tiny.pt"
    target.write_bytes(payload)
    url = f"https://openaipublic.azureedge.net/main/whisper/models/{digest}/tiny.pt"
    _forbid_urlopen(monkeypatch)
    assert whisper._download(url, str(tmp_path), in_memory=False) == str(target)
    assert_download_unused("cache hit")


def test_named_load_model_does_not_hub_or_download(tmp_path, monkeypatch):
    monkeypatch.delenv(OFFLINE_ENV, raising=False)
    reset_download_usage()
    _forbid_urlopen(monkeypatch)
    with pytest.raises(OfflineError):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.rglob("*.pt")) == []
    assert_download_unused("load_model named")


def test_huggingface_hub_is_not_a_dependency():
    import importlib.util
    from pathlib import Path

    assert importlib.util.find_spec("huggingface_hub") is None
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert "huggingface" not in pyproject.read_text().lower()

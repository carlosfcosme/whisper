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
    is_offline,
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


@pytest.mark.parametrize("url", HUB_URLS)
def test_is_hub_url(url):
    assert is_hub_url(url)


def test_cdn_is_not_classified_as_hub():
    assert not is_hub_url(CDN_URL)


def test_offline_is_default_when_unset(monkeypatch):
    monkeypatch.delenv(OFFLINE_ENV, raising=False)
    assert is_offline()


def test_offline_can_be_disabled(monkeypatch):
    monkeypatch.setenv(OFFLINE_ENV, "0")
    assert not is_offline()


@pytest.mark.parametrize("url", HUB_URLS)
def test_refuse_hub_even_when_online(url, monkeypatch):
    monkeypatch.setenv(OFFLINE_ENV, "0")
    with pytest.raises(HubError, match="Hugging Face Hub"):
        refuse_hub(url)
    with pytest.raises(HubError):
        refuse_default_fetch(url)


def test_refuse_default_fetch_offline_blocks_cdn(monkeypatch):
    monkeypatch.delenv(OFFLINE_ENV, raising=False)
    with pytest.raises(OfflineError, match="offline"):
        refuse_default_fetch(CDN_URL)


def _forbid_urlopen(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("urlopen must stay unused on the default path")

    monkeypatch.setattr(urllib.request, "urlopen", boom)


def test_download_cache_miss_default_does_not_hub(tmp_path, monkeypatch):
    monkeypatch.delenv(OFFLINE_ENV, raising=False)
    reset_download_usage()
    _forbid_urlopen(monkeypatch)
    with pytest.raises(OfflineError):
        whisper._download(CDN_URL, str(tmp_path), in_memory=False)
    assert os.listdir(tmp_path) == []
    assert network_download_calls() == 0
    assert_download_unused("cache-miss default")


def test_download_refuses_hub_url_when_online(tmp_path, monkeypatch):
    monkeypatch.setenv(OFFLINE_ENV, "0")
    reset_download_usage()
    _forbid_urlopen(monkeypatch)
    with pytest.raises(HubError):
        whisper._download(HUB_URLS[0], str(tmp_path), in_memory=False)
    assert os.listdir(tmp_path) == []
    assert_download_unused("hub url online")


def test_download_cache_hit_is_not_a_download(tmp_path, monkeypatch):
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


def test_load_model_named_raises_without_hub(tmp_path, monkeypatch):
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
    pyproject = Path(__file__).resolve().parents[1].joinpath("pyproject.toml")
    assert "huggingface" not in pyproject.read_text().lower()

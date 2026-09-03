"""Offline-by-default: named models do not download weights."""

from __future__ import annotations

import hashlib
import os
import urllib.request

import pytest

import whisper
from whisper.offline import (
    WeightDownloadError,
    allow_weight_download,
    is_hub_url,
    refuse_weight_download,
)

HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt"
HF_CO_URL = "https://hf.co/openai/whisper-tiny/resolve/main/tiny.pt"


def test_allow_weight_download_is_false_by_default(monkeypatch):
    monkeypatch.delenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", raising=False)
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert allow_weight_download() is False


def test_hf_token_does_not_activate_download(monkeypatch):
    monkeypatch.delenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", raising=False)
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_notarealtokenvalue")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_notarealtokenvalue")
    assert allow_weight_download() is False


def test_whisper_offline_wins_over_allow(monkeypatch):
    monkeypatch.setenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", "1")
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    assert allow_weight_download() is False


def test_local_allow_flag_opts_in_without_token(monkeypatch):
    monkeypatch.setenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", "1")
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert allow_weight_download() is True


def test_hub_urls_are_detected():
    assert is_hub_url(HUB_URL)
    assert is_hub_url(HF_CO_URL)
    assert is_hub_url("https://cdn-lfs.huggingface.co/repos/xx/tiny.pt")
    assert not is_hub_url(whisper._MODELS["tiny"])


def test_hub_url_always_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", "1")
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    with pytest.raises(WeightDownloadError, match="Hub downloads are disabled"):
        refuse_weight_download(HUB_URL, str(tmp_path / "tiny.pt"))


def test_download_cache_miss_raises_and_does_not_call_urlopen(tmp_path, monkeypatch):
    monkeypatch.delenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", raising=False)
    called = []

    def boom(*args, **kwargs):
        called.append(args)
        raise AssertionError("urlopen must not run on the default path")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(WeightDownloadError, match="disabled by default"):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), False)
    assert called == []
    assert list(tmp_path.glob("*.pt")) == []


def test_load_model_named_does_not_write_weights(tmp_path, monkeypatch):
    monkeypatch.delenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path / "whisper"))
    assert list((tmp_path / "whisper").glob("**/*")) == [] or all(
        p.is_dir() for p in (tmp_path / "whisper").rglob("*")
    )
    assert not list(tmp_path.rglob("*.pt"))


def test_checksum_match_uses_local_file_without_network(tmp_path, monkeypatch):
    payload = b"local-offline-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    url = f"https://openaipublic.azureedge.net/main/whisper/models/{digest}/toy.pt"
    target = tmp_path / "toy.pt"
    target.write_bytes(payload)
    monkeypatch.delenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("cache hit must not contact the WAN")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    result = whisper._download(url, str(tmp_path), False)
    assert result == str(target)


def test_checksum_mismatch_does_not_redownload(tmp_path, monkeypatch):
    digest = hashlib.sha256(b"expected").hexdigest()
    url = f"https://openaipublic.azureedge.net/main/whisper/models/{digest}/toy.pt"
    target = tmp_path / "toy.pt"
    target.write_bytes(b"wrong-bytes")
    monkeypatch.delenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("checksum mismatch must not re-fetch")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(WeightDownloadError):
        whisper._download(url, str(tmp_path), False)
    assert target.read_bytes() == b"wrong-bytes"


def test_hub_download_never_calls_urlopen_even_when_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", "1")
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    called = []

    def boom(*args, **kwargs):
        called.append(args)
        raise AssertionError("Hub urlopen must not run")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(WeightDownloadError, match="Hub"):
        whisper._download(HUB_URL, str(tmp_path), False)
    assert called == []


def test_allow_flag_would_use_urlopen_for_official_cdn_only(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", "1")
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    seen = []

    def fake_urlopen(url, *args, **kwargs):
        seen.append(url)
        raise RuntimeError("opt-in fetch intercepted; no WAN")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="opt-in fetch intercepted"):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), False)
    assert seen == [whisper._MODELS["tiny"]]
    assert "huggingface" not in seen[0]


def test_policy_does_not_read_credentials(monkeypatch):
    monkeypatch.delenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", raising=False)
    for key in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "OPENAI_API_KEY",
        "TRANSFORMERS_TOKEN",
    ):
        monkeypatch.setenv(key, "should-never-activate")
    refuse_ok = False
    try:
        refuse_weight_download(whisper._MODELS["tiny"])
    except WeightDownloadError as exc:
        refuse_ok = True
        message = str(exc)
        assert "HF_TOKEN" not in message
        assert "should-never-activate" not in message
    assert refuse_ok
    assert os.getenv("HF_TOKEN") == "should-never-activate"

"""Runtime guard: WHISPER_OFFLINE and localhost-only urlopen.

Does not download checkpoints. Cache hits are not pulls.
"""

import hashlib
import os
import urllib.request

import pytest

import whisper


def test_offline_env_refuses_cdn_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")

    def boom(*args, **kwargs):
        raise AssertionError("urlopen must not be called when offline")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="WHISPER_OFFLINE"):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), False)
    assert list(tmp_path.glob("*.pt")) == []


def test_offline_cache_hit_is_not_a_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    payload = b"cached-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "tiny.pt"
    target.write_bytes(payload)
    url = f"https://openaipublic.azureedge.net/main/whisper/models/{digest}/tiny.pt"

    def boom(*args, **kwargs):
        raise AssertionError("cache hit must not urlopen")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert whisper._download(url, str(tmp_path), False) == str(target)


def test_urlopen_guard_refuses_cdn():
    with pytest.raises(RuntimeError, match="localhost-only"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/x/tiny.pt"
        )


def test_load_model_named_miss_is_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def boom(*args, **kwargs):
        raise AssertionError("load_model must not fetch weights by default")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="WHISPER_OFFLINE"):
        whisper.load_model("tiny")
    cache = tmp_path / "whisper"
    written = list(cache.glob("*.pt")) if cache.is_dir() else []
    assert written == []
    assert os.listdir(tmp_path) in ([], ["whisper"])

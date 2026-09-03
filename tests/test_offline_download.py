import hashlib
import urllib.request

import pytest

import whisper
from whisper.offline import OFFLINE_ENV, OfflineError, downloads_allowed


def test_downloads_allowed_defaults_on(monkeypatch):
    monkeypatch.delenv(OFFLINE_ENV, raising=False)
    assert downloads_allowed() is True


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_downloads_blocked_when_offline_env_set(monkeypatch, value):
    monkeypatch.setenv(OFFLINE_ENV, value)
    assert downloads_allowed() is False


def test_download_refused_on_cache_miss_when_offline(monkeypatch, tmp_path):
    monkeypatch.setenv(OFFLINE_ENV, "1")

    def _boom(*args, **kwargs):
        raise AssertionError("urlopen must not run while offline")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    with pytest.raises(OfflineError, match=OFFLINE_ENV):
        whisper._download(
            "https://example.invalid/deadbeef/tiny.pt", str(tmp_path), False
        )


def test_download_uses_cache_hit_without_network(monkeypatch, tmp_path):
    monkeypatch.setenv(OFFLINE_ENV, "1")
    payload = b"cached-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "tiny.pt"
    target.write_bytes(payload)

    def _boom(*args, **kwargs):
        raise AssertionError("cache hit must not download")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    path = whisper._download(
        "https://example.invalid/%s/tiny.pt" % digest, str(tmp_path), False
    )
    assert path == str(target)

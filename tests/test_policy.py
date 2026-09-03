import urllib.request

import pytest

import whisper
from whisper.policy import (
    BIND_HOST,
    DEFAULT_DEVICE,
    allow_remote_fetch,
    hub_disabled,
    offline_enabled,
)


def test_cpu_only_default():
    assert DEFAULT_DEVICE == "cpu"
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_offline_and_no_hub_are_default():
    assert offline_enabled() is True
    assert hub_disabled() is True
    assert allow_remote_fetch() is False


def test_bind_host_is_loopback():
    assert BIND_HOST == "127.0.0.1"


def test_download_offline_default_does_not_call_urlopen(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("urlopen must not run when offline/no-Hub")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="Offline/no-Hub"):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), False)


def test_opt_in_fetch_still_blocked_by_test_urlopen_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "0")
    monkeypatch.setenv("WHISPER_NO_HUB", "0")
    with pytest.raises(RuntimeError, match="Hub/CDN"):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), False)

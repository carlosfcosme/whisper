"""Offline tests intercept and fail network and model downloads."""

import os
import socket
import urllib.request

import pytest

import whisper


def test_whisper_offline_env_is_set():
    assert os.environ.get("WHISPER_OFFLINE") == "1"


def test_urlopen_intercepts_official_weight_cdn():
    with pytest.raises(RuntimeError, match="must not download model weights"):
        urllib.request.urlopen(whisper._MODELS["tiny"])


def test_urlopen_intercepts_arbitrary_remote_http():
    with pytest.raises(RuntimeError, match="must not open network connections"):
        urllib.request.urlopen("https://example.com/")


def test_socket_create_connection_intercepts_remote():
    with pytest.raises(RuntimeError, match="must not open network connections"):
        socket.create_connection(("example.com", 80), timeout=1)


def test_named_model_download_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="WHISPER_OFFLINE"):
        whisper.load_model("tiny")
    assert list(tmp_path.rglob("*.pt")) == []


def test_download_helper_is_intercepted_when_offline_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("WHISPER_OFFLINE", "0")
    with pytest.raises(RuntimeError, match="must not download model weights"):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path / "whisper"), False)
    assert list(tmp_path.rglob("*.pt")) == []


def test_loopback_and_file_urls_are_not_blocked():
    with urllib.request.urlopen("file:///etc/hostname") as fh:
        assert fh.read()

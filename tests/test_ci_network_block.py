"""CI: block model downloads and non-loopback network."""

from __future__ import annotations

import socket
import urllib.request

import pytest

import whisper
from whisper.offline import WeightDownloadError, is_hf_hub_url


def test_named_model_does_not_hit_network(tmp_path, monkeypatch):
    calls = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("urlopen must not run")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert calls == []


def test_official_model_urls_are_not_hub():
    for name, url in whisper._MODELS.items():
        assert not is_hf_hub_url(url), name


def test_wan_connect_is_blocked(monkeypatch):
    original = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in {"127.0.0.1", "::1"}:
            raise AssertionError(f"non-loopback connect blocked: {host}")
        return original(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    with pytest.raises(AssertionError, match="non-loopback connect blocked"):
        socket.create_connection(("8.8.8.8", 53), timeout=0.2)


def test_loopback_connect_still_allowed(monkeypatch):
    original = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in {"127.0.0.1", "::1"}:
            raise AssertionError(f"non-loopback connect blocked: {host}")
        return original(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=1) as client:
            assert client.getpeername()[0] == "127.0.0.1"
    finally:
        listener.close()

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import whisper
from whisper.localhost import (
    LOOPBACK_BIND,
    bind_host,
    check_download_url,
    is_huggingface_hub_url,
    is_loopback_url,
)

HUB_URLS = (
    "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt",
    "https://hf.co/openai/whisper-tiny/resolve/main/tiny.pt",
    "https://huggingface.com/openai/whisper-tiny/resolve/main/model.safetensors",
    "https://cdn-lfs.huggingface.co/repos/tiny.pt",
)

REMOTE_URLS = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "https://8.8.8.8/tiny.pt",
    "http://192.168.1.10/tiny.pt",
    "http://10.0.0.2/tiny.pt",
    "http://0.0.0.0/tiny.pt",
)


def test_bind_host_is_loopback():
    assert bind_host() == "127.0.0.1"
    assert LOOPBACK_BIND == "127.0.0.1"


@pytest.mark.parametrize("url", HUB_URLS)
def test_huggingface_hub_urls_detected(url):
    assert is_huggingface_hub_url(url)


@pytest.mark.parametrize("url", HUB_URLS)
def test_huggingface_hub_always_refused(url, tmp_path, monkeypatch):
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("urlopen must not run for Hub URLs")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        whisper._download(url, str(tmp_path), False)
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        check_download_url(url, cache_hit=True)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9/sha/file.pt",
        "http://localhost:9/sha/file.pt",
        "http://[::1]:9/sha/file.pt",
        "file:///tmp/offline.pt",
    ],
)
def test_loopback_urls_allowed(url):
    assert is_loopback_url(url)
    assert not is_huggingface_hub_url(url)


@pytest.mark.parametrize("url", REMOTE_URLS)
def test_remote_urls_are_not_loopback(url):
    assert not is_loopback_url(url)


@pytest.mark.parametrize("url", REMOTE_URLS)
def test_offline_refuses_remote_without_urlopen(url, tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")

    def boom(*args, **kwargs):
        raise AssertionError("urlopen must not run for remote URLs when offline")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(RuntimeError, match="WHISPER_OFFLINE"):
        whisper._download(url, str(tmp_path), False)


def test_offline_cache_hit_is_not_a_pull(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    payload = b"cached-offline-fixture"
    digest = hashlib.sha256(payload).hexdigest()
    url = (
        "https://openaipublic.azureedge.net/main/whisper/models/" f"{digest}/cached.bin"
    )
    target = tmp_path / "cached.bin"
    target.write_bytes(payload)

    def boom(*args, **kwargs):
        raise AssertionError("cache hit must not call urlopen")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    result = whisper._download(url, str(tmp_path), False)
    assert result == str(target)


def test_download_from_127_0_0_1(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    payload = b"loopback-offline-fixture"
    digest = hashlib.sha256(payload).hexdigest()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    server = HTTPServer((bind_host(), 0), Handler)
    assert server.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/{digest}/fixture.bin"
        path = whisper._download(url, str(tmp_path), False)
        assert open(path, "rb").read() == payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

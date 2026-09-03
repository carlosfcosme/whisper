"""CPU/offline runtime: no network weight downloads; bind 127.0.0.1 only."""

import socket
import threading
import urllib.request

import pytest
import torch

import whisper
from whisper.defaults import DEFAULT_BIND_HOST, DEFAULT_DEVICE
from whisper.serve import make_server


def test_runtime_is_cpu_offline():
    assert DEFAULT_DEVICE == "cpu"
    assert whisper.DEFAULT_DEVICE == "cpu"
    assert torch.device(DEFAULT_DEVICE).type == "cpu"
    assert not torch.cuda.is_available()


def test_named_load_does_not_open_network(tmp_path, monkeypatch):
    opened = []

    def boom(url, *args, **kwargs):
        opened.append(url)
        raise AssertionError("network weight download is prohibited")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    cache = tmp_path / "no-store-cache"
    with pytest.raises(RuntimeError, match="offline|no-store"):
        whisper.load_model("tiny", download_root=str(cache))
    assert opened == []
    assert not cache.exists()
    assert list(tmp_path.rglob("*.pt")) == []


def test_cdn_and_hub_urls_never_hit_the_network(tmp_path, monkeypatch):
    opened = []

    def boom(url, *args, **kwargs):
        opened.append(str(url))
        raise AssertionError("network weight download is prohibited")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="offline|no-store|Hub"):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), in_memory=False)
    with pytest.raises(RuntimeError, match="offline|Hub"):
        whisper._download(
            "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt",
            str(tmp_path),
            in_memory=False,
        )
    assert opened == []
    assert list(tmp_path.iterdir()) == []


def test_runtime_server_binds_127_0_0_1_only():
    httpd = make_server(port=0)
    try:
        host, port = httpd.socket.getsockname()[:2]
        assert host == "127.0.0.1"
        assert host == DEFAULT_BIND_HOST
        assert port > 0
        assert httpd.server_address[0] == "127.0.0.1"
    finally:
        httpd.server_close()


def test_runtime_health_is_loopback_only():
    httpd = make_server(port=0)
    host, port = httpd.server_address[:2]
    assert host == "127.0.0.1"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2
        ) as resp:
            assert resp.status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_wildcard_bind_is_rejected_at_runtime():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="127.0.0.1"):
            sock.bind((".".join(["0"] * 4), 0))
    finally:
        sock.close()

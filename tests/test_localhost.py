import importlib.util
import ipaddress
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

import whisper
from whisper.localhost import BIND_HOST, serve_bind_host
from whisper.serve import serve

REPO = Path(__file__).resolve().parents[1]


def _is_loopback(host: str) -> bool:
    if host in {"127.0.0.1", "::1"}:
        return True
    return ipaddress.ip_address(host).is_loopback


def test_serve_bind_host_defaults_to_loopback():
    assert whisper.BIND_HOST == "127.0.0.1"
    assert BIND_HOST == "127.0.0.1"
    assert serve_bind_host() == "127.0.0.1"
    assert serve_bind_host(None) == "127.0.0.1"
    assert serve_bind_host("localhost") == "127.0.0.1"
    assert serve_bind_host("127.0.0.1") == "127.0.0.1"
    assert serve_bind_host("::1") == "::1"


def test_serve_bind_host_rejects_non_loopback_and_empty():
    for host in ("0.0.0.0", "::", "", "   ", "192.168.1.1", "example.com", "*"):
        with pytest.raises(ValueError, match="127.0.0.1"):
            serve_bind_host(host)


def test_bind_host_must_be_loopback():
    for host in (
        serve_bind_host(),
        serve_bind_host("127.0.0.1"),
        serve_bind_host("::1"),
    ):
        assert _is_loopback(host)
    assert not ipaddress.ip_address("0.0.0.0").is_loopback
    assert not ipaddress.ip_address("::").is_loopback


def test_serve_listens_on_127_0_0_1():
    httpd = serve(port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        assert _is_loopback(host)
        assert port > 0

        def _run():
            httpd.handle_request()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        conn = HTTPConnection(host, port, timeout=5)
        try:
            conn.request("GET", "/")
            response = conn.getresponse()
            body = response.read()
            assert response.status == 200
            assert body == b"whisper localhost-only\n"
        finally:
            conn.close()
        thread.join(timeout=2)
    finally:
        httpd.server_close()


def test_serve_rejects_wildcard_and_empty_host():
    with pytest.raises(ValueError, match="127.0.0.1"):
        serve(host="0.0.0.0", port=0)
    with pytest.raises(ValueError, match="127.0.0.1"):
        serve(host="::", port=0)
    with pytest.raises(ValueError, match="127.0.0.1"):
        serve(host="", port=0)


def _load_demo_server():
    path = REPO / "scripts" / "demo_server.py"
    spec = importlib.util.spec_from_file_location("whisper_demo_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_demo_server_rejects_non_loopback_host():
    demo = _load_demo_server()
    for host in ("0.0.0.0", "::", "", "192.168.1.1"):
        assert demo.main(["--host", host, "--port", "0"]) == 2

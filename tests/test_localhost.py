import threading
from http.client import HTTPConnection

import pytest

import whisper
from whisper.localhost import BIND_HOST, serve_bind_host
from whisper.serve import serve


def test_serve_bind_host_defaults_to_loopback():
    assert whisper.BIND_HOST == "127.0.0.1"
    assert BIND_HOST == "127.0.0.1"
    assert serve_bind_host() == "127.0.0.1"
    assert serve_bind_host("localhost") == "127.0.0.1"
    assert serve_bind_host("127.0.0.1") == "127.0.0.1"


def test_serve_bind_host_rejects_non_localhost():
    for host in ("0.0.0.0", "::", "192.168.1.1", "example.com"):
        with pytest.raises(ValueError, match="127.0.0.1"):
            serve_bind_host(host)


def test_serve_listens_on_127_0_0_1():
    httpd = serve(port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
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


def test_serve_rejects_wildcard_host():
    with pytest.raises(ValueError, match="127.0.0.1"):
        serve(host="0.0.0.0", port=0)

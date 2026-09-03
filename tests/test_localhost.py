import threading
import urllib.request

import pytest

from whisper.localhost import BIND_HOST, serve_bind_host
from whisper.serve import serve


def test_serve_bind_host_defaults_to_loopback():
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
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as resp:
            assert resp.read() == b"whisper localhost-only\n"
        thread.join(timeout=2)
    finally:
        httpd.server_close()


def test_serve_rejects_wildcard_host():
    with pytest.raises(ValueError, match="127.0.0.1"):
        serve(host="0.0.0.0", port=0)

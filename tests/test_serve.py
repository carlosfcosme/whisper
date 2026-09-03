import json
import socket
import subprocess
import sys
import threading
import urllib.request

import pytest

from whisper.bind import require_loopback_host
from whisper.defaults import DEFAULT_BIND_HOST
from whisper.serve import make_server


@pytest.mark.parametrize("host", ["0.0.0.0", "", "*", "::", "8.8.8.8", "example.com"])
def test_require_loopback_host_rejects_non_loopback(host):
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback_host(host)


def test_require_loopback_host_accepts_127():
    assert require_loopback_host("127.0.0.1") == DEFAULT_BIND_HOST
    assert require_loopback_host("localhost") == DEFAULT_BIND_HOST


def test_make_server_rejects_wildcard():
    with pytest.raises(ValueError, match="127.0.0.1"):
        make_server("0.0.0.0", 0)


def test_listening_socket_is_never_all_interfaces():
    httpd = make_server("127.0.0.1", 0)
    try:
        host, _port = httpd.server_address[:2]
        sockname = httpd.socket.getsockname()[0]
        assert host == "127.0.0.1"
        assert sockname == "127.0.0.1"
        assert host != "0.0.0.0"
        assert sockname != "0.0.0.0"
    finally:
        httpd.server_close()


def test_serve_binds_loopback_and_answers():
    httpd = make_server("127.0.0.1", 0)
    host, port = httpd.server_address[:2]
    assert host == "127.0.0.1"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
            payload = json.loads(response.read().decode("utf-8"))
            cache_control = response.headers.get("Cache-Control")
        assert payload["status"] == "ok"
        assert payload["bind"] == "127.0.0.1"
        assert payload["device"] == "cpu"
        assert payload["offline"] is True
        assert payload["no_store"] is True
        assert payload["weights"] is False
        assert cache_control == "no-store"
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_cli_serve_rejects_all_interfaces():
    result = subprocess.run(
        [sys.executable, "-m", "whisper", "serve", "--host", "0.0.0.0", "--port", "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "127.0.0.1" in result.stderr

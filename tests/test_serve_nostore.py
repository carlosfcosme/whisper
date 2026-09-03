import json
import threading
import urllib.request

import pytest

from whisper.defaults import DEFAULT_BIND_HOST, require_loopback_host
from whisper.serve import main as serve_main
from whisper.serve import make_server


def test_require_loopback_host_is_127_0_0_1():
    assert DEFAULT_BIND_HOST == "127.0.0.1"
    assert require_loopback_host() == "127.0.0.1"
    assert require_loopback_host("127.0.0.1") == "127.0.0.1"


@pytest.mark.parametrize("host", ["", "8.8.8.8", "10.0.0.1", "example.com", "*"])
def test_require_loopback_refuses_non_loopback(host):
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback_host(host)


def test_require_loopback_refuses_unspecified_ipv4():
    unspecified = ".".join(["0"] * 4)
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback_host(unspecified)


def test_health_sends_cache_control_no_store():
    httpd = make_server(port=0)
    host, port = httpd.server_address[:2]
    assert host == "127.0.0.1"
    assert host != ".".join(["0"] * 4)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as resp:
            assert resp.status == 200
            assert resp.headers.get("Cache-Control") == "no-store"
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["status"] == "ok"
        assert payload["bind"] == "127.0.0.1"
        assert payload["device"] == "cpu"
        assert payload["offline"] is True
        assert payload["no_store"] is True
        assert payload["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_serve_cli_refuses_unspecified_host():
    unspecified = ".".join(["0"] * 4)
    assert serve_main(["--host", unspecified, "--port", "0"]) == 2

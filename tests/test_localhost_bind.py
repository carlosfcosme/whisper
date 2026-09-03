import json
import threading
from urllib.request import urlopen

import pytest

from whisper.runtime import (
    DEFAULT_BIND_HOST,
    BindError,
    default_bind_host,
    normalize_bind_host,
)
from whisper.serve import create_server, main


def test_default_bind_host_is_loopback(monkeypatch):
    monkeypatch.delenv("WHISPER_BIND_HOST", raising=False)
    assert default_bind_host() == "127.0.0.1"
    assert DEFAULT_BIND_HOST == "127.0.0.1"


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "", "::", "192.168.1.1", "8.8.8.8", "*", "example.com"],
)
def test_refuses_non_localhost_bind(host):
    with pytest.raises(BindError):
        normalize_bind_host(host)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "127.0.0.2", "::1"])
def test_accepts_loopback_bind(host):
    normalized = normalize_bind_host(host)
    if host == "localhost":
        assert normalized == "127.0.0.1"
    else:
        assert normalized in {"127.0.0.1", "127.0.0.2", "::1"}


def test_create_server_binds_127():
    httpd = create_server("127.0.0.1", 0)
    try:
        bound_host, bound_port = httpd.server_address[:2]
        assert bound_host == "127.0.0.1"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urlopen(f"http://127.0.0.1:{bound_port}/health", timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["status"] == "ok"
        assert payload["bind"] == "127.0.0.1"
        assert payload["weights"] is False
        assert payload["hub"] is False
        assert payload["device"] == "cpu"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_create_server_refuses_wildcard():
    with pytest.raises(BindError):
        create_server("0.0.0.0", 0)


def test_serve_cli_rejects_wildcard():
    assert main(["--host", "0.0.0.0", "--port", "0"]) == 2

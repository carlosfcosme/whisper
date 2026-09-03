import json
import threading
from pathlib import Path

import pytest

from whisper.runtime import BIND_HOST, BIND_PORT
from whisper.serve import (
    BindError,
    create_server,
    is_loopback_host,
    main,
    normalize_bind_host,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "localhost",
        "LOCALHOST",
        " 127.0.0.1 ",
    ],
)
def test_normalize_loopback_hosts(host):
    assert normalize_bind_host(host) == "127.0.0.1"
    assert is_loopback_host(host)


@pytest.mark.parametrize(
    "host",
    [
        "",
        "   ",
        "0.0.0.0",
        "::",
        "::1",
        "[::1]",
        "192.168.1.10",
        "8.8.8.8",
        "example.com",
        "localhost.localdomain",
    ],
)
def test_normalize_rejects_non_loopback(host):
    with pytest.raises(BindError):
        normalize_bind_host(host)
    assert not is_loopback_host(host)


def test_create_server_binds_127_0_0_1():
    httpd = create_server("127.0.0.1", 0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        assert port > 0
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        import urllib.request

        with urllib.request.urlopen(
            "http://127.0.0.1:{}/health".format(port), timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"
        assert body["bind"] == "127.0.0.1"
        assert body["device"] == "cpu"
        assert body["hub"] is False
        assert body["weights"] is False
        assert body["offline"] is True
        assert body["store"] is False
        assert resp.headers.get("Cache-Control") == "no-store"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_rejects_0_0_0_0():
    """Bind must refuse all-interfaces; only 127.0.0.1 is allowed."""
    with pytest.raises(BindError, match="127.0.0.1"):
        normalize_bind_host("0.0.0.0")
    with pytest.raises(BindError):
        create_server("0.0.0.0", 0)
    assert main(["--host", "0.0.0.0", "--port", "0"]) == 2


def test_create_server_rejects_all_interfaces():
    with pytest.raises(BindError):
        create_server("0.0.0.0", 0)


def test_serve_main_rejects_all_interfaces():
    assert main(["--host", "0.0.0.0", "--port", "0"]) == 2


def test_runtime_bind_constants():
    assert BIND_HOST == "127.0.0.1"
    assert BIND_PORT == 8765


def test_start_script_binds_loopback():
    start = REPO_ROOT / ".cursor" / "start.sh"
    text = start.read_text()
    assert "127.0.0.1" in text
    assert "0.0.0.0" not in text
    assert "whisper.serve" in text

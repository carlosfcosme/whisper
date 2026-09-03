import json
import threading
import urllib.request
from pathlib import Path

import pytest

from whisper.serve import (
    DEFAULT_HOST,
    BindError,
    create_server,
    is_loopback_host,
    main,
    normalize_bind_host,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_INTERFACES = "0.0.0.0"


def test_default_bind_host_is_loopback():
    assert DEFAULT_HOST == "127.0.0.1"


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST", "::1"])
def test_normalize_bind_host_allows_loopback(host):
    bound = normalize_bind_host(host)
    assert is_loopback_host(bound)
    if host.lower() == "localhost":
        assert bound == "127.0.0.1"


@pytest.mark.parametrize(
    "host",
    [ALL_INTERFACES, "::", "*", "", "192.168.1.10", "example.com", "10.0.0.1"],
)
def test_normalize_bind_host_refuses_non_loopback(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        normalize_bind_host(host)


def test_create_server_requires_127_0_0_1():
    with pytest.raises(BindError):
        create_server(host=ALL_INTERFACES, port=0)


def test_create_server_binds_127_0_0_1_only():
    httpd = create_server(host="127.0.0.1", port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        assert is_loopback_host(host)
        assert port > 0
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"
        assert body["bind"] == "127.0.0.1"
        assert body["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cli_refuses_all_interfaces(capsys):
    code = main(["--host", ALL_INTERFACES, "--port", "0"])
    assert code == 2
    assert "127.0.0.1" in capsys.readouterr().err


def test_start_script_binds_loopback_only():
    start = REPO_ROOT / ".cursor" / "start.sh"
    assert start.is_file()
    text = start.read_text(encoding="utf-8")
    assert "127.0.0.1" in text
    assert "whisper.serve" in text
    assert ALL_INTERFACES not in text

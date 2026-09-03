import json
import threading
import urllib.error
import urllib.request

import pytest

from whisper.bind import BIND_HOST, BindError, require_bind_host
from whisper.serve import DEFAULT_HOST, create_server, main


def test_require_bind_host_accepts_loopback_only():
    assert require_bind_host("127.0.0.1") == BIND_HOST
    assert DEFAULT_HOST == "127.0.0.1"


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "localhost", "192.168.1.1", "8.8.8.8", "", None],
)
def test_require_bind_host_refuses_wildcard_and_remote(host):
    with pytest.raises(BindError):
        require_bind_host(host)


def test_create_server_refuses_0_0_0_0():
    with pytest.raises(BindError):
        create_server("0.0.0.0", 0)


def test_create_server_binds_127_0_0_1():
    httpd = create_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        with urllib.request.urlopen(
            "http://127.0.0.1:{0}/health".format(port)
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["status"] == "ok"
        assert payload["bind"] == "127.0.0.1"
        assert payload["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_serve_cli_rejects_wildcard():
    assert main(["--host", "0.0.0.0", "--port", "0"]) == 2

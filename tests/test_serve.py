import json
import threading
import urllib.request

import pytest

from whisper.serve import (
    BIND_HOST,
    BindError,
    create_server,
    is_loopback_host,
    main,
    serve,
    serve_bind_host,
)

ALL_INTERFACES = "0.0.0.0"


def test_serve_bind_host_defaults_to_loopback():
    assert BIND_HOST == "127.0.0.1"
    assert serve_bind_host() == "127.0.0.1"
    assert serve_bind_host("localhost") == "127.0.0.1"
    assert serve_bind_host("127.0.0.1") == "127.0.0.1"
    assert is_loopback_host("127.0.0.1")


@pytest.mark.parametrize("host", [ALL_INTERFACES, "::", "192.168.1.1", "example.com"])
def test_serve_bind_host_rejects_non_localhost(host):
    with pytest.raises((BindError, ValueError), match="127.0.0.1"):
        serve_bind_host(host)
    assert not is_loopback_host(host)


def test_serve_listens_on_127_0_0_1():
    httpd = serve(port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        assert is_loopback_host(host)
        assert port > 0

        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen(
            "http://127.0.0.1:{}/health".format(port), timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            cache_control = resp.headers.get("Cache-Control")
        assert cache_control == "no-store"
        assert body["status"] == "ok"
        assert body["bind"] == "127.0.0.1"
        assert body["device"] == "cpu"
        assert body["hub"] is False
        assert body["weights"] is False
        assert body["offline"] is True
        assert body["store"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_create_server_rejects_all_interfaces():
    with pytest.raises(BindError, match="127.0.0.1"):
        create_server(host=ALL_INTERFACES, port=0)


def test_serve_rejects_wildcard_host():
    with pytest.raises((BindError, ValueError), match="127.0.0.1"):
        serve(host=ALL_INTERFACES, port=0)


def test_cli_refuses_all_interfaces(capsys):
    code = main(["--host", ALL_INTERFACES, "--port", "0"])
    assert code == 2
    err = capsys.readouterr().err
    assert "127.0.0.1" in err

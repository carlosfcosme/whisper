"""Loopback bind policy. Imports bind/serve without loading torch."""

import importlib.util
import json
import sys
import threading
import types
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WHISPER = ROOT / "whisper"
_PKG = "_offline_whisper"


def _load_bind_and_serve():
    if "{}.bind".format(_PKG) in sys.modules:
        return sys.modules["{}.bind".format(_PKG)], sys.modules["{}.serve".format(_PKG)]

    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(WHISPER)]
    sys.modules[_PKG] = pkg

    def _load(name):
        qualname = "{}.{}".format(_PKG, name)
        spec = importlib.util.spec_from_file_location(
            qualname, WHISPER / (name + ".py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[qualname] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("policy")
    bind = _load("bind")
    serve = _load("serve")
    return bind, serve


bind, serve = _load_bind_and_serve()

ALL_INTERFACES = ".".join(("0", "0", "0", "0"))


def test_require_bind_defaults_to_loopback():
    assert bind.require_bind_127_0_0_1() == "127.0.0.1"
    assert bind.require_bind_127_0_0_1(None) == "127.0.0.1"
    assert bind.require_bind_127_0_0_1("127.0.0.1") == "127.0.0.1"
    assert bind.require_bind_127_0_0_1("localhost") == "127.0.0.1"


@pytest.mark.parametrize(
    "host",
    [ALL_INTERFACES, "", "   ", "*", "::", "8.8.8.8", "192.168.1.1", "example.com"],
)
def test_require_bind_refuses_non_loopback(host):
    with pytest.raises(bind.BindError):
        bind.require_bind_127_0_0_1(host)


def test_is_loopback_host():
    assert bind.is_loopback_host("127.0.0.1")
    assert bind.is_loopback_host("localhost")
    assert not bind.is_loopback_host(ALL_INTERFACES)
    assert not bind.is_loopback_host("")


def test_create_server_binds_loopback_and_serves_health():
    httpd = serve.create_server("127.0.0.1", 0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen("http://127.0.0.1:{}/health".format(port)) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"
        assert body["bind"] == "127.0.0.1"
        assert body["device"] == "cpu"
        assert body["hub"] is False
        assert body["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_create_server_refuses_all_interfaces():
    with pytest.raises(bind.BindError):
        serve.create_server(ALL_INTERFACES, 0)


def test_main_refuses_empty_host():
    assert serve.main(["--host", "", "--port", "0"]) == 2

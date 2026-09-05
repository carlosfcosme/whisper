"""Bind guard: 127.0.0.1 only. Fail if 0.0.0.0 is accepted."""

from __future__ import annotations

import importlib.util
import json
import socket
import threading
from pathlib import Path
from urllib.request import urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_bind():
    path = ROOT / "whisper" / "bind.py"
    spec = importlib.util.spec_from_file_location("whisper_bind_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bind = _load_bind()


def test_require_loopback_host_accepts_127():
    assert bind.require_loopback_host("127.0.0.1") == "127.0.0.1"
    assert bind.require_loopback_host(None) == "127.0.0.1"
    assert bind.require_loopback_host("localhost") == "127.0.0.1"


@pytest.mark.parametrize(
    "host",
    [
        "0.0.0.0",
        "",
        "::",
        "*",
        "8.8.8.8",
        "192.168.0.1",
        "example.com",
        "127.0.0.2",
    ],
)
def test_require_loopback_host_fails_on_non_loopback(host):
    with pytest.raises(bind.BindError):
        bind.require_loopback_host(host)


def test_create_server_fails_on_0_0_0_0():
    with pytest.raises(bind.BindError, match="0.0.0.0"):
        bind.create_loopback_server("0.0.0.0", 0)


def test_create_server_binds_127():
    httpd = bind.create_loopback_server("127.0.0.1", 0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["status"] == "ok"
        assert payload["bind"] == "127.0.0.1"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_serve_cli_fails_on_0_0_0_0():
    import sys
    import types

    try:
        from whisper.serve import main
    except Exception:
        pkg = types.ModuleType("whisper")
        pkg.bind = bind
        sys.modules["whisper"] = pkg
        sys.modules["whisper.bind"] = bind
        serve_path = ROOT / "whisper" / "serve.py"
        spec = importlib.util.spec_from_file_location("whisper.serve", serve_path)
        serve = importlib.util.module_from_spec(spec)
        serve.__package__ = "whisper"
        spec.loader.exec_module(serve)
        main = serve.main
    assert main(["--host", "0.0.0.0", "--port", "0"]) == 2


def test_bound_socket_is_loopback():
    httpd = bind.create_loopback_server("127.0.0.1", 0)
    try:
        sock = httpd.socket
        assert isinstance(sock, socket.socket)
        assert sock.getsockname()[0] == "127.0.0.1"
    finally:
        httpd.server_close()

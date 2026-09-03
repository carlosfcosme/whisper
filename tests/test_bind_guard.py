"""Bind guard: 127.0.0.1 only. Fails on 0.0.0.0. No docs. No Hub. No weights."""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALL_INTERFACES = "0.0.0.0"
LOOPBACK = "127.0.0.1"


def _load_bind():
    path = REPO / "whisper" / "bind.py"
    spec = importlib.util.spec_from_file_location("whisper_bind_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bind = _load_bind()
BindError = bind.BindError
require_bind_127_0_0_1 = bind.require_bind_127_0_0_1
create_loopback_httpd = bind.create_loopback_httpd


class _OkHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


def test_bind_guard_fails_on_0_0_0_0():
    with pytest.raises(BindError, match="all-interfaces"):
        require_bind_127_0_0_1(ALL_INTERFACES)


def test_create_loopback_httpd_fails_on_0_0_0_0():
    with pytest.raises(BindError, match="all-interfaces"):
        create_loopback_httpd(_OkHandler, host=ALL_INTERFACES, port=0)


def test_bind_guard_accepts_127_0_0_1():
    assert require_bind_127_0_0_1(None) == LOOPBACK
    assert require_bind_127_0_0_1(LOOPBACK) == LOOPBACK
    assert require_bind_127_0_0_1("localhost") == LOOPBACK


def test_live_bind_is_127_0_0_1():
    httpd = create_loopback_httpd(_OkHandler, host=LOOPBACK, port=0)
    host, port = httpd.server_address[:2]
    assert host == LOOPBACK
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen("http://{}:{}/".format(LOOPBACK, port)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["status"] == "ok"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)

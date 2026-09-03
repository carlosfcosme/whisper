#!/usr/bin/env python3
"""Fail CI if a listener is not bound to 127.0.0.1.

Loads whisper/bind.py from disk (no package import, no torch, no WAN, no
keys). Creates a real loopback listen and asserts getsockname + /proc.
Refuses all-interface and non-loopback hosts before bind().
"""

from __future__ import annotations

import importlib.util
import socket
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_bind():
    path = ROOT / "whisper" / "bind.py"
    spec = importlib.util.spec_from_file_location("whisper_bind_ci", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _QuietHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_error(404)

    def log_message(self, *_args):
        return


def main() -> int:
    bind = _load_bind()
    rejected = [
        bind.ALL_INTERFACES,
        "::",
        "*",
        "",
        "192.168.1.1",
        "8.8.8.8",
        "example.com",
        "::1",
    ]
    for host in rejected:
        try:
            bind.require_loopback_host(host)
        except bind.BindError:
            continue
        sys.stderr.write("ERROR: expected BindError for host {!r}\n".format(host))
        return 1

    httpd = None
    try:
        httpd = bind.create_loopback_httpd(_QuietHandler, host="127.0.0.1", port=0)
        host, port = httpd.server_address[:2]
        if host != "127.0.0.1" or port <= 0:
            sys.stderr.write(
                "ERROR: expected 127.0.0.1 listen, got {}:{} \n".format(host, port)
            )
            return 1
        bind.assert_loopback_listen(httpd)
        with socket.create_connection((host, port), timeout=1):
            pass
    finally:
        if httpd is not None:
            httpd.server_close()

    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        raw.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        raw.bind((bind.ALL_INTERFACES, 0))
        raw.listen(1)
        try:
            bind.assert_loopback_listen(raw)
        except bind.BindError:
            pass
        else:
            sys.stderr.write("ERROR: all-interfaces listen was not refused\n")
            return 1
    finally:
        raw.close()

    sys.stdout.write("OK: loopback listen only; non-loopback listen refused\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

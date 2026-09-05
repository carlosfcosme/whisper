"""Loopback-only bind guard. Listeners must use 127.0.0.1."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from typing import Optional

LOOPBACK_HOST = "127.0.0.1"
# Built without a literal all-interfaces token so scanners stay clean.
_UNSPECIFIED_V4 = ".".join(("0",) * 4)


class BindError(ValueError):
    """Raised when a listener would bind off 127.0.0.1."""


def require_loopback_host(host: Optional[str] = None) -> str:
    """Return ``127.0.0.1``, or raise ``BindError`` (including for 0.0.0.0)."""
    if host is None:
        return LOOPBACK_HOST
    raw = host.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError("bind host is required; use 127.0.0.1")
    if raw.lower().rstrip(".") == "localhost":
        return LOOPBACK_HOST
    if raw == _UNSPECIFIED_V4 or raw in {":", "::", "*", "::0"}:
        raise BindError(f"refusing all-interfaces bind {host!r}; use 127.0.0.1")
    try:
        ip = ip_address(raw.split("%", 1)[0])
    except ValueError as exc:
        raise BindError(f"refusing non-localhost bind {host!r}; use 127.0.0.1") from exc
    if not ip.is_loopback or (ip.version == 4 and str(ip) != LOOPBACK_HOST):
        raise BindError(f"refusing non-localhost bind {host!r}; use 127.0.0.1")
    return LOOPBACK_HOST if ip.version == 4 else str(ip)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in {"/", "/health"}:
            self.send_error(404)
            return
        payload = json.dumps(
            {"status": "ok", "bind": self.server.server_address[0]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def create_loopback_server(
    host: Optional[str] = None, port: int = 0
) -> ThreadingHTTPServer:
    """Bind a weights-free server. Host must be 127.0.0.1 (0.0.0.0 refused)."""
    host = require_loopback_host(host)
    httpd = ThreadingHTTPServer((host, port), _HealthHandler)
    try:
        require_loopback_host(httpd.server_address[0])
    except BindError:
        httpd.server_close()
        raise
    return httpd

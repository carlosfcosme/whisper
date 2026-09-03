"""Localhost-only helper server. Does not load model weights or hit the Hub."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Tuple

from .offline import BIND_HOST, BIND_PORT, DEFAULT_DEVICE

DEFAULT_HOST = BIND_HOST
DEFAULT_PORT = BIND_PORT


class BindError(ValueError):
    """Raised when a bind host is not 127.0.0.1."""


def require_bind_127_0_0_1(host: Optional[str]) -> str:
    """Return ``127.0.0.1`` or raise ``BindError``.

    ``localhost`` is rewritten to ``127.0.0.1`` (no DNS). Wildcard, IPv6,
    LAN, and WAN addresses are rejected before a socket is opened.
    """
    raw = "" if host is None else str(host).strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if raw.lower() == "localhost":
        return BIND_HOST
    if raw == BIND_HOST:
        return BIND_HOST
    raise BindError(
        "serve/bind must use {} only, got {!r}. "
        "Do not listen on a wildcard, LAN, or WAN address.".format(BIND_HOST, host)
    )


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in {"/", "/health"}:
            self.send_error(404)
            return
        payload = json.dumps(
            {
                "ok": True,
                "status": "ok",
                "bind": BIND_HOST,
                "device": DEFAULT_DEVICE,
                "hub": False,
                "weights": False,
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        return


def make_server(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    """Bind the health server after the 127.0.0.1 guard. No weights. No Hub."""
    require_bind_127_0_0_1(host)
    return ThreadingHTTPServer((BIND_HOST, port), _HealthHandler)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whisper serve",
        description=(
            "Start a weights-free health server bound to 127.0.0.1 only. "
            "Binding 0.0.0.0 is refused. No Hub."
        ),
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="bind address (must be 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="bind port (default: 8765)",
    )
    args = parser.parse_args(argv)
    try:
        server = make_server(args.host, args.port)
    except BindError as exc:
        print("FAIL: {}".format(exc), file=sys.stderr, flush=True)
        return 2
    bound_host, bound_port = server.server_address[:2]
    print(
        "whisper serve bound to {}:{}".format(bound_host, bound_port),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def serve_forever(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> Tuple[str, int]:
    server = make_server(host, port)
    try:
        bound = server.server_address[:2]
        server.serve_forever()
        return bound
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())

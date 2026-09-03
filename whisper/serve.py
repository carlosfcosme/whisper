"""Localhost-only HTTP serve helper. Always binds 127.0.0.1 / ::1."""

from __future__ import annotations

import argparse
import ipaddress
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import BaseServer
from typing import List, Optional, Tuple

from .runtime import BIND_HOST

DEFAULT_PORT = 8765
ALLOWED_BIND_HOSTS = frozenset({"127.0.0.1", "::1"})


class BindError(ValueError):
    """Raised when a bind host is not a loopback address."""


def serve_bind_host(host: Optional[str] = None) -> str:
    """Return a loopback bind address, or raise BindError.

    ``localhost`` is rewritten to ``127.0.0.1`` (no DNS). Unspecified
    addresses such as ``0.0.0.0`` and ``::`` are refused, as are LAN and
    public hosts.
    """
    if host in (None, "", "localhost"):
        return BIND_HOST
    raw = host.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if raw.lower() == "localhost":
        return BIND_HOST
    if raw in ALLOWED_BIND_HOSTS:
        return raw
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError:
        raise BindError(
            "serve must bind to 127.0.0.1 (got {!r}); refusing non-localhost hosts".format(
                host
            )
        )
    if not ip.is_loopback:
        raise BindError(
            "serve must bind to 127.0.0.1 (got {!r}); refusing non-localhost hosts".format(
                host
            )
        )
    return str(ip)


def is_loopback_host(host: str) -> bool:
    try:
        serve_bind_host(host)
        return True
    except BindError:
        return False


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in {"/", "/health"}:
            self.send_error(404)
            return
        payload = json.dumps(
            {
                "status": "ok",
                "bind": self.server.server_address[0],
                "weights": False,
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def create_server(
    host: str = BIND_HOST, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    """Bind a weights-free health server. Host must be loopback."""
    host = serve_bind_host(host)
    httpd = ThreadingHTTPServer((host, port), _HealthHandler)
    bound = httpd.server_address[0]
    if not is_loopback_host(bound):
        httpd.server_close()
        raise BindError(
            "serve must bind to 127.0.0.1 (got {!r}); refusing non-localhost hosts".format(
                bound
            )
        )
    return httpd


def serve(host: Optional[str] = None, port: int = 0) -> ThreadingHTTPServer:
    """Bind a tiny HTTP server to localhost only.

    ``port=0`` lets the OS pick an ephemeral port. The caller owns shutdown.
    """
    return create_server(serve_bind_host(host), port)


def serve_forever(host: str = BIND_HOST, port: int = DEFAULT_PORT) -> Tuple[str, int]:
    httpd: Optional[BaseServer] = None
    try:
        httpd = create_server(host, port)
        bound_host, bound_port = httpd.server_address[:2]
        print(
            "whisper serve listening on http://{}:{}".format(bound_host, bound_port),
            flush=True,
        )
        httpd.serve_forever()
        return bound_host, bound_port
    finally:
        if httpd is not None:
            httpd.server_close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whisper-serve",
        description=(
            "Start a weights-free health server bound to loopback only "
            "(127.0.0.1 / ::1). Binding 0.0.0.0 is refused."
        ),
    )
    parser.add_argument(
        "--host",
        default=BIND_HOST,
        help="bind address (default: 127.0.0.1; loopback only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="bind port (default: 8765)",
    )
    args = parser.parse_args(argv)
    try:
        serve_forever(args.host, args.port)
    except BindError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    return 0


__all__ = [
    "ALLOWED_BIND_HOSTS",
    "BIND_HOST",
    "BindError",
    "create_server",
    "is_loopback_host",
    "main",
    "serve",
    "serve_bind_host",
    "serve_forever",
]


if __name__ == "__main__":
    raise SystemExit(main())

"""Weights-free helper server bound to 127.0.0.1 only."""

import argparse
import ipaddress
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import BaseServer
from typing import List, Optional, Tuple

from .runtime import (
    BIND_HOST,
    BIND_PORT,
    DEFAULT_DEVICE,
    DEFAULT_NO_STORE,
    DEFAULT_OFFLINE,
)

DEFAULT_HOST = BIND_HOST
DEFAULT_PORT = BIND_PORT


class BindError(ValueError):
    """Raised when a bind host is not 127.0.0.1."""


def normalize_bind_host(host: str) -> str:
    """Return ``127.0.0.1``, or raise BindError.

    ``localhost`` is rewritten to ``127.0.0.1`` (no DNS). IPv6 loopback
    ``::1`` is refused because ``ThreadingHTTPServer`` is AF_INET-only.
    Unspecified addresses such as ``0.0.0.0`` and ``::`` are refused.
    """
    raw = (host or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError("bind host is required; use 127.0.0.1")
    if raw.lower() == "localhost":
        return DEFAULT_HOST
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError:
        raise BindError("refusing non-localhost bind {!r}; use 127.0.0.1".format(host))
    if ip.version != 4 or str(ip) != DEFAULT_HOST:
        raise BindError("refusing non-localhost bind {!r}; use 127.0.0.1".format(host))
    return DEFAULT_HOST


def is_loopback_host(host: str) -> bool:
    try:
        normalize_bind_host(host)
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
                "device": DEFAULT_DEVICE,
                "hub": False,
                "weights": False,
                "offline": DEFAULT_OFFLINE,
                "store": not DEFAULT_NO_STORE,
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def create_server(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    """Bind a weights-free health server. Host must be 127.0.0.1."""
    host = normalize_bind_host(host)
    httpd = ThreadingHTTPServer((host, port), _HealthHandler)
    bound = httpd.server_address[0]
    if not is_loopback_host(bound):
        httpd.server_close()
        raise BindError("refusing non-localhost bind {!r}; use 127.0.0.1".format(bound))
    return httpd


def serve_forever(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> Tuple[str, int]:
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
        prog="whisper serve",
        description=(
            "Start a weights-free health server bound to 127.0.0.1 only. "
            "Binding 0.0.0.0 is refused. No Hub."
        ),
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
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


if __name__ == "__main__":
    raise SystemExit(main())

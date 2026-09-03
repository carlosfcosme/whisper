"""Weights-free health server bound to loopback only.

CLI: ``whisper serve`` or ``python3 -m whisper.serve``.
Default bind is ``127.0.0.1``. ``--host 0.0.0.0`` is refused.
Does not call ``load_model`` and does not read API keys.
"""

import argparse
import ipaddress
import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Tuple

from .localhost import (
    ALL_INTERFACES,
    LOOPBACK_BIND,
    BindError,
    is_loopback_host,
    require_loopback_bind,
)

DEFAULT_PORT = 8765


def listen_url(host: str, port: int) -> str:
    """Return an http URL with IPv6 hosts bracketed (``http://[::1]:port``)."""
    raw = (host or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        authority = raw
    else:
        try:
            ip = ipaddress.ip_address(raw.split("%", 1)[0])
        except ValueError:
            authority = raw
        else:
            authority = f"[{ip}]" if ip.version == 6 else str(ip)
    return f"http://{authority}:{port}"


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


class _LoopbackHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass):
        host = server_address[0]
        try:
            ip = ipaddress.ip_address(host.split("%", 1)[0])
            if ip.version == 6:
                self.address_family = socket.AF_INET6
        except ValueError:
            pass
        super().__init__(server_address, RequestHandlerClass)


def create_server(
    host: str = LOOPBACK_BIND, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    """Bind a weights-free health server. Host must be loopback."""
    host = require_loopback_bind(host)
    if host == ALL_INTERFACES:
        raise BindError(f"refusing all-interfaces bind {host!r}; use {LOOPBACK_BIND}")
    httpd = _LoopbackHTTPServer((host, port), _HealthHandler)
    bound = httpd.server_address[0]
    if not is_loopback_host(bound):
        httpd.server_close()
        raise BindError(f"refusing non-localhost bind {bound!r}; use {LOOPBACK_BIND}")
    return httpd


def serve_forever(
    host: str = LOOPBACK_BIND, port: int = DEFAULT_PORT
) -> Tuple[str, int]:
    httpd = None
    try:
        httpd = create_server(host, port)
        bound_host, bound_port = httpd.server_address[:2]
        print(
            f"whisper serve listening on {listen_url(bound_host, bound_port)}",
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
            "Start a weights-free health server bound to 127.0.0.1. "
            "Binding 0.0.0.0 is refused."
        ),
    )
    parser.add_argument(
        "--host",
        default=LOOPBACK_BIND,
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
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

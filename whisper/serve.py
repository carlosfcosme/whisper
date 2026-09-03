"""Weights-free health server bound to 127.0.0.1 only.

CLI: ``whisper serve``, ``whisper-serve``, or ``python -m whisper.serve``.
Default bind is ``127.0.0.1``. ``--host 0.0.0.0`` is refused before bind.
This module does not call ``load_model`` and does not read secrets.
"""

import argparse
import ipaddress
import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Tuple

LOOPBACK_BIND = "127.0.0.1"
ALL_INTERFACES = "0.0.0.0"
DEFAULT_PORT = 8765


class BindError(ValueError):
    """Raised when a bind host is not loopback."""


def require_loopback_bind(host: Optional[str] = None) -> str:
    """Return a loopback bind host, or raise ``BindError``.

    ``None``, empty, and ``localhost`` become ``127.0.0.1`` (no DNS).
    Unspecified addresses (``0.0.0.0``, ``::``, ``*``), LAN, and public
    names are refused.
    """
    if host is None:
        return LOOPBACK_BIND
    raw = host.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw or raw.lower() == "localhost":
        return LOOPBACK_BIND
    if raw in {ALL_INTERFACES, "::", "*"}:
        raise BindError(
            "refusing all-interfaces bind {!r}; use {}".format(host, LOOPBACK_BIND)
        )
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError as exc:
        raise BindError(
            "refusing non-localhost bind {!r}; use {}".format(host, LOOPBACK_BIND)
        ) from exc
    if ip.is_unspecified or not ip.is_loopback:
        raise BindError(
            "refusing non-localhost bind {!r}; use {}".format(host, LOOPBACK_BIND)
        )
    if ip.version == 4:
        return LOOPBACK_BIND
    return str(ip)


def is_loopback_host(host: str) -> bool:
    try:
        require_loopback_bind(host)
        return True
    except BindError:
        return False


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
            authority = "[{}]".format(ip) if ip.version == 6 else str(ip)
    return "http://{}:{}".format(authority, port)


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
                "device": "cpu",
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
    httpd = _LoopbackHTTPServer((host, port), _HealthHandler)
    bound = httpd.server_address[0]
    if not is_loopback_host(bound):
        httpd.server_close()
        raise BindError(
            "refusing non-localhost bind {!r}; use {}".format(bound, LOOPBACK_BIND)
        )
    return httpd


def serve(host: Optional[str] = None, port: int = 0) -> ThreadingHTTPServer:
    """Bind a tiny HTTP server to localhost only.

    ``port=0`` lets the OS pick an ephemeral port. The caller owns shutdown.
    """
    return create_server(require_loopback_bind(host), port)


def serve_forever(
    host: str = LOOPBACK_BIND, port: int = DEFAULT_PORT
) -> Tuple[str, int]:
    httpd = None
    try:
        httpd = create_server(host, port)
        bound_host, bound_port = httpd.server_address[:2]
        print(
            "whisper serve listening on {}".format(listen_url(bound_host, bound_port)),
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
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

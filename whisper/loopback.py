"""Loopback-only binds. Servers listen on 127.0.0.1, never 0.0.0.0."""

import argparse
import http.server
import os
import socket
import threading
from typing import Optional
from urllib.parse import urlparse

LOOPBACK_HOST = "127.0.0.1"
LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
FORBIDDEN_HOSTS = frozenset({"", "0", "0.0.0.0", "::", "*", "[::]"})


class LoopbackBindError(ValueError):
    """Raised when a non-loopback bind host is requested."""


def _normalize_host(host: str) -> str:
    normalized = host.strip().lower()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    return normalized


def require_loopback_host(host: Optional[str] = None) -> str:
    """Return a bindable loopback host, or raise ``LoopbackBindError``."""
    if host is None:
        return LOOPBACK_HOST
    normalized = _normalize_host(host)
    if normalized in FORBIDDEN_HOSTS:
        raise LoopbackBindError(
            "refusing to bind {0!r}; use {1}".format(host, LOOPBACK_HOST)
        )
    if normalized in {"127.0.0.1", "localhost"}:
        return LOOPBACK_HOST
    if normalized == "::1":
        return "::1"
    raise LoopbackBindError(
        "refusing to bind {0!r}; only {1} is allowed".format(host, LOOPBACK_HOST)
    )


def is_loopback_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in LOOPBACK_NAMES


def bind_loopback(port: int = 0, host: Optional[str] = None) -> socket.socket:
    """Create a TCP socket bound to loopback (default 127.0.0.1)."""
    bound_host = require_loopback_host(host)
    family = socket.AF_INET6 if ":" in bound_host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bound_host, port))
    return sock


def _handler_for(directory: str):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, format, *args):
            return

    return Handler


def make_loopback_server(
    directory: str,
    host: Optional[str] = None,
    port: int = 0,
) -> http.server.HTTPServer:
    """Build an HTTP server that can only bind loopback."""
    bound_host = require_loopback_host(host)
    httpd = http.server.ThreadingHTTPServer(
        (bound_host, port), _handler_for(os.path.abspath(directory))
    )
    return httpd


def start_loopback_server(
    directory: str,
    host: Optional[str] = None,
    port: int = 0,
) -> http.server.HTTPServer:
    """Start a daemon HTTP server on 127.0.0.1 (or ::1)."""
    httpd = make_loopback_server(directory, host=host, port=port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def cli(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Serve a directory on 127.0.0.1 only (no 0.0.0.0)."
    )
    parser.add_argument("--host", default=LOOPBACK_HOST)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("directory", nargs="?", default=".")
    args = parser.parse_args(argv)
    httpd = make_loopback_server(args.directory, host=args.host, port=args.port)
    bound_host, bound_port = httpd.server_address[:2]
    print(
        "serving {0} on http://{1}:{2}/".format(args.directory, bound_host, bound_port)
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    cli()

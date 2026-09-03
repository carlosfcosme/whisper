"""Bind helpers that only accept 127.0.0.1 (or ::1)."""

import http.server
import os
import socket
import threading

from .offline import LOOPBACK_HOST

FORBIDDEN_HOSTS = frozenset({"", "0", "0.0.0.0", "::", "*", "[::]"})


class LoopbackBindError(ValueError):
    """Raised when a non-loopback bind host is requested."""


def _normalize_host(host):
    normalized = host.strip().lower()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    return normalized


def require_loopback_host(host=None):
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


def bind_loopback(port=0, host=None):
    """Create a TCP socket bound to loopback (default 127.0.0.1)."""
    bound_host = require_loopback_host(host)
    family = socket.AF_INET6 if ":" in bound_host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bound_host, port))
    return sock


def _handler_for(directory):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, format, *args):
            return

    return Handler


def start_loopback_server(directory, host=None, port=0):
    """Start a daemon HTTP server on 127.0.0.1 (or ::1)."""
    bound_host = require_loopback_host(host)
    httpd = http.server.ThreadingHTTPServer(
        (bound_host, port), _handler_for(os.path.abspath(directory))
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd

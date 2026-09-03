"""Listen on 127.0.0.1 only.

Helpers and tests that open a socket must go through ``require_bind_host``
or ``bind_tcp``. Wildcard, LAN, and WAN addresses are rejected before bind.
"""

import socket
from typing import Optional

BIND_HOST = "127.0.0.1"


class BindError(ValueError):
    """Raised when a bind host is not 127.0.0.1."""


def require_bind_host(host: Optional[str] = None) -> str:
    """Return ``127.0.0.1`` or raise ``BindError``.

    ``None`` defaults to loopback. ``0.0.0.0``, ``localhost``, ``::1``,
    LAN, and WAN addresses are rejected.
    """
    if host is None:
        return BIND_HOST
    normalized = str(host).strip()
    if normalized == BIND_HOST:
        return BIND_HOST
    raise BindError(
        "bind must be {0}, got {1!r}. "
        "Do not listen on a wildcard, LAN, or WAN address.".format(BIND_HOST, host)
    )


def bind_tcp(port: int = 0, host: Optional[str] = None) -> socket.socket:
    """Create a TCP socket bound to 127.0.0.1. Caller must close it."""
    address = require_bind_host(host)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((address, port))
    except Exception:
        sock.close()
        raise
    return sock


def install_localhost_bind_guard(monkeypatch) -> None:
    """Monkeypatch ``socket.socket.bind`` so only 127.0.0.1 is accepted."""
    original = socket.socket.bind

    def guarded(self, address):
        if self.family == socket.AF_INET:
            host = address[0] if isinstance(address, tuple) else address
            require_bind_host(host)
        elif self.family == socket.AF_INET6:
            raise BindError("IPv6 bind is not allowed; use {0}".format(BIND_HOST))
        return original(self, address)

    monkeypatch.setattr(socket.socket, "bind", guarded)

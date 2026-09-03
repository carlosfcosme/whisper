"""Listen on 127.0.0.1 only.

Helpers that open a socket must go through ``require_bind_host`` /
``bind_tcp``. Wildcard, LAN, and WAN addresses are rejected before bind.
"""

import socket
from typing import Optional

BIND_HOST = "127.0.0.1"


class BindError(ValueError):
    """Raised when a bind host is not 127.0.0.1."""


def require_bind_host(host: Optional[str] = None) -> str:
    """Return ``127.0.0.1`` or raise ``BindError``.

    The only accepted value is the literal IPv4 loopback address.
    ``None`` defaults to ``127.0.0.1``. ``localhost``, ``::1``,
    ``0.0.0.0``, LAN, and WAN addresses are rejected.
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
    """Create a TCP socket bound to 127.0.0.1.

    ``port`` 0 lets the OS pick a free port. The caller owns the socket
    and must close it.
    """
    address = require_bind_host(host)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((address, port))
    except Exception:
        sock.close()
        raise
    return sock

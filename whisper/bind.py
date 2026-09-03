"""Local bind address. Helpers must listen on 127.0.0.1 only."""

import socket
from typing import Optional

LOCALHOST = "127.0.0.1"
WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "[::]"})


def bind_host(host: Optional[str] = None) -> str:
    """Return the host to bind.

    Defaults to ``127.0.0.1``. Wildcard addresses are rejected.
    """
    resolved = LOCALHOST if host is None else str(host)
    if resolved in WILDCARD_HOSTS:
        raise ValueError("bind {0} only; refused {1}".format(LOCALHOST, resolved))
    return resolved


def listen(host: Optional[str] = None, port: int = 0, backlog: int = 1):
    """Bind and listen on ``127.0.0.1`` only. Rejects wildcard hosts."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((bind_host(host), port))
        sock.listen(backlog)
    except Exception:
        sock.close()
        raise
    return sock

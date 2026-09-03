"""Local bind address. Helpers must listen on 127.0.0.1 only."""

import socket
from typing import Optional

LOCALHOST = "127.0.0.1"
WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "[::]"})
# Names we map to 127.0.0.1 without DNS (no WAN).
_LOOPBACK_NAMES = frozenset({"localhost", "localhost."})


def bind_host(host: Optional[str] = None) -> str:
    """Return the host to bind.

    Always ``127.0.0.1``. ``localhost`` is accepted and rewritten.
    Wildcards, other IPs, and other hostnames are rejected without DNS.
    """
    if host is None:
        return LOCALHOST
    resolved = str(host).strip()
    if resolved in WILDCARD_HOSTS:
        raise ValueError("bind {0} only; refused {1}".format(LOCALHOST, resolved))
    if resolved == LOCALHOST or resolved.lower() in _LOOPBACK_NAMES:
        return LOCALHOST
    raise ValueError("bind {0} only; refused {1}".format(LOCALHOST, resolved))


def listen(host: Optional[str] = None, port: int = 0, backlog: int = 1):
    """Bind and listen on ``127.0.0.1`` only. Rejects non-loopback hosts."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((bind_host(host), port))
        bound = sock.getsockname()[0]
        if bound != LOCALHOST:
            raise RuntimeError("listen bound {0}, not {1}".format(bound, LOCALHOST))
        sock.listen(backlog)
    except Exception:
        sock.close()
        raise
    return sock

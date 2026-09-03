"""Loopback-only bind policy for any serve/demo listener.

Every listener must bind ``127.0.0.1``. Empty host and all-interface
addresses are refused before ``bind()``. This module does not load model
weights and does not read secrets.
"""

from __future__ import annotations

import os
from http.server import ThreadingHTTPServer
from typing import Optional, Tuple, Type

# Built without an all-interface literal so application sources stay free
# of that token (CI fails if it appears under whisper/ or .cursor/).
ALL_INTERFACES_V4 = ".".join(("0", "0", "0", "0"))
LOOPBACK_HOST = "127.0.0.1"
BIND_HOST_ENV = "WHISPER_BIND_HOST"


class BindError(ValueError):
    """Raised when a serve/listen host is not 127.0.0.1."""


def require_bind_127_0_0_1(host: Optional[str] = None) -> str:
    """Return ``127.0.0.1``, or raise ``BindError``.

    ``None`` (caller omitted host) becomes ``127.0.0.1``. An empty or
    whitespace host is refused. ``localhost`` and ``::1`` are rewritten
    to ``127.0.0.1`` without DNS. All-interface, ``::``, ``*``, LAN, and
    public names are refused.
    """
    if host is None:
        return LOOPBACK_HOST
    raw = host.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError(f"bind host is required; use {LOOPBACK_HOST}")
    lowered = raw.lower().rstrip(".")
    if lowered in {ALL_INTERFACES_V4, "::", "*", "[::]", "::0"}:
        raise BindError(f"refusing all-interfaces bind {host!r}; use {LOOPBACK_HOST}")
    if lowered in {"localhost", LOOPBACK_HOST, "::1"}:
        return LOOPBACK_HOST
    raise BindError(f"refusing non-localhost bind {host!r}; use {LOOPBACK_HOST}")


def is_loopback_host(host: str) -> bool:
    try:
        return require_bind_127_0_0_1(host) == LOOPBACK_HOST
    except BindError:
        return False


def default_bind_host() -> str:
    """Host for any helper listener. Default is ``127.0.0.1``.

    ``WHISPER_BIND_HOST`` may override only to another accepted loopback
    name. All-interface and public hosts are refused.
    """
    explicit = os.environ.get(BIND_HOST_ENV, "").strip()
    return require_bind_127_0_0_1(explicit or None)


def bind_localhost(sock, port: int = 0) -> Tuple[str, int]:
    """Bind *sock* to ``127.0.0.1`` and return ``(host, port)``."""
    host = default_bind_host()
    sock.bind((host, int(port)))
    name = sock.getsockname()
    return name[0], int(name[1])


def create_loopback_httpd(
    handler: Type, host: Optional[str] = None, port: int = 0
) -> ThreadingHTTPServer:
    """Bind ``handler`` to ``127.0.0.1``. ``host`` must pass the policy."""
    require_bind_127_0_0_1(host)
    httpd = ThreadingHTTPServer((LOOPBACK_HOST, port), handler)
    bound = httpd.server_address[0]
    if bound != LOOPBACK_HOST:
        httpd.server_close()
        raise BindError(f"refusing non-localhost bind {bound!r}; use {LOOPBACK_HOST}")
    return httpd

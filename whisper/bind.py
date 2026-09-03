"""Serve/bind guard: listen on 127.0.0.1 only.

Any serve or start path must pass ``require_bind_127_0_0_1`` before opening
a socket. Wildcard, LAN, and WAN addresses are rejected before bind.
"""

from __future__ import annotations

from typing import Optional

BIND_HOST = "127.0.0.1"


class BindError(ValueError):
    """Raised when a serve/bind path is not 127.0.0.1."""


def require_bind_127_0_0_1(host: Optional[str]) -> str:
    """Return ``127.0.0.1`` or raise ``BindError``.

    The only accepted value is the literal IPv4 loopback address.
    ``localhost``, ``::1``, empty host, LAN, WAN, and wildcards are rejected.
    """
    normalized = "" if host is None else str(host).strip()
    if normalized == BIND_HOST:
        return BIND_HOST
    raise BindError(
        f"serve/bind must use {BIND_HOST} only, got {host!r}. "
        "Do not listen on a wildcard, LAN, or WAN address."
    )

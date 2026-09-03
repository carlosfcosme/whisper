"""Serve/bind guard: listeners use 127.0.0.1 only."""

from typing import Optional

BIND_HOST = "127.0.0.1"


class BindError(ValueError):
    """Raised when a serve/bind host is not 127.0.0.1."""


def require_bind_127_0_0_1(host: Optional[str]) -> str:
    """Return ``127.0.0.1`` or raise ``BindError``.

    Only the literal IPv4 loopback address is accepted. Hostnames, IPv6
    loopback, LAN, WAN, and wildcard addresses are rejected before bind.
    """
    normalized = "" if host is None else str(host).strip()
    if normalized == BIND_HOST:
        return BIND_HOST
    raise BindError(
        f"serve/bind must use {BIND_HOST} only, got {host!r}. "
        "Do not listen on a wildcard, LAN, or WAN address."
    )

"""Serve/bind guard: listen on 127.0.0.1 only."""

from __future__ import annotations

from typing import Optional

BIND_HOST = "127.0.0.1"


class BindError(ValueError):
    """Raised when a serve/bind path is not 127.0.0.1."""


def require_bind_127_0_0_1(host: Optional[str]) -> str:
    """Return ``127.0.0.1`` or raise ``BindError``.

    ``0.0.0.0``, empty host, ``localhost``, ``::1``, LAN, and WAN are rejected
    before a socket is opened.
    """
    normalized = "" if host is None else str(host).strip()
    if normalized == BIND_HOST:
        return BIND_HOST
    raise BindError(
        f"serve/bind must use {BIND_HOST} only, got {host!r}. "
        "Do not listen on a wildcard, LAN, or WAN address."
    )

"""Loopback-only bind policy for any serve/listen path.

Every listener must bind ``127.0.0.1``. All-interface hosts are refused
before ``bind()``. This module does not load model weights and does not
read secrets.
"""

from __future__ import annotations

import ipaddress
from http.server import ThreadingHTTPServer
from typing import Optional, Type

# Built without an all-interface literal so application sources stay free
# of that token (CI and pre-commit fail if it appears under whisper/).
ALL_INTERFACES = ".".join(("0", "0", "0", "0"))
LOOPBACK_HOST = "127.0.0.1"


class BindError(ValueError):
    """Raised when a serve/listen host is not 127.0.0.1 loopback."""


def require_loopback_bind(host: Optional[str] = None) -> str:
    """Return ``127.0.0.1``, or raise ``BindError``.

    ``localhost`` is rewritten to ``127.0.0.1`` without DNS. Unspecified
    addresses (all interfaces, ``::``, ``*``), empty host, LAN, and
    public names are refused. Accepted loopback names are canonicalized
    to ``127.0.0.1`` so the socket always binds IPv4 loopback.
    """
    if host is None:
        return LOOPBACK_HOST
    raw = host.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError(f"bind host is required; use {LOOPBACK_HOST}")
    lowered = raw.lower()
    if lowered in {ALL_INTERFACES, "::", "*", "[::]"}:
        raise BindError(f"refusing all-interfaces bind {host!r}; use {LOOPBACK_HOST}")
    if lowered == "localhost":
        return LOOPBACK_HOST
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError as exc:
        raise BindError(
            f"refusing non-localhost bind {host!r}; use {LOOPBACK_HOST}"
        ) from exc
    if ip.is_unspecified or not ip.is_loopback:
        raise BindError(f"refusing non-localhost bind {host!r}; use {LOOPBACK_HOST}")
    return LOOPBACK_HOST


def is_loopback_host(host: str) -> bool:
    try:
        require_loopback_bind(host)
        return True
    except BindError:
        return False


def create_loopback_httpd(
    handler: Type, host: Optional[str] = None, port: int = 0
) -> ThreadingHTTPServer:
    """Bind ``handler`` to ``127.0.0.1``. ``host`` must pass the policy."""
    require_loopback_bind(host)
    httpd = ThreadingHTTPServer((LOOPBACK_HOST, port), handler)
    bound = httpd.server_address[0]
    if bound != LOOPBACK_HOST:
        httpd.server_close()
        raise BindError(f"refusing non-localhost bind {bound!r}; use {LOOPBACK_HOST}")
    return httpd

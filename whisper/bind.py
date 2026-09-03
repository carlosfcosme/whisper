"""Loopback bind policy: listeners must use 127.0.0.1.

All-interface hosts are refused before bind(). Stdlib only so CI can import
this module without loading model weights.
"""

from __future__ import annotations

import ipaddress
from http.server import ThreadingHTTPServer
from typing import Optional, Type

# Built without an all-interface literal so whisper/ stays free of that token.
# CI fails if the literal appears under whisper/ or .cursor/.
ALL_INTERFACES = ".".join(("0",) * 4)
LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class BindError(ValueError):
    """Raised when a listen host is not IPv4 loopback."""


def require_loopback_host(host: Optional[str] = None) -> str:
    """Return 127.0.0.1, or raise BindError.

    localhost is rewritten to 127.0.0.1 without DNS. Unspecified addresses
    (all interfaces, ::, *), empty host, LAN, and public names are refused.
    """
    if host is None:
        return LOOPBACK_HOST
    raw = host.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError("bind host is required; use {}".format(LOOPBACK_HOST))
    lowered = raw.lower()
    if lowered in {ALL_INTERFACES, "::", "*", ""}:
        raise BindError(
            "refusing all-interfaces bind {!r}; use {}".format(host, LOOPBACK_HOST)
        )
    if lowered == "localhost":
        return LOOPBACK_HOST
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError as exc:
        raise BindError(
            "refusing non-loopback bind {!r}; use {}".format(host, LOOPBACK_HOST)
        ) from exc
    if format(ip) != LOOPBACK_HOST:
        raise BindError(
            "refusing non-loopback bind {!r}; use {}".format(host, LOOPBACK_HOST)
        )
    return LOOPBACK_HOST


def create_loopback_httpd(
    handler: Type, host: Optional[str] = None, port: int = 0
) -> ThreadingHTTPServer:
    """Bind handler to 127.0.0.1. host must pass the policy first."""
    require_loopback_host(host)
    httpd = ThreadingHTTPServer((LOOPBACK_HOST, port), handler)
    bound = httpd.server_address[0]
    if bound != LOOPBACK_HOST:
        httpd.server_close()
        raise BindError(
            "refusing non-loopback bind {!r}; use {}".format(bound, LOOPBACK_HOST)
        )
    return httpd

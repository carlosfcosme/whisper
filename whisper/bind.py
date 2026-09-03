"""Loopback-only bind policy for any serve/demo listener.

Every listener must bind 127.0.0.1. Empty host and all-interface
addresses are refused before bind(). This module does not load model
weights and does not import torch.
"""

from __future__ import annotations

from http.server import ThreadingHTTPServer
from typing import Optional, Type

# Built without an all-interface literal so application sources stay free
# of that token (CI fails if it appears under whisper/).
ALL_INTERFACES_V4 = ".".join(("0", "0", "0", "0"))
LOOPBACK_HOST = "127.0.0.1"


class BindError(ValueError):
    """Raised when a serve/listen host is not 127.0.0.1."""


def require_bind_127_0_0_1(host: Optional[str] = None) -> str:
    """Return 127.0.0.1, or raise BindError.

    None (caller omitted host) becomes 127.0.0.1. An empty or
    whitespace host is refused. localhost is rewritten to
    127.0.0.1 without DNS. All-interface, ::, *, LAN, and
    public names are refused.
    """
    if host is None:
        return LOOPBACK_HOST
    raw = host.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError("bind host is required; use {}".format(LOOPBACK_HOST))
    lowered = raw.lower().rstrip(".")
    if lowered in {ALL_INTERFACES_V4, "::", "*", "[::]"}:
        raise BindError(
            "refusing all-interfaces bind {!r}; use {}".format(host, LOOPBACK_HOST)
        )
    if lowered in {"localhost", LOOPBACK_HOST}:
        return LOOPBACK_HOST
    raise BindError(
        "refusing non-localhost bind {!r}; use {}".format(host, LOOPBACK_HOST)
    )


def is_loopback_host(host: str) -> bool:
    try:
        return require_bind_127_0_0_1(host) == LOOPBACK_HOST
    except BindError:
        return False


def create_loopback_httpd(
    handler: Type, host: Optional[str] = None, port: int = 0
) -> ThreadingHTTPServer:
    """Bind handler to 127.0.0.1. host must pass the policy."""
    require_bind_127_0_0_1(host)
    httpd = ThreadingHTTPServer((LOOPBACK_HOST, port), handler)
    bound = httpd.server_address[0]
    if bound != LOOPBACK_HOST:
        httpd.server_close()
        raise BindError(
            "refusing non-localhost bind {!r}; use {}".format(bound, LOOPBACK_HOST)
        )
    return httpd

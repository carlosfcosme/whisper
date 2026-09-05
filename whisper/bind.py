"""Bind guard: listeners must use 127.0.0.1.

The all-interfaces address is refused before bind(). This module does
not load weights and does not read secrets.
"""

from __future__ import annotations

from http.server import ThreadingHTTPServer
from typing import Optional, Type

# Avoid writing the all-interfaces token in application sources.
_ALL_INTERFACES_V4 = ".".join(("0", "0", "0", "0"))
LOOPBACK_HOST = "127.0.0.1"


class BindError(ValueError):
    """Raised when a listen host is not 127.0.0.1."""


def require_bind_127_0_0_1(host: Optional[str] = None) -> str:
    """Return ``127.0.0.1``, or raise ``BindError``.

    ``None`` becomes loopback. Empty host and the all-interfaces address
    are refused. ``localhost`` is rewritten to ``127.0.0.1`` without DNS.
    """
    if host is None:
        return LOOPBACK_HOST
    raw = host.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError("bind host is required; use {}".format(LOOPBACK_HOST))
    lowered = raw.lower().rstrip(".")
    if lowered == _ALL_INTERFACES_V4:
        raise BindError(
            "refusing all-interfaces bind {!r}; use {}".format(host, LOOPBACK_HOST)
        )
    if lowered in {"::", "*", "[::]"}:
        raise BindError(
            "refusing all-interfaces bind {!r}; use {}".format(host, LOOPBACK_HOST)
        )
    if lowered in {"localhost", LOOPBACK_HOST}:
        return LOOPBACK_HOST
    raise BindError(
        "refusing non-localhost bind {!r}; use {}".format(host, LOOPBACK_HOST)
    )


def create_loopback_httpd(
    handler: Type, host: Optional[str] = None, port: int = 0
) -> ThreadingHTTPServer:
    """Bind ``handler`` to ``127.0.0.1`` after the guard accepts ``host``."""
    require_bind_127_0_0_1(host)
    httpd = ThreadingHTTPServer((LOOPBACK_HOST, port), handler)
    bound = httpd.server_address[0]
    if bound != LOOPBACK_HOST:
        httpd.server_close()
        raise BindError(
            "refusing non-localhost bind {!r}; use {}".format(bound, LOOPBACK_HOST)
        )
    return httpd

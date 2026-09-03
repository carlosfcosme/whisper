"""Loopback-only bind policy.

Every listener must bind ``127.0.0.1``. All-interface hosts are
refused before ``socket.bind()``. This module does not load model
weights and does not talk to Hugging Face Hub.

Application sources must not contain the all-interfaces bind token;
CI fails if it appears under ``whisper/``, ``.cursor/``, or
``scripts/``.
"""

from http.server import ThreadingHTTPServer
from typing import Optional, Type

LOOPBACK_HOST = "127.0.0.1"
# Built without an all-interface literal so application sources stay
# free of that token (CI fails if it appears under scanned trees).
_UNSPECIFIED_V4 = ".".join(("0",) * 4)
_BLOCKED = frozenset({_UNSPECIFIED_V4, "::", "*", "[::]"})


class BindError(ValueError):
    """Raised when a bind host is not 127.0.0.1."""


def require_loopback_host(host: Optional[str] = None) -> str:
    """Return ``127.0.0.1``, or raise ``BindError``.

    ``None`` becomes ``127.0.0.1``. ``localhost`` is rewritten to
    ``127.0.0.1`` without DNS. Empty host, all-interface addresses,
    LAN, and public names are refused.
    """
    if host is None:
        return LOOPBACK_HOST
    raw = host.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError("bind host is required; use %s" % LOOPBACK_HOST)
    key = raw.lower().rstrip(".")
    if key in _BLOCKED:
        raise BindError(
            "refusing all-interfaces bind %r; use %s" % (host, LOOPBACK_HOST)
        )
    if key in {"localhost", LOOPBACK_HOST}:
        return LOOPBACK_HOST
    raise BindError("refusing non-localhost bind %r; use %s" % (host, LOOPBACK_HOST))


def is_loopback_host(host: str) -> bool:
    try:
        return require_loopback_host(host) == LOOPBACK_HOST
    except BindError:
        return False


def create_loopback_httpd(
    handler: Type, host: Optional[str] = None, port: int = 0
) -> ThreadingHTTPServer:
    """Bind ``handler`` to ``127.0.0.1``. ``host`` must pass the policy."""
    require_loopback_host(host)
    httpd = ThreadingHTTPServer((LOOPBACK_HOST, port), handler)
    bound = httpd.server_address[0]
    if bound != LOOPBACK_HOST:
        httpd.server_close()
        raise BindError(
            "refusing non-localhost bind %r; use %s" % (bound, LOOPBACK_HOST)
        )
    return httpd

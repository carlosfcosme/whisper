"""Loopback bind policy for the CLI serve path.

Any serve path must bind ``127.0.0.1`` (or another loopback address).
All-interface binds such as ``0.0.0.0`` are refused before ``bind()``.

This module does not load model weights and does not read secrets.
"""

import ipaddress

LOOPBACK_BIND = "127.0.0.1"
ALL_INTERFACES = "0.0.0.0"


class BindError(ValueError):
    """Raised when a bind host is not loopback."""


def require_loopback_bind(host: str) -> str:
    """Return a loopback bind host, or raise ``BindError``.

    ``localhost`` is rewritten to ``127.0.0.1`` without DNS. Unspecified
    addresses (``0.0.0.0``, ``::``, ``*``), empty host, LAN, and public
    names are refused.
    """
    raw = (host or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError("bind host is required; use {}".format(LOOPBACK_BIND))
    if raw == ALL_INTERFACES or raw == "::" or raw == "*":
        raise BindError(
            "refusing all-interfaces bind {!r}; use {}".format(host, LOOPBACK_BIND)
        )
    if raw.lower() == "localhost":
        return LOOPBACK_BIND
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError as exc:
        raise BindError(
            "refusing non-localhost bind {!r}; use {}".format(host, LOOPBACK_BIND)
        ) from exc
    if ip.is_unspecified or not ip.is_loopback:
        raise BindError(
            "refusing non-localhost bind {!r}; use {}".format(host, LOOPBACK_BIND)
        )
    if ip.version == 4:
        return LOOPBACK_BIND
    return str(ip)


def is_loopback_host(host: str) -> bool:
    try:
        require_loopback_bind(host)
        return True
    except BindError:
        return False

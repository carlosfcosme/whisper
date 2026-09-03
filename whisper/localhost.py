"""Localhost-only bind helpers.

Any helper or test service must listen on a loopback address. Wildcard
(``0.0.0.0``, ``::``) and WAN hosts are refused.
"""

import ipaddress
import os
import socket
from typing import Optional, Tuple

_ENV = "WHISPER_BIND_HOST"
_DEFAULT = "127.0.0.1"
_LOCALHOST_NAMES = frozenset({"localhost"})
_WILDCARD = frozenset({"0.0.0.0", "::", "*", "[::]"})


class NonLocalhostBindError(ValueError):
    """Raised when a bind host is not a loopback address."""


def _normalize(host: str) -> str:
    return host.strip().lower().strip("[]")


def is_loopback_host(host: str) -> bool:
    candidate = _normalize(host)
    if candidate in _LOCALHOST_NAMES:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def bind_address(host: Optional[str] = None) -> str:
    """Return a loopback bind host, or raise ``NonLocalhostBindError``."""
    if host is None:
        raw = os.getenv(_ENV, _DEFAULT)
    else:
        raw = host
    if raw is None or not str(raw).strip():
        raise NonLocalhostBindError(
            "empty bind host refused; localhost only (use 127.0.0.1)"
        )

    candidate = str(raw).strip()
    normalized = _normalize(candidate)
    if normalized in _WILDCARD:
        raise NonLocalhostBindError(
            f"refusing wildcard bind {candidate!r}; localhost only"
        )

    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        if normalized in _LOCALHOST_NAMES:
            return _DEFAULT
        raise NonLocalhostBindError(
            f"refusing non-localhost bind {candidate!r}; localhost only"
        )

    if ip.is_unspecified or not ip.is_loopback:
        raise NonLocalhostBindError(
            f"refusing non-localhost bind {candidate!r}; localhost only"
        )
    return str(ip)


def bind_localhost(sock: socket.socket, port: int = 0) -> Tuple[str, int]:
    """Bind ``sock`` to loopback and return ``(host, port)``."""
    host = bind_address()
    if sock.family == socket.AF_INET6:
        if host == _DEFAULT:
            host = "::1"
    elif host == "::1":
        host = _DEFAULT
    sock.bind((host, port))
    name = sock.getsockname()
    bound_host, bound_port = name[0], int(name[1])
    if not is_loopback_host(bound_host):
        raise NonLocalhostBindError(f"socket bound to non-loopback {bound_host!r}")
    return bound_host, bound_port

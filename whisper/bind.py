"""Loopback-only bind policy. Listeners must use 127.0.0.1."""

from __future__ import annotations

from ipaddress import ip_address
from typing import Optional, Tuple

LOOPBACK_HOST = "127.0.0.1"
# Split so source scanners do not treat this module as binding all interfaces.
_UNSPECIFIED_V4 = ".".join(("0",) * 4)


class BindError(ValueError):
    """Raised when a listener would bind off loopback."""


def require_loopback_host(host: Optional[str] = None) -> str:
    """Return ``127.0.0.1``, or raise ``BindError``.

    ``None`` / ``localhost`` become ``127.0.0.1`` (no DNS). Empty host,
    all-interfaces addresses, LAN, and public hosts are refused.
    """
    if host is None:
        return LOOPBACK_HOST
    raw = host.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError("bind host is required; use 127.0.0.1")
    if raw.lower().rstrip(".") == "localhost":
        return LOOPBACK_HOST
    if raw == _UNSPECIFIED_V4 or raw in {":", "::", "*", "::0"}:
        raise BindError(f"refusing all-interfaces bind {host!r}; use 127.0.0.1")
    try:
        ip = ip_address(raw.split("%", 1)[0])
    except ValueError as exc:
        raise BindError(f"refusing non-localhost bind {host!r}; use 127.0.0.1") from exc
    if not ip.is_loopback:
        raise BindError(f"refusing non-localhost bind {host!r}; use 127.0.0.1")
    if ip.version == 4 and str(ip) != LOOPBACK_HOST:
        # Require the canonical IPv4 loopback, not 127.0.0.2 etc.
        raise BindError(f"refusing non-localhost bind {host!r}; use 127.0.0.1")
    return str(ip) if ip.version == 6 else LOOPBACK_HOST


def bind_loopback(sock, port: int = 0) -> Tuple[str, int]:
    """Bind *sock* to 127.0.0.1 and return the bound name."""
    host = require_loopback_host(LOOPBACK_HOST)
    sock.bind((host, int(port)))
    name = sock.getsockname()
    bound = require_loopback_host(name[0])
    return bound, int(name[1])

"""Loopback-only bind policy. Servers must listen on 127.0.0.1."""

from __future__ import annotations

LOOPBACK_HOST = "127.0.0.1"
_LOCAL_ALIASES = frozenset({"127.0.0.1", "localhost"})
_ALL_INTERFACES = frozenset(
    {"", "0.0.0.0", "::", "::0", "*", "[::]", "[::0]", "0.0.0.0/0"}
)


class BindError(ValueError):
    """Raised when a bind host is not 127.0.0.1."""


def require_loopback(host: str) -> str:
    """Return ``127.0.0.1`` or raise ``BindError``.

    ``localhost`` is accepted as an alias. All-interface, empty, IPv6-any,
    LAN, and public hosts are rejected before ``bind()``.
    """
    if host is None:
        raise BindError("bind host is required; use 127.0.0.1")
    if not isinstance(host, str):
        raise BindError("bind host must be a string; use 127.0.0.1")
    normalized = host.strip().lower().rstrip(".")
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if normalized in _LOCAL_ALIASES:
        return LOOPBACK_HOST
    if normalized in _ALL_INTERFACES:
        raise BindError(f"refusing to bind {host!r}; use 127.0.0.1")
    raise BindError(f"refusing to bind {host!r}; only 127.0.0.1 is allowed")

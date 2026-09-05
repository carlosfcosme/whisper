"""Refuse non-loopback bind hosts. Servers must listen on 127.0.0.1."""

import socket

from .defaults import DEFAULT_BIND_HOST

_FORBIDDEN = frozenset(
    {
        "",
        "0.0.0.0",
        "0",
        "*",
        "::",
        "::0",
        "[::]",
        "[::0]",
        "0:0:0:0:0:0:0:0",
    }
)
_ALLOWED = frozenset({"127.0.0.1", "localhost"})


def require_loopback_host(host: str = DEFAULT_BIND_HOST) -> str:
    """Return a loopback host, or raise ValueError for wildcards / public binds."""
    if host is None:
        raise ValueError("bind host is required; use 127.0.0.1")
    normalized = host.strip().lower().strip("[]")
    if normalized in _FORBIDDEN or normalized.startswith("0.0.0.0"):
        raise ValueError(
            f"refusing non-loopback bind {host!r}; use {DEFAULT_BIND_HOST}"
        )
    if normalized not in _ALLOWED:
        raise ValueError(
            f"refusing bind host {host!r}; only {DEFAULT_BIND_HOST} is allowed"
        )
    return DEFAULT_BIND_HOST


_orig_socket_bind = socket.socket.bind
_guard_installed = False


def _bind_loopback_only(self, address, *args, **kwargs):
    host = ""
    if isinstance(address, tuple) and address:
        raw = address[0]
        if raw is None:
            host = ""
        elif isinstance(raw, bytes):
            host = raw.decode("ascii", "replace")
        else:
            host = str(raw)
    require_loopback_host(host)
    return _orig_socket_bind(self, address, *args, **kwargs)


def install_loopback_bind_guard() -> None:
    """Wrap socket.socket.bind so non-loopback binds raise ValueError."""
    global _guard_installed
    if _guard_installed:
        return
    socket.socket.bind = _bind_loopback_only
    _guard_installed = True

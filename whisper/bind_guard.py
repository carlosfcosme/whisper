"""Loopback bind guard. Accepts 127.0.0.1 / ::1; fails on 0.0.0.0."""

from __future__ import annotations

import ipaddress

DEFAULT_HOST = "127.0.0.1"
ALL_INTERFACE_HOSTS = frozenset({"0.0.0.0", "::", "[::]"})


class BindError(ValueError):
    """Raised when a bind host is not a loopback address."""


def bind_guard(host: str) -> str:
    """Return a loopback bind address, or raise BindError.

    ``127.0.0.1``, ``::1``, and ``localhost`` (rewritten, no DNS) are
    accepted. ``0.0.0.0``, ``::``, LAN, and public hosts fail.
    """
    raw = (host or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError("bind host is required; use 127.0.0.1")
    if raw in ALL_INTERFACE_HOSTS:
        raise BindError(f"refusing non-localhost bind {host!r}; use 127.0.0.1")
    if raw.lower() == "localhost":
        return DEFAULT_HOST
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError as exc:
        raise BindError(f"refusing non-localhost bind {host!r}; use 127.0.0.1") from exc
    if not ip.is_loopback:
        raise BindError(f"refusing non-localhost bind {host!r}; use 127.0.0.1")
    return str(ip)


def normalize_bind_host(host: str) -> str:
    return bind_guard(host)


def is_loopback_host(host: str) -> bool:
    try:
        bind_guard(host)
        return True
    except BindError:
        return False

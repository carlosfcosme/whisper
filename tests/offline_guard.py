"""Disable non-loopback network and binds during tests."""

from __future__ import annotations

import socket
from typing import Callable, Optional

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class NetworkDisabledError(OSError):
    """Raised when a test tries to use a non-loopback host."""


_installed = False
_orig_connect: Optional[Callable] = None
_orig_connect_ex: Optional[Callable] = None
_orig_bind: Optional[Callable] = None
_orig_create_connection: Optional[Callable] = None
_orig_getaddrinfo: Optional[Callable] = None


def _host_from_address(address) -> str:
    if isinstance(address, bytes):
        try:
            address = address.decode("utf-8", "replace")
        except Exception:
            return ""
    if isinstance(address, str):
        return address
    if isinstance(address, tuple) and address:
        host = address[0]
        if isinstance(host, bytes):
            try:
                host = host.decode("utf-8", "replace")
            except Exception:
                return ""
        return "" if host is None else str(host)
    return ""


def is_loopback_host(host: str) -> bool:
    if host is None:
        return False
    cleaned = str(host).strip().strip("[]").lower()
    if cleaned in LOOPBACK_HOSTS:
        return True
    if cleaned.startswith("127."):
        return True
    return False


def _refuse(action: str, host: str) -> None:
    raise NetworkDisabledError(
        "offline tests: {0} refused for {1!r} (loopback only)".format(action, host)
    )


def install_network_guard() -> None:
    global _installed
    global _orig_connect, _orig_connect_ex, _orig_bind
    global _orig_create_connection, _orig_getaddrinfo
    if _installed:
        return

    _orig_connect = socket.socket.connect
    _orig_connect_ex = socket.socket.connect_ex
    _orig_bind = socket.socket.bind
    _orig_create_connection = socket.create_connection
    _orig_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address):
        if getattr(self, "family", None) == socket.AF_UNIX:
            return _orig_connect(self, address)
        host = _host_from_address(address)
        if not is_loopback_host(host):
            _refuse("connect", host)
        return _orig_connect(self, address)

    def guarded_connect_ex(self, address):
        if getattr(self, "family", None) == socket.AF_UNIX:
            return _orig_connect_ex(self, address)
        host = _host_from_address(address)
        if not is_loopback_host(host):
            _refuse("connect", host)
        return _orig_connect_ex(self, address)

    def guarded_bind(self, address):
        if getattr(self, "family", None) == socket.AF_UNIX:
            return _orig_bind(self, address)
        host = _host_from_address(address)
        if host in {"", "0.0.0.0", "::", "::0"}:
            _refuse("bind", host or "0.0.0.0")
        if not is_loopback_host(host):
            _refuse("bind", host)
        return _orig_bind(self, address)

    def guarded_create_connection(address, *args, **kwargs):
        host = _host_from_address(address)
        if not is_loopback_host(host):
            _refuse("connect", host)
        return _orig_create_connection(address, *args, **kwargs)

    def guarded_getaddrinfo(host, port, *args, **kwargs):
        if host in (None, ""):
            return _orig_getaddrinfo(host, port, *args, **kwargs)
        if not is_loopback_host(str(host)):
            _refuse("dns", str(host))
        return _orig_getaddrinfo(host, port, *args, **kwargs)

    socket.socket.connect = guarded_connect  # type: ignore[assignment]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[assignment]
    socket.socket.bind = guarded_bind  # type: ignore[assignment]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
    _installed = True


def uninstall_network_guard() -> None:
    global _installed
    if not _installed:
        return
    socket.socket.connect = _orig_connect  # type: ignore[assignment]
    socket.socket.connect_ex = _orig_connect_ex  # type: ignore[assignment]
    socket.socket.bind = _orig_bind  # type: ignore[assignment]
    socket.create_connection = _orig_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = _orig_getaddrinfo  # type: ignore[assignment]
    _installed = False


def guard_is_installed() -> bool:
    return _installed

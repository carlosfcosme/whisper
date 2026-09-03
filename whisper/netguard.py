"""Disable WAN while leaving 127.0.0.1 loopback available.

Used by tests and by subprocess integration (via tests/netdisable/sitecustomize.py).
DNS and connect to non-loopback hosts fail immediately. Loopback HTTP still works.
"""

import ipaddress
import socket
from typing import Any, Optional

LOOPBACK_NAMES = frozenset({"127.0.0.1", "::1", "localhost"})
_DISABLED_MSG = "network disabled: refusing WAN to {!r}"

_installed = False
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo


class NetworkDisabled(OSError):
    """Raised when a non-loopback connect or DNS lookup is attempted."""


def _host_from_address(address: Any) -> Optional[object]:
    if isinstance(address, tuple) and address:
        return address[0]
    return address


def is_loopback_connect_host(host: Any) -> bool:
    """Return True if *host* is loopback, or not a network host (AF_UNIX)."""
    if not isinstance(host, str):
        return True
    raw = host.split("%", 1)[0].strip()
    if raw.lower() in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(raw).is_loopback
    except ValueError:
        return False


def _guarded_connect(self, address, *args, **kwargs):
    if not isinstance(address, tuple):
        return _real_connect(self, address, *args, **kwargs)
    host = _host_from_address(address)
    if is_loopback_connect_host(host):
        return _real_connect(self, address, *args, **kwargs)
    raise NetworkDisabled(_DISABLED_MSG.format(host))


def _guarded_connect_ex(self, address, *args, **kwargs):
    if not isinstance(address, tuple):
        return _real_connect_ex(self, address, *args, **kwargs)
    host = _host_from_address(address)
    if is_loopback_connect_host(host):
        return _real_connect_ex(self, address, *args, **kwargs)
    raise NetworkDisabled(_DISABLED_MSG.format(host))


def _guarded_getaddrinfo(host, port, *args, **kwargs):
    if host is None or is_loopback_connect_host(host):
        return _real_getaddrinfo(host, port, *args, **kwargs)
    raise NetworkDisabled(_DISABLED_MSG.format(host))


def install() -> None:
    """Patch socket so non-loopback DNS and connect fail."""
    global _installed
    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = _guarded_getaddrinfo
    _installed = True


def uninstall() -> None:
    """Restore the original socket connect and DNS helpers."""
    global _installed
    socket.socket.connect = _real_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _real_connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = _real_getaddrinfo
    _installed = False


def is_installed() -> bool:
    return _installed

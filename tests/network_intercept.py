"""Session-wide network interception for offline tests.

Only loopback (127.0.0.1 / ::1 / localhost) may connect. Hugging Face Hub,
Azure CDN, and any other remote host raise NetworkIntercepted before a
packet is sent. This module does not download weights and does not read keys.
"""

from __future__ import annotations

import socket
from typing import Any, Iterable

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class NetworkIntercepted(RuntimeError):
    """Raised when a test tries to open a non-loopback connection."""


_installed = False
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_create_connection = socket.create_connection


def _peer_host(address: Any) -> str:
    if isinstance(address, (tuple, list)) and address:
        return str(address[0]).strip("[]")
    return str(address)


def is_loopback_peer(host: str) -> bool:
    return (host or "").strip().lower().rstrip(".") in LOOPBACK_HOSTS


def deny_if_remote(address: Any) -> None:
    host = _peer_host(address)
    if is_loopback_peer(host):
        return
    raise NetworkIntercepted(
        "offline intercept: refusing connect to {!r}; loopback only".format(host)
    )


def _connect(self, address):  # type: ignore[no-untyped-def]
    deny_if_remote(address)
    return _orig_connect(self, address)


def _connect_ex(self, address):  # type: ignore[no-untyped-def]
    deny_if_remote(address)
    return _orig_connect_ex(self, address)


def _create_connection(address, *args, **kwargs):  # type: ignore[no-untyped-def]
    deny_if_remote(address)
    return _orig_create_connection(address, *args, **kwargs)


def install() -> None:
    global _installed
    if _installed:
        return
    socket.socket.connect = _connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _connect_ex  # type: ignore[method-assign]
    socket.create_connection = _create_connection  # type: ignore[assignment]
    _installed = True


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    socket.socket.connect = _orig_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _orig_connect_ex  # type: ignore[method-assign]
    socket.create_connection = _orig_create_connection  # type: ignore[assignment]
    _installed = False


def installed() -> bool:
    return _installed


def remote_hosts() -> Iterable[str]:
    return (
        "huggingface.co",
        "hf.co",
        "openaipublic.azureedge.net",
        "8.8.8.8",
    )

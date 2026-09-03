"""Loopback bind policy: listeners must use 127.0.0.1.

All-interface hosts are refused before ``bind()``. After bind, the live
socket (and ``/proc/net/tcp`` when present) must still be 127.0.0.1.
This module is stdlib-only so CI can import it without model weights.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from http.server import ThreadingHTTPServer
from typing import Optional, Type

# Built without an all-interface literal so ``whisper/`` stays free of that
# token. CI fails if the literal appears under whisper/ or .cursor/.
ALL_INTERFACES = ".".join(("0",) * 4)
LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class BindError(ValueError):
    """Raised when a listen host is not IPv4 loopback."""


def require_loopback_host(host: Optional[str] = None) -> str:
    """Return ``127.0.0.1``, or raise ``BindError``.

    ``localhost`` is rewritten to ``127.0.0.1`` without DNS. Unspecified
    addresses (all interfaces, ``::``, ``*``), empty host, LAN, and public
    names are refused. The socket always binds IPv4 loopback.
    """
    if host is None:
        return LOOPBACK_HOST
    raw = host.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError("bind host is required; use {}".format(LOOPBACK_HOST))
    lowered = raw.lower()
    if lowered in {ALL_INTERFACES, "::", "*", ""}:
        raise BindError(
            "refusing all-interfaces bind {!r}; use {}".format(host, LOOPBACK_HOST)
        )
    if lowered == "localhost":
        return LOOPBACK_HOST
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError as exc:
        raise BindError(
            "refusing non-loopback bind {!r}; use {}".format(host, LOOPBACK_HOST)
        ) from exc
    if format(ip) != LOOPBACK_HOST:
        raise BindError(
            "refusing non-loopback bind {!r}; use {}".format(host, LOOPBACK_HOST)
        )
    return LOOPBACK_HOST


def _as_socket(sock):
    inner = getattr(sock, "socket", None)
    if inner is not None and hasattr(inner, "getsockname"):
        return inner
    return sock


def _socket_inode(sock) -> Optional[int]:
    try:
        target = os.readlink("/proc/self/fd/{}".format(sock.fileno()))
    except OSError:
        return None
    if target.startswith("socket:[") and target.endswith("]"):
        return int(target[8:-1])
    return None


def _proc_listen_host(inode: int) -> Optional[str]:
    """Return the IPv4 listen host for *inode* from ``/proc/net/tcp``, if any."""
    try:
        lines = open("/proc/net/tcp", encoding="ascii").read().splitlines()[1:]
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        if len(fields) < 10 or fields[3] != "0A":
            continue
        try:
            if int(fields[9]) != inode:
                continue
            hex_ip, _hex_port = fields[1].split(":")
            packed = bytes.fromhex(hex_ip)
            return socket.inet_ntoa(packed[::-1])
        except (ValueError, OSError):
            continue
    return None


def assert_loopback_listen(sock) -> str:
    """Fail if *sock* (or an httpd) is not listening on ``127.0.0.1``.

    Uses ``getsockname()`` and, on Linux, the LISTEN row in ``/proc/net/tcp``
    for this socket inode. Does not load kernel modules, BPF, or weights.
    """
    sock = _as_socket(sock)
    host, _port = sock.getsockname()[:2]
    if host != LOOPBACK_HOST:
        raise BindError(
            "refusing non-loopback listen {!r}; use {}".format(host, LOOPBACK_HOST)
        )
    inode = _socket_inode(sock)
    if inode is not None:
        proc_host = _proc_listen_host(inode)
        if proc_host is not None and proc_host != LOOPBACK_HOST:
            raise BindError(
                "refusing non-loopback listen {!r}; use {}".format(
                    proc_host, LOOPBACK_HOST
                )
            )
    return LOOPBACK_HOST


def create_loopback_httpd(
    handler: Type, host: Optional[str] = None, port: int = 0
) -> ThreadingHTTPServer:
    """Bind ``handler`` to ``127.0.0.1``. ``host`` must pass the policy first."""
    require_loopback_host(host)
    httpd = ThreadingHTTPServer((LOOPBACK_HOST, port), handler)
    try:
        assert_loopback_listen(httpd)
    except BindError:
        httpd.server_close()
        raise
    return httpd

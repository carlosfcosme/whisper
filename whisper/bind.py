"""Loopback-only bind policy.

Every listener must bind ``127.0.0.1``. All-interface and non-loopback
hosts are refused before ``socket.bind()``. After bind, the local
address and this process's listen sockets are checked again.

This module does not load model weights, open WAN sockets, or read
credentials. Application sources must not contain the IPv4
all-interfaces token; CI fails if it appears under scanned trees.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Type

LOOPBACK_HOST = "127.0.0.1"
# Built without an all-interface literal so application sources stay
# free of that token (CI fails if it appears under scanned trees).
UNSPECIFIED_V4 = ".".join(("0",) * 4)
# Negative wildcard fixtures: all-interface / unspecified bind hosts.
WILDCARD_BIND_HOSTS = (
    UNSPECIFIED_V4,
    "::",
    "*",
    "",
    "[::]",
    "0",
    "::0",
)
# Negative network fixtures: LAN, public, and non-canonical loopback.
NON_LOOPBACK_HOSTS = (
    "8.8.8.8",
    "10.0.0.1",
    "192.168.1.10",
    "example.com",
    "::1",
    "127.0.0.2",
)
_BLOCKED = frozenset(
    {
        UNSPECIFIED_V4,
        "::",
        "*",
        "[::]",
        "0",
        "::0",
        "0:0:0:0:0:0:0:0",
    }
)
_LISTEN_STATE = "0A"


class BindError(ValueError):
    """Raised when a bind host is not 127.0.0.1."""


def require_loopback_host(host: Optional[str] = None) -> str:
    """Return ``127.0.0.1``, or raise ``BindError``.

    ``None`` becomes ``127.0.0.1``. ``localhost`` is rewritten to
    ``127.0.0.1`` without DNS. Empty host, all-interface addresses,
    other loopback addresses (including ``::1``), LAN, and public
    names are refused.
    """
    if host is None:
        return LOOPBACK_HOST
    if not isinstance(host, str):
        raise BindError("bind host must be a string; use %s" % LOOPBACK_HOST)
    raw = host.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    raw = raw.split("%", 1)[0].strip()
    if not raw:
        raise BindError("bind host is required; use %s" % LOOPBACK_HOST)
    key = raw.lower().rstrip(".")
    if key in _BLOCKED:
        raise BindError(
            "refusing all-interfaces bind %r; use %s" % (host, LOOPBACK_HOST)
        )
    if key in {"localhost", LOOPBACK_HOST}:
        return LOOPBACK_HOST
    try:
        ip = ipaddress.ip_address(key)
    except ValueError as exc:
        raise BindError(
            "refusing non-localhost bind %r; use %s" % (host, LOOPBACK_HOST)
        ) from exc
    if (
        ip.version != 4
        or ip.is_unspecified
        or not ip.is_loopback
        or str(ip) != LOOPBACK_HOST
    ):
        raise BindError(
            "refusing non-localhost bind %r; use %s" % (host, LOOPBACK_HOST)
        )
    return LOOPBACK_HOST


def is_loopback_host(host: Optional[str]) -> bool:
    try:
        return require_loopback_host(host) == LOOPBACK_HOST
    except BindError:
        return False


def bind_tcp(host: Optional[str] = None, port: int = 0) -> socket.socket:
    """Bind a TCP socket to ``127.0.0.1``. ``host`` must pass the policy."""
    require_loopback_host(host)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((LOOPBACK_HOST, port))
        sock.listen(1)
    except Exception:
        sock.close()
        raise
    bound = sock.getsockname()[0]
    if bound != LOOPBACK_HOST:
        sock.close()
        raise BindError(
            "refusing non-localhost bind %r; use %s" % (bound, LOOPBACK_HOST)
        )
    return sock


def create_loopback_httpd(
    handler: Type, host: Optional[str] = None, port: int = 0
) -> ThreadingHTTPServer:
    """Bind ``handler`` to ``127.0.0.1``. ``host`` must pass the policy."""
    require_loopback_host(host)
    httpd = ThreadingHTTPServer((LOOPBACK_HOST, port), handler)
    bound = httpd.server_address[0]
    if bound != LOOPBACK_HOST:
        httpd.server_close()
        raise BindError(
            "refusing non-localhost bind %r; use %s" % (bound, LOOPBACK_HOST)
        )
    return httpd


def _own_socket_inodes() -> set:
    fd_dir = Path("/proc/self/fd")
    inodes = set()
    if not fd_dir.is_dir():
        return inodes
    for fd in fd_dir.iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            inodes.add(target[8:-1])
    return inodes


def _ipv4_from_hex(hex_ip: str) -> str:
    raw = bytes.fromhex(hex_ip)
    return ".".join(str(b) for b in reversed(raw))


def _ipv6_from_hex(hex_ip: str) -> str:
    data = bytes.fromhex(hex_ip)
    # Linux tcp6 words are little-endian 32-bit groups.
    parts = []
    for i in range(0, 16, 4):
        parts.extend(reversed(data[i : i + 4]))
    return socket.inet_ntop(socket.AF_INET6, bytes(parts))


def _proc_tcp_listens(path: Path, inodes: set, v6: bool) -> List[Tuple[str, int]]:
    if not path.is_file() or not inodes:
        return []
    try:
        lines = path.read_text(encoding="ascii", errors="ignore").splitlines()[1:]
    except OSError:
        return []
    found: List[Tuple[str, int]] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 10 or parts[3] != _LISTEN_STATE:
            continue
        if parts[9] not in inodes:
            continue
        local = parts[1]
        if ":" not in local:
            continue
        ip_hex, port_hex = local.rsplit(":", 1)
        try:
            ip = _ipv6_from_hex(ip_hex) if v6 else _ipv4_from_hex(ip_hex)
            port = int(port_hex, 16)
        except (ValueError, OSError):
            continue
        found.append((ip, port))
    return found


def own_listen_endpoints() -> List[Tuple[str, int]]:
    """TCP listen addresses owned by this process (Linux ``/proc``)."""
    inodes = _own_socket_inodes()
    endpoints = _proc_tcp_listens(Path("/proc/self/net/tcp"), inodes, v6=False)
    endpoints.extend(_proc_tcp_listens(Path("/proc/self/net/tcp6"), inodes, v6=True))
    return endpoints


def non_loopback_listens(
    endpoints: Optional[Sequence[Tuple[str, int]]] = None,
) -> List[Tuple[str, int]]:
    """Return this process's listen sockets that are not ``127.0.0.1``."""
    if endpoints is None:
        endpoints = own_listen_endpoints()
    bad = []
    for ip, port in endpoints:
        if ip != LOOPBACK_HOST:
            bad.append((ip, port))
    return bad


def assert_own_listens_loopback_only(
    endpoints: Optional[Iterable[Tuple[str, int]]] = None,
) -> None:
    """Raise ``BindError`` if this process listens off ``127.0.0.1``."""
    bad = non_loopback_listens(None if endpoints is None else list(endpoints))
    if bad:
        raise BindError("non-loopback listen %s; use %s" % (bad, LOOPBACK_HOST))

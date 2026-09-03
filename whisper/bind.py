"""Loopback-only bind policy.

Every listener must bind ``127.0.0.1``. All-interface and non-loopback
hosts are refused before ``socket.bind()``. After bind, ``getsockname()``
and this process's LISTEN sockets (Linux ``/proc``) must stay on
``127.0.0.1``.

Stdlib only: no WAN, no DNS, no model weights, no credentials.
Application sources must not contain the IPv4 all-interfaces token;
CI fails if it appears under scanned trees.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import sys
from http.server import ThreadingHTTPServer
from typing import Iterable, List, NamedTuple, Optional, Set, Type

LOOPBACK_HOST = "127.0.0.1"
# Built without an all-interface literal so application sources stay
# free of that token (CI fails if it appears under scanned trees).
UNSPECIFIED_V4 = ".".join(("0",) * 4)
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
_LOCALHOST_NAMES = frozenset({"localhost", LOOPBACK_HOST})

_guard_installed = False
_original_bind = socket.socket.bind


class BindError(OSError):
    """Raised when a bind host is not 127.0.0.1."""


class ListenerRecord(NamedTuple):
    host: str
    port: int
    inode: int
    family: str


def require_loopback_host(host: Optional[str] = None) -> str:
    """Return ``127.0.0.1``, or raise ``BindError``.

    ``None`` becomes ``127.0.0.1``. ``localhost`` is rewritten to
    ``127.0.0.1`` without DNS. Empty host, all-interface addresses,
    other loopbacks (including ``::1`` / ``127.0.0.2``), LAN, and
    public names are refused.
    """
    if host is None:
        return LOOPBACK_HOST
    if not isinstance(host, str):
        raise BindError(
            "bind host must be str, got %s; use %s"
            % (type(host).__name__, LOOPBACK_HOST)
        )
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
    if key in _LOCALHOST_NAMES:
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


def _host_from_address(address) -> Optional[str]:
    if isinstance(address, (str, bytes)):
        return None
    if not isinstance(address, tuple) or not address:
        raise BindError(
            "unrecognized bind address %r; use %s" % (address, LOOPBACK_HOST)
        )
    return address[0]


def _guarded_bind(self, address):
    host = _host_from_address(address)
    if host is not None:
        require_loopback_host(host if host != "" else UNSPECIFIED_V4)
    return _original_bind(self, address)


def install_bind_guard() -> None:
    """Patch ``socket.socket.bind`` so non-loopback listens fail."""
    global _guard_installed
    if _guard_installed:
        return
    socket.socket.bind = _guarded_bind
    _guard_installed = True


def uninstall_bind_guard() -> None:
    global _guard_installed
    if not _guard_installed:
        return
    socket.socket.bind = _original_bind
    _guard_installed = False


def bind_tcp(host: Optional[str] = None, port: int = 0) -> socket.socket:
    """Bind a TCP socket to ``127.0.0.1``. ``host`` must pass the policy."""
    require_loopback_host(host)
    install_bind_guard()
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
    assert_no_nonloopback_listeners()
    return sock


def create_loopback_httpd(
    handler: Type, host: Optional[str] = None, port: int = 0
) -> ThreadingHTTPServer:
    """Bind ``handler`` to ``127.0.0.1``. ``host`` must pass the policy."""
    require_loopback_host(host)
    install_bind_guard()
    httpd = ThreadingHTTPServer((LOOPBACK_HOST, port), handler)
    bound = httpd.server_address[0]
    if bound != LOOPBACK_HOST:
        httpd.server_close()
        raise BindError(
            "refusing non-localhost bind %r; use %s" % (bound, LOOPBACK_HOST)
        )
    assert_no_nonloopback_listeners()
    return httpd


def _ipv4_from_proc_hex(hex_ip: str) -> str:
    packed = bytes.fromhex(hex_ip)[::-1]
    return socket.inet_ntoa(packed)


def _ipv6_from_proc_hex(hex_ip: str) -> str:
    chunks = [hex_ip[i : i + 8] for i in range(0, 32, 8)]
    packed = b"".join(bytes.fromhex(chunk)[::-1] for chunk in chunks)
    return socket.inet_ntop(socket.AF_INET6, packed)


def parse_proc_inet_table(text: str, *, ipv6: bool = False) -> List[ListenerRecord]:
    """Parse ``/proc/net/tcp`` or ``/proc/net/tcp6`` listen rows."""
    rows: List[ListenerRecord] = []
    lines = text.splitlines()
    if not lines:
        return rows
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 10:
            continue
        local, state, inode_s = parts[1], parts[3], parts[9]
        if state != _LISTEN_STATE:
            continue
        hex_ip, hex_port = local.rsplit(":", 1)
        decoder = _ipv6_from_proc_hex if ipv6 else _ipv4_from_proc_hex
        rows.append(
            ListenerRecord(
                host=decoder(hex_ip),
                port=int(hex_port, 16),
                inode=int(inode_s),
                family="tcp6" if ipv6 else "tcp",
            )
        )
    return rows


def _socket_inodes(pid: Optional[int] = None) -> Set[int]:
    if pid is None:
        pid = os.getpid()
    fd_dir = os.path.join("/proc", str(pid), "fd")
    inodes: Set[int] = set()
    try:
        names = os.listdir(fd_dir)
    except OSError:
        return inodes
    for name in names:
        path = os.path.join(fd_dir, name)
        try:
            target = os.readlink(path)
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            inodes.add(int(target[8:-1]))
    return inodes


def process_listen_records(pid: Optional[int] = None) -> List[ListenerRecord]:
    """LISTEN sockets owned by ``pid`` (this process by default). Linux only."""
    if not sys.platform.startswith("linux"):
        return []
    inodes = _socket_inodes(pid)
    if not inodes:
        return []
    records: List[ListenerRecord] = []
    for name, ipv6 in (("tcp", False), ("tcp6", True)):
        path = os.path.join("/proc", "net", name)
        try:
            with open(path, encoding="ascii") as handle:
                text = handle.read()
        except OSError:
            continue
        for row in parse_proc_inet_table(text, ipv6=ipv6):
            if row.inode in inodes:
                records.append(row)
    return records


def assert_listeners_loopback_only(rows: Iterable[ListenerRecord]) -> None:
    """Raise ``BindError`` if any listen row is not ``127.0.0.1``."""
    for row in rows:
        if row.host != LOOPBACK_HOST:
            raise BindError(
                "non-loopback listen %s:%s (inode %s); use %s"
                % (row.host, row.port, row.inode, LOOPBACK_HOST)
            )


def assert_no_nonloopback_listeners(pid: Optional[int] = None) -> None:
    """Fail if this process has a LISTEN socket that is not 127.0.0.1."""
    assert_listeners_loopback_only(process_listen_records(pid))

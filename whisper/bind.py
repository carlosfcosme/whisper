"""Loopback-only bind guards.

Helpers that listen must use these functions. Only ``127.0.0.1`` is allowed.
``0.0.0.0``, ``::``, and any non-loopback address are refused before bind.

Offline-safe: stdlib only, no network clients, no credentials, no kernel
modules, no BPF.
"""

from __future__ import annotations

import ast
import os
import socket
import struct
from ipaddress import IPv4Address, IPv6Address
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

ALLOWED_BIND_HOST = "127.0.0.1"
_WILDCARD_SNIPPETS = ("0.0.0.0", "INADDR_ANY", "in6addr_any")
_TCP_LISTEN = 0x0A
_BIND_CALL_NAMES = frozenset(
    {"bind", "listen", "bind_loopback", "listen_loopback", "bind_and_listen"}
)

Host = Union[str, bytes, IPv4Address, IPv6Address, None]


class NonLoopbackBindError(ValueError):
    """Raised when a bind/listen host is not 127.0.0.1."""


class NonLoopbackListenError(RuntimeError):
    """Raised when this process has a LISTEN socket on a non-loopback address."""


def is_allowed_bind_host(host: Host) -> bool:
    """Return True only for the IPv4 loopback address 127.0.0.1."""
    try:
        require_loopback_host(host)
    except NonLoopbackBindError:
        return False
    return True


def require_loopback_host(host: Host) -> str:
    """Return ``127.0.0.1`` or raise :class:`NonLoopbackBindError`."""
    if host is None:
        raise NonLoopbackBindError("bind host is required; only 127.0.0.1 is allowed")
    if isinstance(host, bytes):
        try:
            host = host.decode("ascii")
        except UnicodeDecodeError as exc:
            raise NonLoopbackBindError("bind host must be 127.0.0.1") from exc
    if isinstance(host, (IPv4Address, IPv6Address)):
        host = str(host)
    if not isinstance(host, str):
        raise NonLoopbackBindError(
            f"refusing bind host {host!r}; only 127.0.0.1 is allowed"
        )
    host = host.strip()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if host != ALLOWED_BIND_HOST:
        raise NonLoopbackBindError(
            f"refusing non-loopback bind host {host!r}; only 127.0.0.1 is allowed"
        )
    return ALLOWED_BIND_HOST


def bind_loopback(
    sock: socket.socket,
    port: int,
    host: Host = ALLOWED_BIND_HOST,
) -> socket.socket:
    """Bind ``sock`` to ``127.0.0.1`` after refusing any other host."""
    host = require_loopback_host(host)
    if not isinstance(port, int) or isinstance(port, bool) or not (0 <= port <= 65535):
        raise NonLoopbackBindError(f"invalid bind port {port!r}")
    sock.bind((host, port))
    bound_host, _bound_port = sock.getsockname()[:2]
    if bound_host != ALLOWED_BIND_HOST:
        sock.close()
        raise NonLoopbackListenError(
            f"socket bound to {bound_host!r}; only {ALLOWED_BIND_HOST} is allowed"
        )
    return sock


def listen_loopback(
    sock: socket.socket,
    port: int = 0,
    host: Host = ALLOWED_BIND_HOST,
    backlog: int = 1,
) -> socket.socket:
    """Bind and listen on ``127.0.0.1`` only."""
    bind_loopback(sock, port, host)
    sock.listen(backlog)
    return sock


def is_allowed_listen_host(host: str) -> bool:
    """Return True only for a process LISTEN address of 127.0.0.1."""
    return host == ALLOWED_BIND_HOST


def socket_inodes(fd_dir: Union[str, Path] = "/proc/self/fd") -> List[int]:
    """Return socket inodes from ``/proc/<pid>/fd`` (this process only)."""
    fd_path = Path(fd_dir)
    if not fd_path.is_dir():
        return []
    inodes: List[int] = []
    try:
        names = os.listdir(fd_path)
    except OSError:
        return []
    for name in names:
        try:
            target = os.readlink(str(fd_path / name))
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            try:
                inodes.append(int(target[8:-1]))
            except ValueError:
                continue
    return inodes


def parse_proc_net_tcp(
    text: str, ipv6: bool = False
) -> List[Tuple[str, int, int, int]]:
    """Parse ``/proc/net/tcp`` or ``tcp6`` rows.

    Returns ``(host, port, inode, state)`` tuples. No subprocess, no BPF.
    """
    rows: List[Tuple[str, int, int, int]] = []
    for raw_line in text.splitlines()[1:]:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        local = parts[1]
        try:
            state = int(parts[3], 16)
            inode = int(parts[9])
        except ValueError:
            continue
        if ":" not in local:
            continue
        addr_hex, port_hex = local.rsplit(":", 1)
        try:
            port = int(port_hex, 16)
            host = _hex_to_ip(addr_hex, ipv6=ipv6)
        except (ValueError, OSError):
            continue
        rows.append((host, port, inode, state))
    return rows


def iter_process_listening_endpoints(
    fd_dir: Union[str, Path] = "/proc/self/fd",
    tcp_path: Union[str, Path] = "/proc/net/tcp",
    tcp6_path: Union[str, Path] = "/proc/net/tcp6",
) -> List[Tuple[str, int]]:
    """LISTEN endpoints owned by this process (inode-matched, not host-wide)."""
    inodes = set(socket_inodes(fd_dir))
    if not inodes:
        return []
    endpoints: List[Tuple[str, int]] = []
    for path, ipv6 in ((tcp_path, False), (tcp6_path, True)):
        proc_path = Path(path)
        if not proc_path.is_file():
            continue
        try:
            text = proc_path.read_text()
        except OSError:
            continue
        for host, port, inode, state in parse_proc_net_tcp(text, ipv6=ipv6):
            if state == _TCP_LISTEN and inode in inodes:
                endpoints.append((host, port))
    return endpoints


def assert_only_loopback_listeners(
    endpoints: Optional[Sequence[Tuple[str, int]]] = None,
) -> None:
    """Fail if this process listens on 0.0.0.0 or any host other than 127.0.0.1."""
    if endpoints is None:
        endpoints = iter_process_listening_endpoints()
    bad = [(host, port) for host, port in endpoints if not is_allowed_listen_host(host)]
    if bad:
        raise NonLoopbackListenError(
            "non-loopback LISTEN socket(s): "
            + ", ".join(f"{host}:{port}" for host, port in bad)
            + f"; only {ALLOWED_BIND_HOST} is allowed"
        )


def find_wildcard_host_literals(
    root: Union[str, Path],
    skip_names: Iterable[str] = ("bind.py",),
) -> List[str]:
    """Return ``whisper/`` Python files (except the guard) that mention 0.0.0.0."""
    root_path = Path(root)
    skip = frozenset(skip_names)
    offenders: List[str] = []
    for path in sorted(root_path.rglob("*.py")):
        if path.name in skip:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if any(snippet in text for snippet in _WILDCARD_SNIPPETS):
            offenders.append(str(path))
    return offenders


def find_non_loopback_bind_calls(root: Union[str, Path]) -> List[str]:
    """AST-scan for ``bind``/``listen`` calls with a wildcard host literal."""
    root_path = Path(root)
    offenders: List[str] = []
    for path in sorted(root_path.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError):
            continue
        visitor = _BindHostVisitor()
        visitor.visit(tree)
        if visitor.hosts:
            shown = ", ".join(repr(h) for h in visitor.hosts)
            offenders.append(f"{path}: {shown}")
    return offenders


def _hex_to_ip(addr_hex: str, ipv6: bool) -> str:
    if ipv6:
        if len(addr_hex) != 32:
            raise ValueError(addr_hex)
        packed = b"".join(
            struct.pack("<I", int(addr_hex[i : i + 8], 16)) for i in range(0, 32, 8)
        )
        return socket.inet_ntop(socket.AF_INET6, packed)
    if len(addr_hex) != 8:
        raise ValueError(addr_hex)
    packed = struct.pack("<I", int(addr_hex, 16))
    return socket.inet_ntop(socket.AF_INET, packed)


def _call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _constant_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _BindHostVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.hosts: List[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name in _BIND_CALL_NAMES:
            for host in _hosts_from_call(node):
                if host != ALLOWED_BIND_HOST:
                    self.hosts.append(host)
        self.generic_visit(node)


def _hosts_from_call(node: ast.Call) -> List[str]:
    hosts: List[str] = []
    if node.args:
        # socket.bind((host, port)) or listen_loopback(sock, port, host)
        hosts.extend(_hosts_from_expr(node.args[0]))
    if len(node.args) >= 3:
        hosts.extend(_hosts_from_expr(node.args[2]))
    for kw in node.keywords:
        if kw.arg in {"host", "address", "ip"}:
            hosts.extend(_hosts_from_expr(kw.value))
    return hosts


def _hosts_from_expr(node: ast.AST) -> List[str]:
    direct = _constant_str(node)
    if direct is not None:
        return [direct]
    if isinstance(node, (ast.Tuple, ast.List)) and node.elts:
        first = _constant_str(node.elts[0])
        return [first] if first is not None else []
    return []

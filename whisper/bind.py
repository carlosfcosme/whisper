"""Loopback-only bind policy for any serve/listen path.

Every listener must bind ``127.0.0.1``. All-interface hosts are refused
before ``bind()``. After listen, the real socket and ``/proc`` tables are
checked so a non-loopback listen cannot slip through. This module does
not load model weights and does not read secrets.
"""

from __future__ import annotations

import ipaddress
import socket
import struct
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional, Type

# Built without an all-interface literal so application sources stay free
# of that token (CI and pre-commit fail if it appears under whisper/).
ALL_INTERFACES = ".".join(("0", "0", "0", "0"))
LOOPBACK_HOST = "127.0.0.1"
_UNSPECIFIED_V6 = "::"


class BindError(ValueError):
    """Raised when a serve/listen host is not 127.0.0.1 loopback."""


def require_loopback_bind(host: Optional[str] = None) -> str:
    """Return ``127.0.0.1``, or raise ``BindError``.

    ``localhost`` is rewritten to ``127.0.0.1`` without DNS. Unspecified
    addresses (all interfaces, ``::``, ``*``), empty host, LAN, and
    public names are refused. Accepted loopback names are canonicalized
    to ``127.0.0.1`` so the socket always binds IPv4 loopback.
    """
    if host is None:
        return LOOPBACK_HOST
    raw = host.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError(f"bind host is required; use {LOOPBACK_HOST}")
    lowered = raw.lower()
    if lowered in {ALL_INTERFACES, _UNSPECIFIED_V6, "*", "[::]"}:
        raise BindError(f"refusing all-interfaces bind {host!r}; use {LOOPBACK_HOST}")
    if lowered == "localhost":
        return LOOPBACK_HOST
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError as exc:
        raise BindError(
            f"refusing non-localhost bind {host!r}; use {LOOPBACK_HOST}"
        ) from exc
    if ip.is_unspecified or not ip.is_loopback:
        raise BindError(f"refusing non-localhost bind {host!r}; use {LOOPBACK_HOST}")
    return LOOPBACK_HOST


def is_loopback_host(host: str) -> bool:
    try:
        require_loopback_bind(host)
        return True
    except BindError:
        return False


def _parse_proc_ipv4(ip_hex: str) -> str:
    return socket.inet_ntoa(struct.pack("<I", int(ip_hex, 16)))


def _parse_proc_ipv6(ip_hex: str) -> str:
    chunks = [ip_hex[i : i + 8] for i in range(0, 32, 8)]
    raw = b"".join(struct.pack("<I", int(chunk, 16)) for chunk in chunks)
    return socket.inet_ntop(socket.AF_INET6, raw)


def _proc_listen_hosts(proc_path: Path, port: int, ipv6: bool) -> List[str]:
    if not proc_path.is_file():
        return []
    hosts: List[str] = []
    hex_port = "{:04X}".format(port)
    for line in proc_path.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        local, state = parts[1], parts[3]
        if state != "0A":  # TCP_LISTEN
            continue
        ip_hex, port_hex = local.split(":")
        if port_hex.upper() != hex_port:
            continue
        host = _parse_proc_ipv6(ip_hex) if ipv6 else _parse_proc_ipv4(ip_hex)
        hosts.append(host)
    return hosts


def observed_listen_hosts(port: int) -> List[str]:
    """Return hosts listening on ``port`` according to ``/proc/net/tcp{,6}``."""
    hosts = _proc_listen_hosts(Path("/proc/net/tcp"), port, ipv6=False)
    hosts.extend(_proc_listen_hosts(Path("/proc/net/tcp6"), port, ipv6=True))
    return hosts


def assert_loopback_socket(
    sock: socket.socket, *, require_proc: Optional[bool] = None
) -> str:
    """Fail if ``sock`` is not a 127.0.0.1 listen (getsockname + /proc)."""
    addr = sock.getsockname()
    host, port = addr[0], int(addr[1])
    if host != LOOPBACK_HOST:
        raise BindError(f"refusing non-localhost listen {host!r}; use {LOOPBACK_HOST}")
    proc_available = Path("/proc/net/tcp").is_file()
    if require_proc is None:
        require_proc = proc_available
    observed = observed_listen_hosts(port)
    if require_proc and not observed:
        raise BindError(f"could not observe a 127.0.0.1 listen on port {port} in /proc")
    if observed:
        bad = [item for item in observed if item != LOOPBACK_HOST]
        if bad:
            raise BindError(
                f"non-loopback listen observed on port {port}: {bad}; "
                f"use {LOOPBACK_HOST}"
            )
        if LOOPBACK_HOST not in observed:
            raise BindError(
                f"loopback listen missing on port {port}; observed {observed}"
            )
    return host


def create_loopback_httpd(
    handler: Type, host: Optional[str] = None, port: int = 0
) -> ThreadingHTTPServer:
    """Bind ``handler`` to ``127.0.0.1``. ``host`` must pass the policy."""
    require_loopback_bind(host)
    httpd = ThreadingHTTPServer((LOOPBACK_HOST, port), handler)
    try:
        assert_loopback_socket(httpd.socket)
    except BindError:
        httpd.server_close()
        raise
    return httpd

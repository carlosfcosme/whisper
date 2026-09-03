"""Localhost-only bind policy. Listen hosts must be loopback."""

import ipaddress
import socket

BIND_HOST = "127.0.0.1"


def _resolves_to_loopback(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    addrs = {info[4][0] for info in infos}
    if not addrs:
        return False
    try:
        return all(ipaddress.ip_address(addr).is_loopback for addr in addrs)
    except ValueError:
        return False


def serve_bind_host(host=None) -> str:
    """Return a loopback bind address, or raise if the host is not loopback.

    ``None`` defaults to 127.0.0.1. Empty host, wildcard (unspecified), and
    non-loopback addresses are refused.
    """
    if host is None:
        return BIND_HOST
    if not isinstance(host, str):
        raise ValueError(
            f"serve must bind to 127.0.0.1 (got {host!r}); "
            "refusing non-loopback or empty host"
        )
    host = host.strip()
    if host == "":
        raise ValueError("serve must bind to 127.0.0.1 (got ''); refusing empty host")
    if host == "localhost":
        if not _resolves_to_loopback("localhost"):
            raise ValueError(
                "serve must bind to 127.0.0.1; localhost did not resolve to loopback"
            )
        return BIND_HOST
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(
            f"serve must bind to 127.0.0.1 (got {host!r}); "
            "refusing non-loopback or empty host"
        )
    if ip.is_loopback:
        return host
    raise ValueError(
        f"serve must bind to 127.0.0.1 (got {host!r}); "
        "refusing non-loopback or empty host"
    )

"""Helpers to keep unit tests on loopback: no WAN fetch, bind 127.0.0.1 only."""

from ipaddress import ip_address
from urllib.parse import urlparse


class NetworkBlocked(RuntimeError):
    """Raised when a test tries to open a non-loopback connection."""


def is_loopback_host(host):
    if host is None:
        return False
    if not isinstance(host, str):
        return False
    normalized = host.strip().lower().rstrip(".")
    if normalized in {"127.0.0.1", "localhost", "::1"}:
        return True
    if "%" in normalized:
        normalized = normalized.split("%", 1)[0]
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def hostname_from_urlopen_target(url):
    if hasattr(url, "full_url"):
        url = url.full_url
    if isinstance(url, str):
        return urlparse(url).hostname
    return None


def hostname_from_address(address):
    if isinstance(address, tuple) and address:
        return address[0]
    return address

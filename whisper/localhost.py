"""Loopback bind/listen and download URL policy.

Servers bind and listen on 127.0.0.1 only. Hugging Face Hub URLs are
never fetched. Hostnames are not resolved (no DNS rebinding).
"""

import ipaddress
import os
from typing import Optional, Tuple
from urllib.parse import urlparse

LOOPBACK_BIND = "127.0.0.1"
_HUB_HOSTS = frozenset({"huggingface.co", "hf.co", "huggingface.com"})
_OFFLINE_VALUES = frozenset({"1", "true", "yes", "on"})


def bind_host() -> str:
    """Address to bind/listen on. Always IPv4 loopback."""
    return LOOPBACK_BIND


def listen(host: Optional[str] = None, port: int = 0) -> Tuple[str, int]:
    """Return a (host, port) pair. Rejects anything other than 127.0.0.1."""
    chosen = bind_host() if host is None else host
    if chosen != LOOPBACK_BIND:
        raise ValueError(f"bind/listen must be {LOOPBACK_BIND}, not {chosen!r}")
    return (LOOPBACK_BIND, port)


def _hostname(url: str) -> str:
    return (urlparse(url).hostname or "").lower().rstrip(".")


def is_huggingface_hub_url(url: str) -> bool:
    host = _hostname(url)
    if not host:
        return False
    if host in _HUB_HOSTS:
        return True
    return any(host.endswith("." + suffix) for suffix in _HUB_HOSTS)


def is_loopback_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return True
    host = _hostname(url)
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def offline_mode() -> bool:
    return os.getenv("WHISPER_OFFLINE", "").lower() in _OFFLINE_VALUES


def check_download_url(url: str, *, cache_hit: bool = False) -> None:
    """Raise if this URL must not be fetched.

    Hugging Face Hub is always refused. In WHISPER_OFFLINE=1, remote
    (non-loopback) fetches are refused; a local cache hit is not a fetch.
    """
    if is_huggingface_hub_url(url):
        raise RuntimeError(
            "Hugging Face Hub downloads are disabled; use a local checkpoint path"
        )
    if cache_hit:
        return
    if offline_mode() and not is_loopback_url(url):
        raise RuntimeError(
            "WHISPER_OFFLINE=1 refuses remote downloads; use a local checkpoint "
            "or a 127.0.0.1 URL"
        )

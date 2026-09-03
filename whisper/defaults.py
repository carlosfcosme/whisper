"""Ticket 4 defaults: CPU-only, offline, no-store."""

from __future__ import annotations

import os
from ipaddress import ip_address
from urllib.parse import urlparse

DEFAULT_DEVICE = "cpu"
DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_OFFLINE = True
DEFAULT_NO_STORE = True

_FALSEY = frozenset({"0", "false", "no", "off"})
_HF_HUB_HOSTS = frozenset({"huggingface.co", "hf.co"})
_HF_HUB_SUFFIXES = (".huggingface.co", ".hf.co")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSEY


def offline_enabled() -> bool:
    """Remote weight fetches are off by default. Opt in with WHISPER_OFFLINE=0."""
    if os.getenv("WHISPER_OFFLINE") is not None:
        return _env_flag("WHISPER_OFFLINE", DEFAULT_OFFLINE)
    if os.getenv("HF_HUB_OFFLINE") is not None:
        return _env_flag("HF_HUB_OFFLINE", DEFAULT_OFFLINE)
    return DEFAULT_OFFLINE


def no_store_enabled() -> bool:
    """Do not persist checkpoints. Opt in to storing with WHISPER_NO_STORE=0."""
    return _env_flag("WHISPER_NO_STORE", DEFAULT_NO_STORE)


def is_hf_hub_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    return host in _HF_HUB_HOSTS or host.endswith(_HF_HUB_SUFFIXES)


def is_loopback_bind_host(host: str) -> bool:
    if host is None:
        return False
    normalized = str(host).strip().lower().rstrip(".")
    if not normalized:
        return False
    if normalized in {"127.0.0.1", "localhost", "::1"}:
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def require_loopback_host(host: str = DEFAULT_BIND_HOST) -> str:
    """Return a loopback host, or raise ValueError for wildcard / public binds."""
    if is_loopback_bind_host(host):
        return DEFAULT_BIND_HOST if host in (None, "") else str(host).strip()
    shown = host if host else "<wildcard>"
    raise ValueError(f"refusing bind host {shown!r}; use {DEFAULT_BIND_HOST}")

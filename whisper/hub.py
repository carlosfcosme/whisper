"""Refuse Hugging Face Hub fetches and default remote weight downloads.

Named checkpoints are loaded from a local cache or an explicit file path.
Hugging Face Hub hosts are never contacted. Other remote fetches require
``WHISPER_ALLOW_WEIGHT_FETCH=1``.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

ALLOW_FETCH_ENV = "WHISPER_ALLOW_WEIGHT_FETCH"
_TRUTHY = {"1", "true", "yes", "on"}

HUB_HOSTS = frozenset(
    {
        "huggingface.co",
        "hf.co",
        "hub.huggingface.co",
        "cdn-lfs.huggingface.co",
        "cdn-lfs-us-1.huggingface.co",
        "cas-bridge.xethub.hf.co",
    }
)


class HubError(RuntimeError):
    """Raised when a Hugging Face Hub URL is refused."""


class WeightDownloadError(RuntimeError):
    """Raised when a remote weight download is refused on a cache miss."""


def _hostname(url: str) -> str:
    return (urlparse(url).hostname or "").lower().rstrip(".")


def is_hub_host(host: str) -> bool:
    normalized = (host or "").lower().rstrip(".")
    if not normalized:
        return False
    if normalized in HUB_HOSTS:
        return True
    return normalized.endswith(".huggingface.co") or normalized.endswith(".hf.co")


def is_hub_url(url: str) -> bool:
    return is_hub_host(_hostname(url))


def refuse_hub(url: str) -> None:
    """Always refuse Hugging Face Hub URLs (No Hub)."""
    if not is_hub_url(url):
        return
    host = _hostname(url) or "<unknown>"
    raise HubError(
        f"Refusing Hugging Face Hub pull from host {host!r}. "
        "Load a local checkpoint path instead."
    )


def allow_weight_fetch() -> bool:
    return os.environ.get(ALLOW_FETCH_ENV, "").strip().lower() in _TRUTHY


def refuse_weight_download(url: str) -> None:
    """Refuse a cache-miss network fetch unless an explicit opt-in is set.

    Hugging Face Hub hosts are refused even when the opt-in is set.
    """
    refuse_hub(url)
    if allow_weight_fetch():
        return
    host = _hostname(url) or "<unknown>"
    raise WeightDownloadError(
        f"Refusing weight download from host {host!r}. "
        f"Pass a local checkpoint path or set {ALLOW_FETCH_ENV}=1. "
        "Tests and CI must not Hub."
    )

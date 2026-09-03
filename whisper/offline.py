"""Offline-by-default weight loading. No Hugging Face Hub. Download unused.

The default path never opens a network connection for checkpoints.
Hugging Face Hub hosts are refused even when offline mode is disabled.
A usage counter records actual ``urlopen`` weight fetches so CI can
``assert_download_unused()``.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Any
from urllib.parse import urlparse

OFFLINE_ENV = "WHISPER_OFFLINE"
_FALSEY = {"0", "false", "no", "off"}

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

_network_download_calls = 0


class HubError(RuntimeError):
    """Raised when a Hugging Face Hub URL is refused."""


class OfflineError(RuntimeError):
    """Raised when a weight download is refused on the offline path."""


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
    """Always refuse Hugging Face Hub URLs (no Hub by default)."""
    if not is_hub_url(url):
        return
    host = _hostname(url) or "<unknown>"
    raise HubError(
        f"Refusing Hugging Face Hub pull from host {host!r}. "
        "Load a local checkpoint path instead."
    )


def is_offline() -> bool:
    """True unless ``WHISPER_OFFLINE`` is an explicit falsey value.

    Unset defaults to offline so the default path does not fetch weights.
    """
    raw = os.environ.get(OFFLINE_ENV, "1").strip().lower()
    return raw not in _FALSEY


def network_download_calls() -> int:
    return _network_download_calls


def reset_download_usage() -> None:
    global _network_download_calls
    _network_download_calls = 0


def assert_download_unused(context: str = "") -> None:
    """Fail if a weight ``urlopen`` ran. Cache hits do not count."""
    if _network_download_calls:
        suffix = f" ({context})" if context else ""
        raise AssertionError(
            f"weight download used {_network_download_calls} time(s){suffix}"
        )


def refuse_default_fetch(url: str) -> None:
    """Block Hub always; block every remote fetch while offline."""
    refuse_hub(url)
    if not is_offline():
        return
    host = _hostname(url) or "<unknown>"
    raise OfflineError(
        f"Refusing weight download from host {host!r} (offline path). "
        f"Pass a local checkpoint or set {OFFLINE_ENV}=0. "
        "Default CI/tests must not Hub or pull weights."
    )


def urlopen_weight(url: str, *args: Any, **kwargs: Any):
    """Network entry for checkpoints. Records usage only if urlopen runs."""
    refuse_default_fetch(url)
    global _network_download_calls
    _network_download_calls += 1
    return urllib.request.urlopen(url, *args, **kwargs)

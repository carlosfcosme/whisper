"""Offline weight-pull policy: never hit Hugging Face Hub.

Official checkpoints are on Azure CDN. Hub hosts are always refused.
When ``WHISPER_NO_WEIGHT_DOWNLOAD``, ``HF_HUB_OFFLINE``, or ``CI`` is
set, every cache-miss network pull is refused. A cache hit is a local
file read, not a download.

This module does not load kernels and does not read secrets.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

NO_WEIGHT_DOWNLOAD_ENV = "WHISPER_NO_WEIGHT_DOWNLOAD"
HUB_OFFLINE_ENV = "HF_HUB_OFFLINE"
_TRUTHY = {"1", "true", "yes", "on"}
_HF_HUB_HOSTS = {"huggingface.co", "hf.co"}
_HF_HUB_SUFFIXES = (".huggingface.co", ".hf.co")


class WeightDownloadError(RuntimeError):
    """Raised when a model-weight network pull is refused."""


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def is_hf_hub_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    return host in _HF_HUB_HOSTS or host.endswith(_HF_HUB_SUFFIXES)


def weight_network_pull_allowed() -> bool:
    """False in CI / offline tests; True for a normal cache-miss CDN pull."""
    if _env_truthy(NO_WEIGHT_DOWNLOAD_ENV) or _env_truthy(HUB_OFFLINE_ENV):
        return False
    if _env_truthy("CI"):
        return False
    return True


def refuse_weight_network_pull(url: str) -> None:
    """Refuse Hub always. Refuse every network pull when CI/offline."""
    if is_hf_hub_url(url):
        host = urlparse(url).hostname or "<unknown>"
        raise WeightDownloadError(
            f"Refusing Hugging Face Hub weight pull from host {host!r}. "
            "Tests and CI must not contact the Hub."
        )
    if weight_network_pull_allowed():
        return
    raise WeightDownloadError(
        "Weight download is disabled (WHISPER_NO_WEIGHT_DOWNLOAD, "
        "HF_HUB_OFFLINE, or CI). Pass a local checkpoint path."
    )

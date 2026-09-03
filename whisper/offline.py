"""Offline-by-default weight loading: no Hub fetch, no auto-download.

The default ``load_model`` path must not contact the Hugging Face Hub or
pull checkpoints from the network. Pass a local file, or set
``WHISPER_ALLOW_WEIGHT_DOWNLOAD=1`` to opt into non-Hub URLs.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

OFFLINE_ENV = "WHISPER_OFFLINE"
NO_WEIGHT_DOWNLOAD_ENV = "WHISPER_NO_WEIGHT_DOWNLOAD"
ALLOW_WEIGHT_DOWNLOAD_ENV = "WHISPER_ALLOW_WEIGHT_DOWNLOAD"
HF_HUB_OFFLINE_ENV = "HF_HUB_OFFLINE"
TRANSFORMERS_OFFLINE_ENV = "TRANSFORMERS_OFFLINE"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_HF_HUB_HOSTS = frozenset({"huggingface.co", "hf.co"})
_HF_HUB_SUFFIXES = (".huggingface.co", ".hf.co")


class WeightDownloadError(RuntimeError):
    """Raised when an automatic model-weight pull is refused."""


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def is_hf_hub_url(url: str) -> bool:
    """True for Hugging Face Hub hosts (``huggingface.co``, ``hf.co``, subdomains)."""
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    return host in _HF_HUB_HOSTS or host.endswith(_HF_HUB_SUFFIXES)


def weight_auto_download_allowed() -> bool:
    """Whether a cache-miss may open a non-Hub URL.

    Default is ``False`` (offline). ``WHISPER_ALLOW_WEIGHT_DOWNLOAD=1`` is
    the only opt-in. ``WHISPER_OFFLINE`` / ``WHISPER_NO_WEIGHT_DOWNLOAD``
    win over the allow flag. Hub URLs are refused regardless.
    """
    if _env_truthy(OFFLINE_ENV) or _env_truthy(NO_WEIGHT_DOWNLOAD_ENV):
        return False
    if _env_truthy(ALLOW_WEIGHT_DOWNLOAD_ENV):
        return True
    return False


def refuse_weight_auto_download(url: str) -> None:
    """Refuse Hub URLs always, and all cache-miss pulls unless explicitly allowed."""
    if is_hf_hub_url(url):
        host = urlparse(url).hostname or "<unknown>"
        raise WeightDownloadError(
            f"Refusing Hugging Face Hub weight pull from host {host!r}. "
            "Pass a local checkpoint path to load_model()."
        )
    if weight_auto_download_allowed():
        return
    raise WeightDownloadError(
        "Auto-download of model weights is disabled by default. "
        "Pass a local checkpoint path to load_model(), or set "
        f"{ALLOW_WEIGHT_DOWNLOAD_ENV}=1 to opt into non-Hub URLs."
    )

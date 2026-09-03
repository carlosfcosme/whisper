"""Offline defaults: no weight fetch, no Hugging Face Hub."""

import os
from typing import Iterable
from urllib.parse import urlparse

# Applied at import unless the caller already set a value.
OFFLINE_ENV_DEFAULTS = {
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}

# Official Whisper weights are Azure CDN, not the Hub. These hosts are refused.
BLOCKED_HUB_HOSTS = (
    "huggingface.co",
    "hf.co",
)


class OfflineDownloadError(RuntimeError):
    """Raised when a checkpoint is missing or a Hub/WAN fetch is refused."""


def apply_offline_env(environ=None) -> None:
    """Set Hugging Face / transformers offline flags if unset."""
    target = os.environ if environ is None else environ
    for key, value in OFFLINE_ENV_DEFAULTS.items():
        target.setdefault(key, value)


def _host(url: str) -> str:
    netloc = urlparse(url).netloc.lower().split("@")[-1]
    if netloc.startswith("[") and "]" in netloc:
        return netloc[1 : netloc.index("]")]
    return netloc.rsplit(":", 1)[0]


def is_blocked_hub_url(url: str, hosts: Iterable[str] = BLOCKED_HUB_HOSTS) -> bool:
    host = _host(url)
    if not host:
        return False
    for blocked in hosts:
        if host == blocked or host.endswith("." + blocked):
            return True
    return False


def refuse_hub_url(url: str) -> None:
    """Raise if ``url`` points at Hugging Face Hub (or a Hub CDN)."""
    if is_blocked_hub_url(url):
        raise OfflineDownloadError(
            f"Hugging Face Hub downloads are disabled (offline default): {url}"
        )


apply_offline_env()

"""Refuse Hub URLs and optional network fetches of model checkpoints."""

from __future__ import annotations

import os
from urllib.parse import urlparse

HUB_HOSTS = ("huggingface.co", "hf.co")
NO_DOWNLOAD_ENV = "WHISPER_NO_DOWNLOAD"


class HubDisabledError(RuntimeError):
    """Raised when a download URL points at Hugging Face Hub."""


class DownloadBlockedError(RuntimeError):
    """Raised when a checkpoint fetch is blocked (offline / CI)."""


def downloads_blocked() -> bool:
    return os.getenv(NO_DOWNLOAD_ENV, "").strip().lower() in {"1", "true", "yes"}


def assert_not_hub_url(url: str) -> None:
    """Raise ``HubDisabledError`` if ``url`` is a Hugging Face Hub address."""
    if not url:
        raise HubDisabledError("empty download URL")
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if any(host == blocked or host.endswith("." + blocked) for blocked in HUB_HOSTS):
        raise HubDisabledError(f"Hugging Face Hub downloads are disabled: {url}")


def assert_can_fetch(url: str) -> None:
    """Reject Hub URLs always, and all fetches when ``WHISPER_NO_DOWNLOAD`` is set."""
    assert_not_hub_url(url)
    if downloads_blocked():
        raise DownloadBlockedError(
            f"Model download blocked ({NO_DOWNLOAD_ENV}=1): {url}"
        )

"""Refuse Hugging Face Hub weight pulls.

Whisper loads official checkpoints from the OpenAI CDN or a local path.
It does not download from the Hugging Face Hub (``huggingface.co`` / ``hf.co``).
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

HUB_HOSTS = frozenset(
    {
        "huggingface.co",
        "www.huggingface.co",
        "hf.co",
        "www.hf.co",
    }
)
HUB_HOST_SUFFIXES = (".huggingface.co", ".hf.co")


class HubPullError(RuntimeError):
    """Raised when a Hugging Face Hub weight URL is refused."""


def hostname_is_hub(host: Optional[str]) -> bool:
    """True for Hugging Face Hub hostnames (no DNS)."""
    if not host:
        return False
    normalized = host.strip().lower().rstrip(".")
    if normalized in HUB_HOSTS:
        return True
    return any(normalized.endswith(suffix) for suffix in HUB_HOST_SUFFIXES)


def is_hub_url(url: str) -> bool:
    """True when ``url`` targets the Hugging Face Hub."""
    return hostname_is_hub(urlparse(url).hostname)


def refuse_hub_pull(url: str) -> None:
    """Raise ``HubPullError`` when ``url`` is a Hub weight location."""
    if not is_hub_url(url):
        return
    host = urlparse(url).hostname or "<unknown>"
    raise HubPullError(
        f"Refusing Hugging Face Hub pull from host {host!r}. "
        "Whisper does not download weights from the Hub."
    )

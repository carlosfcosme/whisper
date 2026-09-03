"""Refuse Hugging Face Hub checkpoint URLs. Official Azure URLs stay allowed."""

from __future__ import annotations

from urllib.parse import urlparse

HUB_HOSTS = ("huggingface.co", "hf.co")


class HubDisabledError(RuntimeError):
    """Raised when a download URL points at Hugging Face Hub."""


def assert_not_hub_url(url: str) -> None:
    """Raise ``HubDisabledError`` if ``url`` is a Hugging Face Hub address."""
    if not url:
        raise HubDisabledError("empty download URL")
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if any(host == blocked or host.endswith("." + blocked) for blocked in HUB_HOSTS):
        raise HubDisabledError(f"Hugging Face Hub downloads are disabled: {url}")

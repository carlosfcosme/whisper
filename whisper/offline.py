"""Offline / Cloud Agent defaults: CPU inference, no Hub, loopback bind."""

from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse

DEFAULT_DEVICE = "cpu"
BIND_HOST = "127.0.0.1"
BIND_PORT = 8765

HUB_HOST_MARKERS = (
    "huggingface.co",
    "hf.co",
    "huggingface.com",
)

OFFLINE_ENV_VARS = (
    "WHISPER_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)

_TRUTHY = {"1", "true", "yes", "on"}


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def offline_enabled() -> bool:
    return any(env_flag(key) for key in OFFLINE_ENV_VARS)


def is_hub_host(host: Optional[str]) -> bool:
    if not host:
        return False
    lowered = host.strip().lower().rstrip(".")
    return any(
        lowered == marker or lowered.endswith("." + marker)
        for marker in HUB_HOST_MARKERS
    )


def is_hub_url(url: str) -> bool:
    parsed = urlparse(url or "")
    if is_hub_host(parsed.hostname):
        return True
    # Hugging Face Hub also uses path-style / hf-mirror hosts in some setups.
    lowered = (url or "").lower()
    return any(marker in lowered for marker in HUB_HOST_MARKERS)


def refuse_remote_download(url: str, dest: str) -> None:
    """Raise if a Hub or offline remote weight pull is forbidden.

    Hub URLs are always refused. When WHISPER_OFFLINE / HF_HUB_OFFLINE is
    set, any remaining remote pull is also refused (use a local checkpoint).
    """
    if is_hub_url(url):
        raise RuntimeError(
            "no Hub: refusing Hugging Face Hub download ({}); "
            "use a local checkpoint".format(url)
        )
    if offline_enabled():
        raise RuntimeError(
            "offline: no weight pulls; missing local checkpoint {}".format(dest)
        )

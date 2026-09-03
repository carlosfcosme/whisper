"""Sovereign defaults: offline weight path, no Hugging Face Hub, CPU, loopback.

Weight pulls are offline by default. Hugging Face Hub URLs are always
refused. There is no commercial download flag and no API-key path.
"""

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

# Reinforced by CI. Library import also setdefaults these so a bare
# interpreter is Hub-offline without a workflow file.
OFFLINE_ENV_VARS = (
    "WHISPER_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)

# The only documented opt-in for a non-Hub CDN pull. Not a commercial door.
ALLOW_WEIGHT_DOWNLOAD_ENV = "WHISPER_ALLOW_WEIGHT_DOWNLOAD"

_TRUTHY = {"1", "true", "yes", "on"}


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _apply_default_offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


_apply_default_offline_env()


def offline_enabled() -> bool:
    """Return True unless an explicit non-Hub weight-download opt-in is set.

    Offline is the default. Unsetting HF_HUB_OFFLINE / WHISPER_OFFLINE does
    not open a download path. ``WHISPER_ALLOW_WEIGHT_DOWNLOAD=1`` is the
    only opt-in, and it never authorizes Hugging Face Hub.
    """
    if env_flag(ALLOW_WEIGHT_DOWNLOAD_ENV):
        return False
    return True


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
    lowered = (url or "").lower()
    return any(marker in lowered for marker in HUB_HOST_MARKERS)


def refuse_remote_download(url: str, dest: str) -> None:
    """Refuse Hub always. Refuse other remote weight pulls when offline.

    Cache hits never call this. A missing local file is not fetched from
    Hugging Face Hub or, by default, from any other host.
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

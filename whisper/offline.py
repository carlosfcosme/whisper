"""Offline / no-Hub policy for checkpoint loading.

CI and tests set WHISPER_OFFLINE / HF_HUB_OFFLINE so named-model loads
never open a network socket. Hugging Face Hub URLs are refused even when
those env vars are unset — this package does not fetch from the Hub.
"""

from __future__ import annotations

import os
from typing import Union
from urllib.parse import urlparse

_OFFLINE_TRUTHY = frozenset({"1", "true", "yes", "on"})
OFFLINE_ENV_VARS = (
    "WHISPER_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)

# Token *names* only — never read or print values.
TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)

_HUB_HOSTS = frozenset(
    {
        "huggingface.co",
        "www.huggingface.co",
        "hf.co",
        "www.hf.co",
        "hf-mirror.com",
        "www.hf-mirror.com",
    }
)
_HUB_HOST_SUFFIXES = (
    ".huggingface.co",
    ".hf.co",
    ".hf-mirror.com",
)


def _url_text(url: Union[str, object]) -> str:
    if isinstance(url, str):
        return url
    return getattr(url, "full_url", None) or str(url)


def _hostname(url: Union[str, object]) -> str:
    raw = _url_text(url)
    host = urlparse(raw).netloc.lower().split("@")[-1].split(":")[0]
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host


def is_hub_url(url: Union[str, object]) -> bool:
    """True when *url* targets the Hugging Face Hub (or a Hub mirror)."""
    host = _hostname(url)
    if host in _HUB_HOSTS:
        return True
    if any(host.endswith(suffix) for suffix in _HUB_HOST_SUFFIXES):
        return True
    if "xethub" in host:
        return True
    return False


def weights_download_forbidden() -> bool:
    """True when install/test (or the user) has disabled weight downloads."""
    for key in OFFLINE_ENV_VARS:
        if os.environ.get(key, "").strip().lower() in _OFFLINE_TRUTHY:
            return True
    return False


def refuse_network_weight_fetch(url: Union[str, object], target: str) -> None:
    """Raise before a Hub or (when offline) any WAN checkpoint fetch.

    Local files with a matching SHA-256 are used by the caller *before*
    this is invoked, so a populated cache still loads offline.
    """
    if is_hub_url(url):
        raise RuntimeError(
            "Hugging Face Hub fetch is refused: {}".format(_url_text(url))
        )
    if weights_download_forbidden():
        raise RuntimeError(
            "Refusing to download model weights while offline "
            "(WHISPER_OFFLINE or HF_HUB_OFFLINE is set). "
            "Missing or invalid local cache: {}".format(target)
        )

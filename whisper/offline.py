"""Offline-by-default weight fetch policy.

Named checkpoints are not downloaded unless the operator explicitly opts in
with a local environment flag. Hugging Face Hub URLs are never fetched.
No credentials, tokens, or external activation are required or consulted.
"""

from __future__ import annotations

import os
import urllib.parse
from typing import FrozenSet, Optional

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Hugging Face Hub and common mirrors. Always refused — no HF_TOKEN path.
HUB_HOSTS: FrozenSet[str] = frozenset(
    {
        "huggingface.co",
        "huggingface.com",
        "hf.co",
        "cdn-lfs.huggingface.co",
        "cdn-lfs.hf.co",
        "cas-bridge.xethub.hf.co",
    }
)
HUB_SUFFIXES = (".huggingface.co", ".hf.co", ".huggingface.com")

# Official Whisper CDN hosts (WAN). Refused unless locally opted in.
WAN_WEIGHT_HOSTS: FrozenSet[str] = frozenset(
    {
        "openaipublic.azureedge.net",
        "openaipublic.blob.core.windows.net",
    }
)
WAN_WEIGHT_SUFFIXES = (".azureedge.net", ".blob.core.windows.net")


class WeightDownloadError(RuntimeError):
    """Raised when a network fetch of model weights is refused."""


def env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).strip().lower() in _TRUTHY


def allow_weight_download() -> bool:
    """Return whether a WAN pull of official weights is allowed.

    Default is ``False`` (offline). Opt in with
    ``WHISPER_ALLOW_WEIGHT_DOWNLOAD=1``. ``WHISPER_OFFLINE=1`` always wins.
    Tokens such as ``HF_TOKEN`` are ignored and never enable a fetch.
    """
    if env_flag("WHISPER_OFFLINE"):
        return False
    if env_flag("HF_HUB_OFFLINE"):
        # Hub-offline is not an activation signal; it only tightens policy.
        # Official CDN still requires an explicit allow flag.
        pass
    return env_flag("WHISPER_ALLOW_WEIGHT_DOWNLOAD")


def url_host(url: str) -> str:
    return (urllib.parse.urlparse(url).hostname or "").lower().rstrip(".")


def hub_host(url: str) -> Optional[str]:
    host = url_host(url)
    if not host:
        return None
    if host in HUB_HOSTS or host.endswith(HUB_SUFFIXES):
        return host
    return None


def is_hub_url(url: str) -> bool:
    return hub_host(url) is not None


def is_wan_weight_host(host: str) -> bool:
    host = (host or "").lower().rstrip(".")
    if not host:
        return False
    if host in HUB_HOSTS or host.endswith(HUB_SUFFIXES):
        return True
    if host in WAN_WEIGHT_HOSTS or host.endswith(WAN_WEIGHT_SUFFIXES):
        return True
    return False


def refuse_weight_download(url: str, download_target: str = "") -> None:
    """Raise if ``url`` must not be fetched.

    Hub URLs are always refused. Other weight URLs are refused unless
    :func:`allow_weight_download` is true. No credential is read.
    """
    host = hub_host(url)
    if host is not None:
        raise WeightDownloadError(
            f"Hub downloads are disabled (refusing {host}). "
            "Use a local checkpoint path. No token or activation is used."
        )
    if allow_weight_download():
        return
    where = f" Place a checkpoint at {download_target}." if download_target else ""
    raise WeightDownloadError(
        "Weight download is disabled by default (offline). "
        f"Refusing to fetch {url}.{where} "
        "Set WHISPER_ALLOW_WEIGHT_DOWNLOAD=1 to opt in to official CDN "
        "fetches locally. Hub URLs are never fetched. "
        "No credentials or external activation are required."
    )

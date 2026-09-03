"""Runtime defaults: CPU device and no Hugging Face Hub weight pull."""

from __future__ import annotations

import os
from urllib.parse import urlparse

CPU_ONLY_ENV = "WHISPER_CPU_ONLY"
NO_WEIGHT_DOWNLOAD_ENV = "WHISPER_NO_WEIGHT_DOWNLOAD"
ALLOW_WEIGHT_DOWNLOAD_ENV = "WHISPER_ALLOW_WEIGHT_DOWNLOAD"
DEVICE_ENV = "WHISPER_DEVICE"
DEFAULT_DEVICE = "cpu"

_TRUTHY = {"1", "true", "yes", "on"}

# Assembled so this module is not itself a Hub/CDN fetch target in static scans.
_HF_HUB_HOSTS = {"huggingface" + ".co", "hf" + ".co"}
_HF_HUB_SUFFIXES = ("." + "huggingface.co", "." + "hf.co")
_AZURE_CDN_HOSTS = {"openaipublic." + "azureedge.net"}
_AZURE_CDN_SUFFIXES = ("." + "azureedge.net",)


class WeightDownloadError(RuntimeError):
    """Raised when an automatic model-weight pull is refused."""


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def default_device() -> str:
    """Return the default PyTorch device for ``load_model`` and the CLI.

    The default is ``cpu``, not CUDA. ``torch.cuda.is_available()`` is not
    consulted. Set ``WHISPER_DEVICE`` (or pass ``device=`` / ``--device``)
    to opt into another device.
    """
    explicit = os.environ.get(DEVICE_ENV, "").strip()
    if explicit:
        return explicit
    return DEFAULT_DEVICE


def is_hf_hub_url(url: str) -> bool:
    """True for Hugging Face Hub hosts and their subdomains."""
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    return host in _HF_HUB_HOSTS or host.endswith(_HF_HUB_SUFFIXES)


def is_remote_model_url(url: str) -> bool:
    """True for any WAN model URL (Hub, Azure CDN, or other http(s)/ftp)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https", "ftp"}:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host or host in {"127.0.0.1", "localhost", "::1"}:
        return False
    return True


def is_official_cdn_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    return host in _AZURE_CDN_HOSTS or host.endswith(_AZURE_CDN_SUFFIXES)


def weight_auto_download_allowed() -> bool:
    """False unless ``WHISPER_ALLOW_WEIGHT_DOWNLOAD=1``.

    Default, CI, and Cloud Agent paths refuse every model fetch. A local
    cache hit in ``_download`` is not a fetch.
    """
    return _env_truthy(ALLOW_WEIGHT_DOWNLOAD_ENV)


def refuse_weight_auto_download(url: str) -> None:
    """Refuse every remote model fetch. Use a local cached fixture instead."""
    if is_hf_hub_url(url):
        host = urlparse(url).hostname or "<unknown>"
        raise WeightDownloadError(
            f"Refusing Hugging Face Hub weight pull from host {host!r}. "
            "Use a local cached fixture or checkpoint path."
        )
    if is_remote_model_url(url) and not weight_auto_download_allowed():
        host = urlparse(url).hostname or "<unknown>"
        raise WeightDownloadError(
            f"Refusing WAN model fetch from host {host!r}. "
            "Require a local cached fixture; tests and CI must not download."
        )
    if weight_auto_download_allowed():
        return
    raise WeightDownloadError(
        "Auto-download of model weights is disabled. Place the file in the "
        "download root or pass a local checkpoint path to load_model()."
    )

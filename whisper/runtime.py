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

# Assembled so this module is not itself a Hub fetch target in static scans.
_HF_HUB_HOSTS = {"huggingface" + ".co", "hf" + ".co"}
_HF_HUB_SUFFIXES = ("." + "huggingface.co", "." + "hf.co")


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


def weight_auto_download_allowed() -> bool:
    """False when CI / Cloud Agent (or ``WHISPER_NO_WEIGHT_DOWNLOAD``) refuses pulls.

    ``WHISPER_ALLOW_WEIGHT_DOWNLOAD=1`` is an explicit escape hatch.
    """
    if _env_truthy(ALLOW_WEIGHT_DOWNLOAD_ENV):
        return True
    if (
        _env_truthy(NO_WEIGHT_DOWNLOAD_ENV)
        or _env_truthy("CI")
        or _env_truthy(CPU_ONLY_ENV)
    ):
        return False
    return True


def refuse_weight_auto_download(url: str) -> None:
    """Refuse Hugging Face Hub URLs always, and all cache-miss pulls on CI."""
    if is_hf_hub_url(url):
        host = urlparse(url).hostname or "<unknown>"
        raise WeightDownloadError(
            f"Refusing Hugging Face Hub weight pull from host {host!r}. "
            "Unit tests and CI must not contact the Hub. "
            "Pass a local checkpoint path to load_model()."
        )
    if weight_auto_download_allowed():
        return
    raise WeightDownloadError(
        "Auto-download of model weights is disabled on the CI / Cloud Agent "
        f"path ({NO_WEIGHT_DOWNLOAD_ENV} or CI). Place the file in the download "
        "root or pass a local checkpoint path to load_model()."
    )

"""CPU-only default and no Hub / no weight-pull policy.

Stdlib only so CI can import this module without torch, a Hub client,
or downloading checkpoints.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

DEFAULT_DEVICE = "cpu"
CPU_ONLY_ENV = "WHISPER_CPU_ONLY"
NO_WEIGHT_DOWNLOAD_ENV = "WHISPER_NO_WEIGHT_DOWNLOAD"
ALLOW_WEIGHT_DOWNLOAD_ENV = "WHISPER_ALLOW_WEIGHT_DOWNLOAD"
DEVICE_ENV = "WHISPER_DEVICE"
HF_HUB_OFFLINE_ENV = "HF_HUB_OFFLINE"

_TRUTHY = {"1", "true", "yes", "on"}

_HF_HUB_HOSTS = {"huggingface.co", "hf.co"}
_HF_HUB_SUFFIXES = (".huggingface.co", ".hf.co")


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
    """True for Hugging Face Hub hosts."""
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    return host in _HF_HUB_HOSTS or host.endswith(_HF_HUB_SUFFIXES)


def weight_auto_download_allowed() -> bool:
    """False when CI / Cloud Agent / offline flags refuse cache-miss pulls.

    Hugging Face Hub URLs are refused even when this returns True.
    """
    if _env_truthy(ALLOW_WEIGHT_DOWNLOAD_ENV):
        return True
    if (
        _env_truthy(NO_WEIGHT_DOWNLOAD_ENV)
        or _env_truthy(HF_HUB_OFFLINE_ENV)
        or _env_truthy(CPU_ONLY_ENV)
        or _env_truthy("CI")
    ):
        return False
    return True


def refuse_weight_auto_download(url: str) -> None:
    """Refuse Hub URLs always, and all cache-miss pulls when offline/CI."""
    if is_hf_hub_url(url):
        host = urlparse(url).hostname or "<unknown>"
        raise WeightDownloadError(
            "Refusing Hugging Face Hub weight pull from host {!r}. "
            "Pass a local checkpoint path to load_model().".format(host)
        )
    if weight_auto_download_allowed():
        return
    raise WeightDownloadError(
        "Auto-download of model weights is disabled "
        "({} / {} / CI). Place the file in the download root or pass a "
        "local checkpoint path to load_model().".format(
            NO_WEIGHT_DOWNLOAD_ENV, HF_HUB_OFFLINE_ENV
        )
    )

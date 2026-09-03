"""Cloud Agent / CI runtime: CPU-only default and no weight auto-download.

Whisper used to pick CUDA vs CPU implicitly via ``torch.cuda.is_available()``
and fetched checkpoints on a cache miss. On the Cloud Agent and CI path the
default device is ``cpu`` and auto-download of model weights is refused,
including any Hugging Face Hub URL.
"""

import os
from urllib.parse import urlparse

import torch

CPU_ONLY_ENV = "WHISPER_CPU_ONLY"
NO_WEIGHT_DOWNLOAD_ENV = "WHISPER_NO_WEIGHT_DOWNLOAD"
ALLOW_WEIGHT_DOWNLOAD_ENV = "WHISPER_ALLOW_WEIGHT_DOWNLOAD"
DEVICE_ENV = "WHISPER_DEVICE"

_TRUTHY = {"1", "true", "yes", "on"}

_HF_HUB_HOSTS = {"huggingface.co", "hf.co"}
_HF_HUB_SUFFIXES = (".huggingface.co", ".hf.co")


class WeightDownloadError(RuntimeError):
    """Raised when an automatic model-weight pull is refused."""


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def cloud_agent_or_ci_path() -> bool:
    """True on the Cloud Agent / CI path (``WHISPER_CPU_ONLY`` or ``CI``)."""
    return _env_truthy(CPU_ONLY_ENV) or _env_truthy("CI")


def default_device() -> str:
    """Return the default PyTorch device for ``load_model`` and the CLI.

    ``WHISPER_DEVICE`` wins when set. On the Cloud Agent / CI path the
    default is ``cpu`` even if CUDA is visible. Otherwise the historical
    ``cuda``-if-available fallback is used.
    """
    explicit = os.environ.get(DEVICE_ENV, "").strip()
    if explicit:
        return explicit
    if cloud_agent_or_ci_path():
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_hf_hub_url(url: str) -> bool:
    """True for Hugging Face Hub hosts (``huggingface.co``, ``hf.co``, and subdomains)."""
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    return host in _HF_HUB_HOSTS or host.endswith(_HF_HUB_SUFFIXES)


def weight_auto_download_allowed() -> bool:
    """False when Cloud Agent / CI (or ``WHISPER_NO_WEIGHT_DOWNLOAD``) refuses pulls.

    ``WHISPER_ALLOW_WEIGHT_DOWNLOAD=1`` is an explicit escape hatch.
    """
    if _env_truthy(ALLOW_WEIGHT_DOWNLOAD_ENV):
        return True
    if _env_truthy(NO_WEIGHT_DOWNLOAD_ENV) or cloud_agent_or_ci_path():
        return False
    return True


def refuse_weight_auto_download(url: str) -> None:
    """Refuse Hugging Face Hub URLs always, and all cache-miss pulls on CI/Cloud Agent."""
    if is_hf_hub_url(url):
        host = urlparse(url).hostname or "<unknown>"
        raise WeightDownloadError(
            f"Refusing Hugging Face Hub weight pull from host {host!r}. "
            "Unit tests and the Cloud Agent / CI path must not contact the Hub. "
            "Pass a local checkpoint path to load_model()."
        )
    if weight_auto_download_allowed():
        return
    raise WeightDownloadError(
        "Auto-download of model weights is disabled on the Cloud Agent / CI "
        f"path ({NO_WEIGHT_DOWNLOAD_ENV} or CI). Place the file in the download "
        "root or pass a local checkpoint path to load_model()."
    )

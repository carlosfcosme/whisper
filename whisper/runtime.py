"""Runtime policy: CPU default, no Hugging Face Hub, bind 127.0.0.1."""

from __future__ import annotations

import os
from ipaddress import ip_address
from typing import Optional
from urllib.parse import urlparse

DEFAULT_DEVICE = "cpu"
DEFAULT_BIND_HOST = "127.0.0.1"
DEVICE_ENV = "WHISPER_DEVICE"
NO_WEIGHT_DOWNLOAD_ENV = "WHISPER_NO_WEIGHT_DOWNLOAD"
ALLOW_WEIGHT_DOWNLOAD_ENV = "WHISPER_ALLOW_WEIGHT_DOWNLOAD"
BIND_HOST_ENV = "WHISPER_BIND_HOST"

_TRUTHY = {"1", "true", "yes", "on"}
_HF_HUB_HOSTS = frozenset({"huggingface.co", "hf.co"})
_HF_HUB_SUFFIXES = (".huggingface.co", ".hf.co")


class WeightDownloadError(RuntimeError):
    """Raised when an automatic model-weight pull is refused."""


class BindError(ValueError):
    """Raised when a listener would bind off loopback (not 127.0.0.1)."""


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def default_device() -> str:
    """Return the default PyTorch device for ``load_model`` and the CLI.

    The default is ``cpu``. ``torch.cuda.is_available()`` is not consulted.
    Set ``WHISPER_DEVICE`` (or pass ``device=`` / ``--device``) to opt in.
    """
    explicit = os.environ.get(DEVICE_ENV, "").strip()
    return explicit or DEFAULT_DEVICE


def is_hf_hub_url(url: str) -> bool:
    """True for Hugging Face Hub hosts (``huggingface.co``, ``hf.co``, subdomains)."""
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    return host in _HF_HUB_HOSTS or host.endswith(_HF_HUB_SUFFIXES)


def weight_auto_download_allowed() -> bool:
    """False when ``WHISPER_NO_WEIGHT_DOWNLOAD`` is set.

    ``WHISPER_ALLOW_WEIGHT_DOWNLOAD=1`` is an explicit escape hatch.
    Hugging Face Hub URLs are refused regardless of this flag.
    """
    if _env_truthy(ALLOW_WEIGHT_DOWNLOAD_ENV):
        return True
    if _env_truthy(NO_WEIGHT_DOWNLOAD_ENV):
        return False
    return True


def refuse_weight_auto_download(url: str) -> None:
    """Refuse Hub URLs always, and all cache-miss pulls when downloads are disabled."""
    if is_hf_hub_url(url):
        host = urlparse(url).hostname or "<unknown>"
        raise WeightDownloadError(
            f"Refusing Hugging Face Hub weight pull from host {host!r}. "
            "Pass a local checkpoint path to load_model()."
        )
    if weight_auto_download_allowed():
        return
    raise WeightDownloadError(
        "Auto-download of model weights is disabled "
        f"({NO_WEIGHT_DOWNLOAD_ENV}=1). Place the file in the download "
        "root or pass a local checkpoint path to load_model()."
    )


def normalize_bind_host(host: Optional[str]) -> str:
    """Return a loopback bind address, or raise BindError.

    ``localhost`` is rewritten to ``127.0.0.1`` (no DNS). Unspecified
    addresses such as ``0.0.0.0`` and ``::`` are refused, as are LAN and
    public hosts.
    """
    raw = (host or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError("bind host is required; use 127.0.0.1")
    if raw.lower().rstrip(".") == "localhost":
        return DEFAULT_BIND_HOST
    try:
        ip = ip_address(raw.split("%", 1)[0])
    except ValueError as exc:
        raise BindError(f"refusing non-localhost bind {host!r}; use 127.0.0.1") from exc
    if not ip.is_loopback:
        raise BindError(f"refusing non-localhost bind {host!r}; use 127.0.0.1")
    return str(ip)


def default_bind_host() -> str:
    """Host for helper listeners. Default is ``127.0.0.1``.

    ``WHISPER_BIND_HOST`` may override only to another loopback address.
    """
    explicit = os.environ.get(BIND_HOST_ENV, "").strip()
    return normalize_bind_host(explicit or DEFAULT_BIND_HOST)


def is_loopback_bind_host(host: str) -> bool:
    try:
        normalize_bind_host(host)
        return True
    except BindError:
        return False

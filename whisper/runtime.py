"""Runtime policy: block weight downloads, bind 127.0.0.1 only."""

from __future__ import annotations

import os
from ipaddress import ip_address
from typing import Tuple
from urllib.parse import urlparse

NO_WEIGHT_DOWNLOAD_ENV = "WHISPER_NO_WEIGHT_DOWNLOAD"
ALLOW_WEIGHT_DOWNLOAD_ENV = "WHISPER_ALLOW_WEIGHT_DOWNLOAD"
CPU_ONLY_ENV = "WHISPER_CPU_ONLY"
DEVICE_ENV = "WHISPER_DEVICE"
BIND_HOST_ENV = "WHISPER_BIND_HOST"
DEFAULT_DEVICE = "cpu"
DEFAULT_BIND_HOST = "127.0.0.1"

_TRUTHY = {"1", "true", "yes", "on"}
_HF_HUB_HOSTS = {"huggingface.co", "hf.co"}
_HF_HUB_SUFFIXES = (".huggingface.co", ".hf.co")
_WILDCARD_HOSTS = frozenset({"", "*", "::", "::0"})


class WeightDownloadError(RuntimeError):
    """Raised when an automatic model-weight pull is refused."""


class BindError(RuntimeError):
    """Raised when a listener would bind off loopback."""


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _all_interfaces_v4() -> str:
    return ".".join(("0", "0", "0", "0"))


def default_device() -> str:
    """Default device is ``cpu``. CUDA availability is not consulted."""
    explicit = os.environ.get(DEVICE_ENV, "").strip()
    return explicit or DEFAULT_DEVICE


def is_hf_hub_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    return host in _HF_HUB_HOSTS or host.endswith(_HF_HUB_SUFFIXES)


def weight_auto_download_allowed() -> bool:
    if _env_truthy(ALLOW_WEIGHT_DOWNLOAD_ENV):
        return True
    if (
        _env_truthy(NO_WEIGHT_DOWNLOAD_ENV)
        or _env_truthy(CPU_ONLY_ENV)
        or _env_truthy("CI")
    ):
        return False
    return True


def refuse_weight_auto_download(url: str) -> None:
    """Refuse Hugging Face Hub always; refuse all cache-miss pulls on CI."""
    if is_hf_hub_url(url):
        host = urlparse(url).hostname or "<unknown>"
        raise WeightDownloadError(
            "Refusing Hugging Face Hub weight pull from host {!r}.".format(host)
        )
    if weight_auto_download_allowed():
        return
    raise WeightDownloadError(
        "Auto-download of model weights is disabled ({} or CI).".format(
            NO_WEIGHT_DOWNLOAD_ENV
        )
    )


def is_loopback_bind_host(host: str) -> bool:
    if not host:
        return False
    normalized = host.strip().lower().rstrip(".")
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if normalized in {DEFAULT_BIND_HOST, "localhost", "::1"}:
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def refuse_non_localhost_bind(host: str) -> None:
    if is_loopback_bind_host(host):
        return
    shown = host if host else "<wildcard>"
    raise BindError(
        "Refusing bind to {!r}. Listeners must bind {}.".format(
            shown, DEFAULT_BIND_HOST
        )
    )


def default_bind_host() -> str:
    explicit = os.environ.get(BIND_HOST_ENV, "").strip()
    host = explicit or DEFAULT_BIND_HOST
    if host == _all_interfaces_v4() or host in _WILDCARD_HOSTS:
        refuse_non_localhost_bind(host)
    refuse_non_localhost_bind(host)
    return host


def bind_localhost(sock, port=0) -> Tuple[str, int]:
    host = default_bind_host()
    sock.bind((host, int(port)))
    name = sock.getsockname()
    return name[0], int(name[1])

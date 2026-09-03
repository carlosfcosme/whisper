"""Runtime defaults: CPU device, no Hugging Face Hub, bind 127.0.0.1."""

import os
from ipaddress import ip_address
from typing import Optional, Tuple
from urllib.parse import urlparse

CPU_ONLY_ENV = "WHISPER_CPU_ONLY"
NO_WEIGHT_DOWNLOAD_ENV = "WHISPER_NO_WEIGHT_DOWNLOAD"
ALLOW_WEIGHT_DOWNLOAD_ENV = "WHISPER_ALLOW_WEIGHT_DOWNLOAD"
DEVICE_ENV = "WHISPER_DEVICE"
LOCALHOST_ONLY_ENV = "WHISPER_LOCALHOST_ONLY"
BIND_HOST_ENV = "WHISPER_BIND_HOST"
DEFAULT_DEVICE = "cpu"
DEFAULT_BIND_HOST = "127.0.0.1"

_TRUTHY = {"1", "true", "yes", "on"}

_HF_HUB_HOSTS = {"huggingface.co", "hf.co"}
_HF_HUB_SUFFIXES = (".huggingface.co", ".hf.co")
_LOOPBACK_NAMES = {"127.0.0.1", "localhost", "::1"}


class WeightDownloadError(RuntimeError):
    """Raised when an automatic model-weight pull is refused."""


class BindError(RuntimeError):
    """Raised when a listener would bind off loopback (not 127.0.0.1)."""


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def default_device() -> str:
    """Return the default PyTorch device for ``load_model`` and the CLI.

    Always ``cpu`` unless ``WHISPER_DEVICE`` is set. Does not call
    ``torch.cuda.is_available()``. ``WHISPER_CPU_ONLY`` forces ``cpu``.
    """
    if _env_truthy(CPU_ONLY_ENV):
        return DEFAULT_DEVICE
    explicit = os.environ.get(DEVICE_ENV, "").strip()
    if explicit:
        return explicit
    return DEFAULT_DEVICE


def is_hf_hub_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    if host in _HF_HUB_HOSTS:
        return True
    return any(host.endswith(suffix) for suffix in _HF_HUB_SUFFIXES)


def weight_auto_download_allowed() -> bool:
    if _env_truthy(ALLOW_WEIGHT_DOWNLOAD_ENV):
        return True
    if _env_truthy(NO_WEIGHT_DOWNLOAD_ENV) or _env_truthy("CI"):
        return False
    if _env_truthy("HF_HUB_OFFLINE") or _env_truthy("TRANSFORMERS_OFFLINE"):
        return False
    return True


def refuse_weight_auto_download(url: str) -> None:
    """Refuse Hugging Face Hub always; refuse other auto-downloads in CI."""
    if is_hf_hub_url(url):
        raise WeightDownloadError(
            "Hugging Face Hub download is disabled: {}".format(url)
        )
    if not weight_auto_download_allowed():
        raise WeightDownloadError(
            "Automatic weight download is disabled: {}".format(url)
        )


def is_loopback_host(host: Optional[str]) -> bool:
    if host is None:
        return False
    text = host.strip().lower()
    if not text:
        return False
    if text in _LOOPBACK_NAMES:
        return True
    try:
        return ip_address(text).is_loopback
    except ValueError:
        return False


def refuse_non_localhost_bind(host: Optional[str]) -> str:
    if not is_loopback_host(host):
        raise BindError(
            "refusing to bind {!r}; listeners must use 127.0.0.1".format(host)
        )
    return host


def default_bind_host() -> str:
    host = os.environ.get(BIND_HOST_ENV, DEFAULT_BIND_HOST).strip() or DEFAULT_BIND_HOST
    return refuse_non_localhost_bind(host)


def bind_localhost(sock, port: int = 0) -> Tuple[str, int]:
    """Bind ``sock`` to 127.0.0.1 and return ``(host, port)``."""
    host = default_bind_host()
    if host in ("localhost", "127.0.0.1"):
        host = "127.0.0.1"
    refuse_non_localhost_bind(host)
    sock.bind((host, port))
    bound = sock.getsockname()
    return bound[0], bound[1]

"""CPU-only and offline/no-store cache defaults."""

import os
import tempfile
from ipaddress import ip_address
from typing import Tuple
from urllib.parse import urlparse

CPU_ONLY_ENV = "WHISPER_CPU_ONLY"
NO_STORE_ENV = "WHISPER_NO_STORE"
OFFLINE_ENV = "WHISPER_OFFLINE"
NO_WEIGHT_DOWNLOAD_ENV = "WHISPER_NO_WEIGHT_DOWNLOAD"
ALLOW_WEIGHT_DOWNLOAD_ENV = "WHISPER_ALLOW_WEIGHT_DOWNLOAD"
DEVICE_ENV = "WHISPER_DEVICE"
CACHE_DIR_ENV = "WHISPER_CACHE_DIR"
BIND_HOST_ENV = "WHISPER_BIND_HOST"

DEFAULT_BIND_HOST = "127.0.0.1"
CACHE_CONTROL_NO_STORE = "no-store"
NO_STORE_CACHE_DIRNAME = "whisper-no-store"

_TRUTHY = {"1", "true", "yes", "on"}

_HF_HUB_HOSTS = {"huggingface.co", "hf.co"}
_HF_HUB_SUFFIXES = (".huggingface.co", ".hf.co")


class WeightDownloadError(RuntimeError):
    """Raised when an automatic model-weight pull or store is refused."""


class BindError(RuntimeError):
    """Raised when a listener would bind off loopback (not 127.0.0.1)."""


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def default_device() -> str:
    """Default PyTorch device: always ``cpu`` unless explicitly overridden.

    ``torch.cuda.is_available()`` is not consulted. ``WHISPER_CPU_ONLY=1``
    forces ``cpu`` even if ``WHISPER_DEVICE`` is set. CI does not need a GPU.
    """
    if _env_truthy(CPU_ONLY_ENV):
        return "cpu"
    explicit = os.environ.get(DEVICE_ENV, "").strip()
    if explicit:
        return explicit
    return "cpu"


def no_store_enabled() -> bool:
    """True when cache writes must not persist (``WHISPER_NO_STORE`` or ``CI``)."""
    return _env_truthy(NO_STORE_ENV) or _env_truthy("CI")


def offline_enabled() -> bool:
    """True when WAN / Hub weight pulls are disabled."""
    return (
        _env_truthy(OFFLINE_ENV)
        or _env_truthy(NO_WEIGHT_DOWNLOAD_ENV)
        or _env_truthy("HF_HUB_OFFLINE")
        or _env_truthy("CI")
    )


def is_hf_hub_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    return host in _HF_HUB_HOSTS or host.endswith(_HF_HUB_SUFFIXES)


def weight_auto_download_allowed() -> bool:
    """False on the offline / no-store / CI path. Hub URLs are always refused."""
    if _env_truthy(ALLOW_WEIGHT_DOWNLOAD_ENV):
        return True
    if no_store_enabled() or offline_enabled():
        return False
    return True


def refuse_weight_auto_download(url: str) -> None:
    """Refuse Hugging Face Hub URLs always, and all cache-miss pulls when offline."""
    if is_hf_hub_url(url):
        host = urlparse(url).hostname or "<unknown>"
        raise WeightDownloadError(
            f"Refusing Hugging Face Hub weight pull from host {host!r}. "
            "CI and tests must not contact the Hub."
        )
    if weight_auto_download_allowed():
        return
    raise WeightDownloadError(
        "Auto-download of model weights is disabled "
        f"({OFFLINE_ENV}/{NO_STORE_ENV}/CI). Pass a local checkpoint path."
    )


def home_cache_root() -> str:
    """Durable home cache (``~/.cache/whisper`` or ``$XDG_CACHE_HOME/whisper``)."""
    default = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")


def default_download_root() -> str:
    """Offline/no-store friendly cache root.

    When no-store or offline, this is a process-local temp directory
    (``$TMP/whisper-no-store``), not ``~/.cache/whisper``. ``WHISPER_CACHE_DIR``
    overrides. CI does not need to persist or download weights.
    """
    explicit = os.environ.get(CACHE_DIR_ENV, "").strip()
    if explicit:
        return explicit
    if no_store_enabled() or offline_enabled():
        return os.path.join(tempfile.gettempdir(), NO_STORE_CACHE_DIRNAME)
    return home_cache_root()


def cache_control_no_store() -> str:
    """HTTP ``Cache-Control`` value for helper responses: ``no-store``."""
    return CACHE_CONTROL_NO_STORE


def is_loopback_bind_host(host: str) -> bool:
    if not host:
        return False
    normalized = host.strip().lower().rstrip(".")
    if normalized in {"127.0.0.1", "localhost"}:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def refuse_non_localhost_bind(host: str) -> None:
    if is_loopback_bind_host(host):
        return
    shown = host if host else "<wildcard>"
    raise BindError(
        f"Refusing bind to {shown!r}. Listeners must bind {DEFAULT_BIND_HOST}."
    )


def default_bind_host() -> str:
    """Helper listeners bind ``127.0.0.1``. No ``0.0.0.0``."""
    explicit = os.environ.get(BIND_HOST_ENV, "").strip()
    host = explicit or DEFAULT_BIND_HOST
    refuse_non_localhost_bind(host)
    return host


def bind_localhost(sock, port: int = 0) -> Tuple[str, int]:
    host = default_bind_host()
    sock.bind((host, int(port)))
    name = sock.getsockname()
    return name[0], int(name[1])

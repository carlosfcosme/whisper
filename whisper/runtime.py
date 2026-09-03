"""Runtime policy: CPU default, no Hugging Face Hub pulls, bind 127.0.0.1."""

import os
from ipaddress import ip_address
from typing import Tuple
from urllib.parse import urlparse

DEFAULT_DEVICE = "cpu"
DEFAULT_BIND_HOST = "127.0.0.1"

DEVICE_ENV = "WHISPER_DEVICE"
BIND_HOST_ENV = "WHISPER_BIND_HOST"
NO_WEIGHT_DOWNLOAD_ENV = "WHISPER_NO_WEIGHT_DOWNLOAD"
ALLOW_WEIGHT_DOWNLOAD_ENV = "WHISPER_ALLOW_WEIGHT_DOWNLOAD"

_TRUTHY = {"1", "true", "yes", "on"}
_HF_HUB_HOSTS = {"huggingface.co", "hf.co"}
_HF_HUB_SUFFIXES = (".huggingface.co", ".hf.co")


class WeightDownloadError(RuntimeError):
    """Raised when an automatic model-weight pull is refused."""


class BindError(RuntimeError):
    """Raised when a listener would bind off loopback."""


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
    """True for Hugging Face Hub hosts (``huggingface.co``, ``hf.co``, subdomains)."""
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    return host in _HF_HUB_HOSTS or host.endswith(_HF_HUB_SUFFIXES)


def weight_auto_download_allowed() -> bool:
    """False when CI or ``WHISPER_NO_WEIGHT_DOWNLOAD`` refuses cache-miss pulls.

    ``WHISPER_ALLOW_WEIGHT_DOWNLOAD=1`` is an explicit escape hatch.
    """
    if _env_truthy(ALLOW_WEIGHT_DOWNLOAD_ENV):
        return True
    if _env_truthy(NO_WEIGHT_DOWNLOAD_ENV) or _env_truthy("CI"):
        return False
    return True


def refuse_weight_auto_download(url: str) -> None:
    """Refuse Hugging Face Hub URLs always, and all cache-miss pulls on CI."""
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
        f"({NO_WEIGHT_DOWNLOAD_ENV} or CI). Place the file in the download "
        "root or pass a local checkpoint path to load_model()."
    )


def is_loopback_bind_host(host: str) -> bool:
    """True for ``127.0.0.1``, other loopback IPs, ``::1``, and ``localhost``."""
    if host is None:
        return False
    normalized = str(host).strip().lower().rstrip(".")
    if not normalized:
        return False
    if normalized in {"127.0.0.1", "localhost", "::1"}:
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def refuse_non_localhost_bind(host: str) -> None:
    """Refuse wildcard and non-loopback bind addresses."""
    if is_loopback_bind_host(host):
        return
    shown = host if host else "<wildcard>"
    raise BindError(
        f"Refusing bind to {shown!r}. Listeners must bind {DEFAULT_BIND_HOST}."
    )


def default_bind_host() -> str:
    """Host for any helper listener. Default is ``127.0.0.1``.

    ``WHISPER_BIND_HOST`` may override only to another loopback address.
    """
    explicit = os.environ.get(BIND_HOST_ENV, "").strip()
    host = explicit or DEFAULT_BIND_HOST
    refuse_non_localhost_bind(host)
    return host


def bind_localhost(sock, port=0) -> Tuple[str, int]:
    """Bind *sock* to ``127.0.0.1`` (or a loopback override) and return the name."""
    host = default_bind_host()
    sock.bind((host, int(port)))
    name = sock.getsockname()
    return name[0], int(name[1])

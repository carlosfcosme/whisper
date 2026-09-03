"""Runtime defaults: CPU device, no Hub weight pull, bind 127.0.0.1."""

import os
from typing import Tuple
from urllib.parse import urlparse

CPU_ONLY_ENV = "WHISPER_CPU_ONLY"
NO_WEIGHT_DOWNLOAD_ENV = "WHISPER_NO_WEIGHT_DOWNLOAD"
ALLOW_WEIGHT_DOWNLOAD_ENV = "WHISPER_ALLOW_WEIGHT_DOWNLOAD"
DEVICE_ENV = "WHISPER_DEVICE"
BIND_HOST_ENV = "WHISPER_BIND_HOST"
DEFAULT_BIND_HOST = "127.0.0.1"

_TRUTHY = {"1", "true", "yes", "on"}
_HF_HUB_HOSTS = {"huggingface.co", "hf.co"}
_HF_HUB_SUFFIXES = (".huggingface.co", ".hf.co")


class WeightDownloadError(RuntimeError):
    """Raised when an automatic model-weight pull is refused."""


class BindError(RuntimeError):
    """Raised when a listener would bind off 127.0.0.1."""


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def default_device() -> str:
    """Return the default PyTorch device for ``load_model`` and the CLI.

    The default is ``cpu``. ``torch.cuda.is_available()`` is not consulted.
    Set ``WHISPER_DEVICE`` or pass ``device=`` / ``--device`` to opt in.
    """
    explicit = os.environ.get(DEVICE_ENV, "").strip()
    if explicit:
        return explicit
    return "cpu"


def is_hf_hub_url(url: str) -> bool:
    """True for Hugging Face Hub hosts and subdomains."""
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    return host in _HF_HUB_HOSTS or host.endswith(_HF_HUB_SUFFIXES)


def weight_auto_download_allowed() -> bool:
    """False when tests/CI or ``WHISPER_NO_WEIGHT_DOWNLOAD`` refuse pulls."""
    if _env_truthy(ALLOW_WEIGHT_DOWNLOAD_ENV):
        return True
    if (
        _env_truthy(NO_WEIGHT_DOWNLOAD_ENV)
        or _env_truthy(CPU_ONLY_ENV)
        or _env_truthy("CI")
        or os.environ.get("PYTEST_CURRENT_TEST")
    ):
        return False
    return True


def refuse_weight_auto_download(url: str) -> None:
    """Refuse Hugging Face Hub URLs always, and cache-miss pulls in tests/CI."""
    if is_hf_hub_url(url):
        host = urlparse(url).hostname or "<unknown>"
        raise WeightDownloadError(
            f"Refusing Hugging Face Hub weight pull from host {host!r}. "
            "Tests must not contact the Hub. Pass a local checkpoint path."
        )
    if weight_auto_download_allowed():
        return
    raise WeightDownloadError(
        "Auto-download of model weights is disabled in tests/CI "
        f"({NO_WEIGHT_DOWNLOAD_ENV} or CI). Place the file in the download "
        "root or pass a local checkpoint path to load_model()."
    )


def serve_bind_host(host=None) -> str:
    """Return ``127.0.0.1``, or raise if the host is not that address."""
    if host in (None, "", "localhost"):
        return DEFAULT_BIND_HOST
    raw = str(host).strip().lower().rstrip(".")
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if raw == DEFAULT_BIND_HOST:
        return DEFAULT_BIND_HOST
    shown = host if host else "<wildcard>"
    raise BindError(
        f"Refusing bind to {shown!r}. Listeners must bind {DEFAULT_BIND_HOST}."
    )


def refuse_non_localhost_bind(host: str) -> None:
    """Refuse wildcard and non-127.0.0.1 bind addresses."""
    serve_bind_host(host)


def default_bind_host() -> str:
    """Host for any helper listener. Default is ``127.0.0.1``.

    ``WHISPER_BIND_HOST`` may override only to ``127.0.0.1``.
    ``0.0.0.0`` / ``::`` / public hosts are refused.
    """
    explicit = os.environ.get(BIND_HOST_ENV, "").strip()
    return serve_bind_host(explicit or DEFAULT_BIND_HOST)


def bind_localhost(sock, port=0) -> Tuple[str, int]:
    """Bind *sock* to ``127.0.0.1`` and return ``(host, port)``."""
    host = default_bind_host()
    sock.bind((host, int(port)))
    name = sock.getsockname()
    return name[0], int(name[1])

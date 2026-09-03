"""Offline and loopback policy used by tests and load_model."""

from __future__ import annotations

import ipaddress
import os
from typing import Any, Optional

BIND_HOST = "127.0.0.1"

_HUB_MARKERS = (
    "huggingface.co",
    "hf.co/",
    "hf-mirror.com",
    "cas-bridge.xethub",
)
_WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors", ".onnx", ".ckpt", ".bin")
_WEIGHT_HOST_MARKERS = (
    "openaipublic.azureedge.net",
    "whisper/models",
)


class BindError(ValueError):
    """Raised when a bind host is not loopback."""


def is_offline() -> bool:
    for key in ("WHISPER_OFFLINE", "HF_HUB_OFFLINE"):
        if os.getenv(key, "").strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def _url_text(url: Any) -> str:
    if hasattr(url, "full_url"):
        return url.full_url
    return str(url)


def is_hub_url(url: Any) -> bool:
    lowered = _url_text(url).lower()
    return any(marker in lowered for marker in _HUB_MARKERS)


def is_weight_url(url: Any) -> bool:
    lowered = _url_text(url).lower()
    if any(lowered.endswith(suffix) for suffix in _WEIGHT_SUFFIXES):
        return True
    return any(marker in lowered for marker in _WEIGHT_HOST_MARKERS)


def refuse_forbidden_fetch(url: Any, offline: Optional[bool] = None) -> None:
    """Raise before a Hub URL or an offline weight download is opened."""
    if offline is None:
        offline = is_offline()
    if is_hub_url(url):
        raise RuntimeError(
            "Hugging Face Hub fetch is refused: {}".format(_url_text(url))
        )
    if offline:
        raise RuntimeError(
            "offline is set; refusing to download model weights from the network"
        )


def serve_bind_host(host: Optional[str] = None) -> str:
    """Return a loopback bind address, or raise BindError."""
    if host in (None, "", "localhost"):
        return BIND_HOST
    raw = host.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if raw.lower() == "localhost":
        return BIND_HOST
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError:
        raise BindError(
            "serve must bind to 127.0.0.1 (got {!r}); refusing non-loopback hosts".format(
                host
            )
        )
    if not ip.is_loopback:
        raise BindError(
            "serve must bind to 127.0.0.1 (got {!r}); refusing non-loopback hosts".format(
                host
            )
        )
    return str(ip)

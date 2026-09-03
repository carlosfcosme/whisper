"""Runtime policy: CPU default, loopback bind, no Hub/weight WAN fetch."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_DEVICE = "cpu"
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


def default_device() -> str:
    return DEFAULT_DEVICE


def service_bind_host() -> str:
    return BIND_HOST


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


def refuse_forbidden_fetch(url: Any, offline: bool = False) -> None:
    """Raise before a Hub or (when offline) weight URL is opened."""
    if is_hub_url(url):
        raise RuntimeError(
            "Hugging Face Hub fetch is refused: {}".format(_url_text(url))
        )
    if offline:
        raise RuntimeError(
            "offline is set; refusing to fetch model weights from the network"
        )

"""Runtime defaults: CPU device, loopback bind, no Hugging Face Hub."""

import os
from typing import Iterable, List

DEFAULT_DEVICE = "cpu"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
LOOPBACK_HOSTS = ("127.0.0.1",)

WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors")

_HUB_MARKERS = (
    "huggingface.co",
    "hf.co/",
    "cdn-lfs.huggingface.co",
    "hf-mirror.com",
)


def is_huggingface_hub_source(value: str) -> bool:
    text = value.lower().replace("\\", "/")
    return any(marker in text for marker in _HUB_MARKERS)


def reject_huggingface_hub(value: str) -> None:
    if is_huggingface_hub_source(value):
        raise ValueError("Hugging Face Hub is not supported; use a local checkpoint")


def is_loopback_host(host: str) -> bool:
    return host in LOOPBACK_HOSTS


def require_loopback_host(host: str) -> str:
    if not is_loopback_host(host):
        raise ValueError("server must bind 127.0.0.1")
    return host


_OFFLINE_TRUTHY = frozenset({"1", "true", "yes", "on"})


def downloads_blocked() -> bool:
    """True when model/network downloads must not run (offline install/CI)."""
    for key in ("WHISPER_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.get(key, "").strip().lower() in _OFFLINE_TRUTHY:
            return True
    return False


def committed_weight_paths(paths: Iterable[str]) -> List[str]:
    found = []
    for path in paths:
        lowered = path.lower()
        if lowered.endswith(WEIGHT_SUFFIXES):
            found.append(path)
    return found

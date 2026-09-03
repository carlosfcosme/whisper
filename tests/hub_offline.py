"""Keep unit tests from contacting the Hugging Face Hub.

Whisper downloads official checkpoints with urllib.request.urlopen
(whisper/__init__.py) from Azure. This module blocks Hub hosts only.
"""

import importlib
import os
import urllib.request
from urllib.parse import urlparse

HUB_OFFLINE_ENV = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
)

HUB_NETLOCS = frozenset(
    {
        "huggingface.co",
        "www.huggingface.co",
        "hf.co",
        "www.hf.co",
    }
)

_original_urlopen = urllib.request.urlopen


def apply_hub_offline_env() -> None:
    for name in HUB_OFFLINE_ENV:
        os.environ.setdefault(name, "1")


def hub_host(url) -> str:
    raw = url.full_url if hasattr(url, "full_url") else url
    if not isinstance(raw, str):
        raw = str(raw)
    return urlparse(raw).netloc.lower().split("@")[-1].split(":")[0]


def is_huggingface_hub_host(host: str) -> bool:
    return (
        host in HUB_NETLOCS
        or host.endswith(".huggingface.co")
        or host.endswith(".hf.co")
    )


def refuse_hub_download(*args, **kwargs):
    raise RuntimeError("unit tests must not contact the Hugging Face Hub")


def urlopen_without_hub(url, *args, **kwargs):
    host = hub_host(url)
    if is_huggingface_hub_host(host):
        raise RuntimeError(f"unit tests must not contact the Hugging Face Hub ({host})")
    return _original_urlopen(url, *args, **kwargs)


def install_hub_client_guard() -> None:
    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
    except ImportError:
        return
    huggingface_hub.hf_hub_download = refuse_hub_download
    if hasattr(huggingface_hub, "snapshot_download"):
        huggingface_hub.snapshot_download = refuse_hub_download


def install_hub_guards() -> None:
    apply_hub_offline_env()
    urllib.request.urlopen = urlopen_without_hub
    install_hub_client_guard()

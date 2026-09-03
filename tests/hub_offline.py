"""Keep unit tests from contacting the Hugging Face Hub or pulling weights.

Whisper would download official checkpoints with urllib.request.urlopen
from Azure. This module blocks Hub hosts and every other remote URL.
Loopback HTTP (127.0.0.1) stays allowed for the serve health check.
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

LOOPBACK_HTTP_HOSTS = frozenset({"127.0.0.1", "localhost"})

_original_urlopen = urllib.request.urlopen


def apply_hub_offline_env() -> None:
    for name in HUB_OFFLINE_ENV:
        os.environ.setdefault(name, "1")


def request_url(url) -> str:
    raw = url.full_url if hasattr(url, "full_url") else url
    if not isinstance(raw, str):
        raw = str(raw)
    return raw


def hub_host(url) -> str:
    return urlparse(request_url(url)).netloc.lower().split("@")[-1].split(":")[0]


def is_huggingface_hub_host(host: str) -> bool:
    return (
        host in HUB_NETLOCS
        or host.endswith(".huggingface.co")
        or host.endswith(".hf.co")
    )


def is_loopback_http(url) -> bool:
    parsed = urlparse(request_url(url))
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    return scheme in {"http", "https"} and host in LOOPBACK_HTTP_HOSTS


def refuse_hub_download(*args, **kwargs):
    raise RuntimeError("unit tests must not contact the Hugging Face Hub")


def urlopen_without_hub(url, *args, **kwargs):
    host = hub_host(url)
    if is_huggingface_hub_host(host):
        raise RuntimeError(
            "unit tests must not contact the Hugging Face Hub ({})".format(host)
        )
    if is_loopback_http(url):
        return _original_urlopen(url, *args, **kwargs)
    raise RuntimeError(
        "unit tests must not pull weights or contact remote hosts ({})".format(
            host or request_url(url)
        )
    )


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

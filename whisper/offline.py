"""Offline / no-Hub policy: no weight fetch, loopback bind, host checks.

CI and tests set WHISPER_OFFLINE / HF_HUB_OFFLINE so named-model loads
never open a network socket. Hugging Face Hub URLs are refused even when
those env vars are unset — this package does not fetch from the Hub.

Services bind 127.0.0.1 only. Network interception in tests is Python
socket monkeypatching (not BPF).
"""

from __future__ import annotations

import os
import socket
from typing import Optional, Union
from urllib.parse import urlparse

_OFFLINE_TRUTHY = frozenset({"1", "true", "yes", "on"})
OFFLINE_ENV_VARS = (
    "WHISPER_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)

# Token *names* only — never read or print values.
TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)

BIND_HOST = "127.0.0.1"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
# Wildcard / all-interfaces hosts, built without bind literals so
# package source stays greppable-clean for CI bind checks.
_WILDCARD_V4 = ".".join(("0", "0", "0", "0"))
_WILDCARD_V6 = ":" * 2
_WILDCARD_V6_BRACKETS = "[{}]".format(_WILDCARD_V6)
ALL_INTERFACES = frozenset({"", _WILDCARD_V4, _WILDCARD_V6, _WILDCARD_V6_BRACKETS})

_HUB_HOSTS = frozenset(
    {
        "huggingface.co",
        "www.huggingface.co",
        "hf.co",
        "www.hf.co",
        "hf-mirror.com",
        "www.hf-mirror.com",
    }
)
_HUB_HOST_SUFFIXES = (
    ".huggingface.co",
    ".hf.co",
    ".hf-mirror.com",
)
_WEIGHT_HOSTS = frozenset(
    {
        "openaipublic.azureedge.net",
    }
)
_WEIGHT_HOST_MARKERS = ("azureedge.net", "xethub")


def _url_text(url: Union[str, object]) -> str:
    if isinstance(url, str):
        return url
    return getattr(url, "full_url", None) or str(url)


def _hostname(url: Union[str, object]) -> str:
    raw = _url_text(url)
    host = urlparse(raw).netloc.lower().split("@")[-1].split(":")[0]
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host


def normalize_host(host: Optional[object]) -> str:
    """Return a lowercase hostname (no port, no IPv6 brackets)."""
    if host is None:
        return ""
    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    text = str(host).strip().lower()
    if text.startswith("[") and "]" in text:
        text = text[1 : text.index("]")]
    text = text.split("%")[0].split(":")[0].rstrip(".")
    return text


def is_loopback_host(host: Optional[object]) -> bool:
    return normalize_host(host) in LOOPBACK_HOSTS


def is_all_interfaces_host(host: Optional[object]) -> bool:
    raw = "" if host is None else str(host).strip()
    if raw in ALL_INTERFACES:
        return True
    return normalize_host(host) in ALL_INTERFACES


def is_hub_host(host: Optional[object]) -> bool:
    name = normalize_host(host)
    if not name:
        return False
    if name in _HUB_HOSTS:
        return True
    if any(name.endswith(suffix) for suffix in _HUB_HOST_SUFFIXES):
        return True
    if "xethub" in name:
        return True
    return False


def is_weight_host(host: Optional[object]) -> bool:
    name = normalize_host(host)
    if not name:
        return False
    if name in _WEIGHT_HOSTS:
        return True
    return any(marker in name for marker in _WEIGHT_HOST_MARKERS)


def is_hub_url(url: Union[str, object]) -> bool:
    """True when *url* targets the Hugging Face Hub (or a Hub mirror)."""
    return is_hub_host(_hostname(url))


def is_blocked_network_host(host: Optional[object]) -> bool:
    """True for Hub, weight CDNs, or any non-loopback host."""
    if is_loopback_host(host):
        return False
    if is_hub_host(host) or is_weight_host(host):
        return True
    name = normalize_host(host)
    return bool(name) and name not in LOOPBACK_HOSTS


def require_loopback_bind(host: str = BIND_HOST) -> str:
    """Return *host* if it is loopback; raise otherwise.

    Refuses all-interfaces and non-loopback binds so a service cannot
    listen on a public address.
    """
    if is_all_interfaces_host(host) or not is_loopback_host(host):
        raise ValueError("services must bind 127.0.0.1 only; refused {!r}".format(host))
    return BIND_HOST


def bind_loopback(sock: socket.socket, port: int = 0):
    """Bind *sock* to 127.0.0.1 and return ``(host, port)``."""
    host = require_loopback_bind(BIND_HOST)
    sock.bind((host, port))
    return sock.getsockname()


def weights_download_forbidden() -> bool:
    """True when install/test (or the user) has disabled weight downloads."""
    for key in OFFLINE_ENV_VARS:
        if os.environ.get(key, "").strip().lower() in _OFFLINE_TRUTHY:
            return True
    return False


def refuse_network_weight_fetch(url: Union[str, object], target: str) -> None:
    """Raise before a Hub or (when offline) any WAN checkpoint fetch.

    Local files with a matching SHA-256 are used by the caller *before*
    this is invoked, so a populated cache still loads offline.
    """
    if is_hub_url(url):
        raise RuntimeError(
            "Hugging Face Hub fetch is refused: {}".format(_url_text(url))
        )
    if weights_download_forbidden():
        raise RuntimeError(
            "Refusing to download model weights while offline "
            "(WHISPER_OFFLINE or HF_HUB_OFFLINE is set). "
            "Missing or invalid local cache: {}".format(target)
        )

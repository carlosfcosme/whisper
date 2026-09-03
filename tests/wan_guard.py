"""Runtime guards that prove tests do not contact model hubs or the WAN.

Normal tests may use loopback and ``file:`` URLs. Hugging Face Hub hosts,
official Whisper CDNs, and any other non-loopback TCP target are refused.
No credentials are read or sent.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import urllib.request
from typing import Any, List, Optional
from urllib.parse import urlparse

# Recorded denied targets (hub / WAN). Intentional probe tests append here too.
DENIED: List[str] = []

HUB_AND_CDN_MARKERS = (
    "huggingface.co",
    "huggingface.com",
    "hf.co",
    "azureedge.net",
    "blob.core.windows.net",
    "openaipublic",
    "hf-mirror.com",
)

_LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1", "localhost.localdomain"})


def configure_offline_env() -> None:
    """Force Hub/transformers offline flags. Never set tokens or allow-download."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("WHISPER_OFFLINE", "1")
    os.environ.pop("WHISPER_ALLOW_WEIGHT_DOWNLOAD", None)
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)


def request_url(url: Any) -> str:
    if isinstance(url, str):
        return url
    return getattr(url, "full_url", None) or str(url)


def is_loopback_host(host: Optional[str]) -> bool:
    if not host:
        return False
    host = host.strip().strip("[]").lower().rstrip(".")
    if host in _LOOPBACK_NAMES or host.startswith("127."):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_forbidden_url(url: str) -> bool:
    raw = request_url(url)
    lowered = raw.lower()
    if any(marker in lowered for marker in HUB_AND_CDN_MARKERS):
        return True
    parsed = urlparse(raw)
    if parsed.scheme in {"file", ""} and not parsed.netloc:
        return False
    host = parsed.hostname
    if host is None:
        return bool(parsed.scheme in {"http", "https"})
    return not is_loopback_host(host)


def deny(target: str) -> None:
    DENIED.append(target)
    raise RuntimeError(
        "tests must not contact model hubs or the WAN "
        f"(refused {target!r}; loopback and file: only)"
    )


def install_urlopen_guard(monkeypatch: Any) -> None:
    real_urlopen = urllib.request.urlopen

    def guarded(url, *args, **kwargs):
        raw = request_url(url)
        if is_forbidden_url(raw):
            deny(raw)
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded)


def install_hub_client_guard(monkeypatch: Any) -> None:
    try:
        import huggingface_hub
    except ImportError:
        return

    def boom(*args, **kwargs):
        deny("huggingface_hub")

    for name in (
        "hf_hub_download",
        "snapshot_download",
        "hf_hub_url",
        "login",
    ):
        if hasattr(huggingface_hub, name):
            monkeypatch.setattr(huggingface_hub, name, boom, raising=False)


def _peer_from_create_connection(address: Any) -> str:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return str(address)


def install_socket_guard(monkeypatch: Any) -> None:
    """Refuse non-loopback TCP. DNS to a stub resolver on loopback is allowed."""
    real_create = socket.create_connection

    def guarded_create(address, *args, **kwargs):
        host = _peer_from_create_connection(address)
        if not is_loopback_host(host):
            deny(f"tcp:{address!r}")
        return real_create(address, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", guarded_create)

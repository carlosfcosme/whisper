"""Runtime guards that fail tests on attempted network/weight download.

Normal tests may use loopback and ``file:`` URLs. Hugging Face Hub hosts,
official Whisper CDNs, weight-file URLs, and any other non-loopback TCP
target are recorded and refused. A pytest hook fails the test if an
attempt was made, even when the exception is swallowed.

No credentials are read or sent.
"""

from __future__ import annotations

import http.client
import ipaddress
import os
import socket
import urllib.request
from typing import Any, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

# Recorded denied targets (hub / WAN / weight URLs / non-loopback bind).
ATTEMPTS: List[str] = []

HUB_AND_CDN_MARKERS = (
    "huggingface.co",
    "huggingface.com",
    "hf.co",
    "azureedge.net",
    "blob.core.windows.net",
    "openaipublic",
    "hf-mirror.com",
)

WEIGHT_SUFFIXES = (
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".gguf",
)

_LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1", "localhost.localdomain"})
_PROBE_MARKERS = frozenset({"probes_network", "probes_bind"})


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


def is_weight_url(url: str) -> bool:
    path = urlparse(request_url(url)).path.lower()
    return any(path.endswith(suffix) for suffix in WEIGHT_SUFFIXES)


def is_forbidden_url(url: str) -> bool:
    raw = request_url(url)
    lowered = raw.lower()
    if any(marker in lowered for marker in HUB_AND_CDN_MARKERS):
        return True
    parsed = urlparse(raw)
    if parsed.scheme in {"file", ""} and not parsed.netloc:
        return False
    if parsed.scheme in {"http", "https"} and is_weight_url(raw):
        return True
    host = parsed.hostname
    if host is None:
        return bool(parsed.scheme in {"http", "https"})
    return not is_loopback_host(host)


def record_attempt(target: str) -> None:
    ATTEMPTS.append(target)


def deny(target: str) -> None:
    record_attempt(target)
    raise RuntimeError(
        "tests must not contact model hubs or the WAN "
        f"(refused {target!r}; loopback and file: only)"
    )


def unexpected_network_attempts(
    marker_names: Iterable[str],
    attempts: Sequence[str],
) -> List[str]:
    """Return attempts that should fail a test.

    Tests marked ``probes_network`` / ``probes_bind`` may attempt a
    refused fetch or bind on purpose. Any other test that records an
    attempt fails.
    """
    names = set(marker_names)
    if names & _PROBE_MARKERS:
        return []
    return list(attempts)


def install_urlopen_guard(monkeypatch: Any) -> None:
    real_urlopen = urllib.request.urlopen
    real_urlretrieve = urllib.request.urlretrieve

    def guarded(url, *args, **kwargs):
        raw = request_url(url)
        if is_forbidden_url(raw):
            deny(raw)
        return real_urlopen(url, *args, **kwargs)

    def guarded_retrieve(url, *args, **kwargs):
        raw = request_url(url)
        if is_forbidden_url(raw):
            deny(raw)
        return real_urlretrieve(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded)
    monkeypatch.setattr(urllib.request, "urlretrieve", guarded_retrieve)


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
    """Refuse non-loopback TCP connect. Bind policy lives in ``whisper.bind``."""
    real_create = socket.create_connection
    real_connect = socket.socket.connect

    def guarded_create(address, *args, **kwargs):
        host = _peer_from_create_connection(address)
        if not is_loopback_host(host):
            deny(f"tcp:{address!r}")
        return real_create(address, *args, **kwargs)

    def guarded_connect(self, address):
        if isinstance(address, tuple) and address:
            host = address[0]
            if isinstance(host, str) and "/" not in host and not is_loopback_host(host):
                deny(f"socket.connect:{address!r}")
        return real_connect(self, address)

    monkeypatch.setattr(socket, "create_connection", guarded_create)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


def install_http_client_guard(monkeypatch: Any) -> None:
    real_http = http.client.HTTPConnection.connect
    real_https = http.client.HTTPSConnection.connect

    def guarded_http(self):
        host = getattr(self, "host", "")
        if not is_loopback_host(host):
            deny(f"http.client:{host}")
        return real_http(self)

    def guarded_https(self):
        host = getattr(self, "host", "")
        if not is_loopback_host(host):
            deny(f"http.client.https:{host}")
        return real_https(self)

    monkeypatch.setattr(http.client.HTTPConnection, "connect", guarded_http)
    monkeypatch.setattr(http.client.HTTPSConnection, "connect", guarded_https)

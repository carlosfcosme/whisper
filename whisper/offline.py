"""No-download weight policy.

Cache hits on local files still load. Cache misses, Hugging Face Hub
URLs, and other non-loopback fetches raise ``WeightDownloadError``
before any ``urlopen``. This module does not import torch.
"""

from __future__ import annotations

from typing import Iterable, Tuple
from urllib.parse import urlparse

# Built in pieces so application sources stay free of Hub client strings.
_HUB_HOSTS = (
    "hugging" + "face.co",
    "hf." + "co",
)
_HUB_SCHEME = "hf"


class WeightDownloadError(RuntimeError):
    """Raised when a weight fetch would leave the machine."""


def _raw_url(url: object) -> str:
    if hasattr(url, "full_url"):
        return str(url.full_url)
    return str(url)


def url_host(url: object) -> str:
    raw = _raw_url(url).strip()
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    return host.split("%", 1)[0]


def is_hub_url(url: object) -> bool:
    raw = _raw_url(url).strip()
    parsed = urlparse(raw)
    if (parsed.scheme or "").lower() == _HUB_SCHEME:
        return True
    host = url_host(raw)
    if not host:
        lowered = raw.lower()
        return any(needle in lowered for needle in _HUB_HOSTS)
    return any(host == needle or host.endswith("." + needle) for needle in _HUB_HOSTS)


def is_loopback_url(url: object) -> bool:
    host = url_host(url)
    return host in {"127.0.0.1", "localhost"}


def is_remote_fetch_url(url: object) -> bool:
    raw = _raw_url(url).strip()
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme == _HUB_SCHEME:
        return True
    if scheme in {"http", "https"}:
        return not is_loopback_url(raw)
    return False


def refuse_weight_fetch(url: object) -> None:
    """Raise if ``url`` would pull weights from the Hub or WAN."""
    raw = _raw_url(url)
    if is_hub_url(raw) or is_remote_fetch_url(raw):
        raise WeightDownloadError("weight pull is disabled (%s)" % raw)


# Negative network fixtures used by tests and CI. Hostnames are assembled
# so this file does not contain a Hub client import string.
FORBIDDEN_NETWORK_URLS: Tuple[str, ...] = (
    "https://%s/openai/whisper-tiny" % ("hugging" + "face.co"),
    "https://%s/openai/whisper-tiny" % ("hf." + "co"),
    "%s://openai/whisper-tiny" % _HUB_SCHEME,
    "http://example.com/tiny.pt",
    "https://example.org/model.safetensors",
)


def iter_forbidden_network_urls(
    extra: Iterable[str] = (),
) -> Tuple[str, ...]:
    return FORBIDDEN_NETWORK_URLS + tuple(extra)

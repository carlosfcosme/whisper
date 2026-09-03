"""Runtime and static guards: no WAN in tests, offline fixtures only."""

from __future__ import annotations

import re
import sys
from ipaddress import ip_address
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent

FORBIDDEN_HUB_APIS = ("huggingface_hub", "hf_hub_download", "snapshot_download")
_FORBIDDEN_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(name) for name in FORBIDDEN_HUB_APIS) + r")\b"
)
SCAN_SKIP = {
    "hub_guard.py",
    "offline_ci.py",
    "test_offline_guards.py",
    "conftest.py",
}

HUB_SUFFIXES = (".huggingface.co", ".hf.co")
HUB_EXACT = {"huggingface.co", "hf.co"}


class HubImportError(ImportError):
    """Raised when a test tries to import huggingface_hub."""


class WanNetworkError(RuntimeError):
    """Raised when a test tries to use WAN."""


class HubNetworkError(WanNetworkError):
    """Raised when a test tries to reach the Hugging Face Hub."""


class _ForbidHuggingFaceHub:
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root == "huggingface_hub":
            raise HubImportError(
                "tests must not import huggingface_hub; use offline fixtures only"
            )
        return None


def install_hub_import_block() -> None:
    if any(isinstance(finder, _ForbidHuggingFaceHub) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _ForbidHuggingFaceHub())


def hostname_is_loopback(host) -> bool:
    """True for localhost / loopback IPs. Hostnames are not resolved."""
    if not host:
        return False
    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    normalized = str(host).strip().lower().rstrip(".")
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def url_host(url) -> str:
    if not isinstance(url, str):
        url = getattr(url, "full_url", None) or str(url)
    return (urlparse(url).hostname or "").strip().lower().rstrip(".")


def is_hub_url(url) -> bool:
    host = url_host(url)
    if not host:
        return False
    if host in HUB_EXACT:
        return True
    return any(host.endswith(suffix) for suffix in HUB_SUFFIXES)


def is_loopback_url(url) -> bool:
    if not isinstance(url, str):
        url = getattr(url, "full_url", None) or str(url)
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return True
    if parsed.scheme not in {"http", "https"}:
        return False
    return hostname_is_loopback(parsed.hostname)


def refuse_hub_url(url) -> None:
    if is_hub_url(url):
        raise HubNetworkError(f"tests must not hit the Hugging Face Hub: {url!r}")


def refuse_wan_url(url) -> None:
    refuse_hub_url(url)
    if is_loopback_url(url):
        return
    raise WanNetworkError(f"tests must not use WAN: {url!r}")


def refuse_wan_host(host) -> None:
    if hostname_is_loopback(host):
        return
    raise WanNetworkError(f"tests must not use WAN host: {host!r}")


def install_hub_urlopen_block() -> None:
    install_offline_network_block()


def install_offline_network_block() -> None:
    import socket
    import urllib.request

    original_urlopen = urllib.request.urlopen
    if not getattr(original_urlopen, "_whisper_offline_guard", False):

        def guarded_urlopen(url, *args, **kwargs):
            refuse_wan_url(url)
            return original_urlopen(url, *args, **kwargs)

        guarded_urlopen._whisper_offline_guard = True
        urllib.request.urlopen = guarded_urlopen

    original_cc = socket.create_connection
    if not getattr(original_cc, "_whisper_offline_guard", False):

        def guarded_cc(address, *args, **kwargs):
            host = (
                address[0]
                if isinstance(address, (tuple, list)) and address
                else address
            )
            refuse_wan_host(host)
            return original_cc(address, *args, **kwargs)

        guarded_cc._whisper_offline_guard = True
        socket.create_connection = guarded_cc


def iter_test_sources(root: Optional[Path] = None) -> Iterable[Path]:
    base = TESTS_DIR if root is None else root
    for path in sorted(base.rglob("*.py")):
        if path.name in SCAN_SKIP:
            continue
        if ".pyc" in path.suffixes:
            continue
        yield path


def forbidden_hub_api_hits(root: Optional[Path] = None) -> List[str]:
    hits = []
    for path in iter_test_sources(root):
        text = path.read_text(encoding="utf-8")
        if _FORBIDDEN_RE.search(text):
            rel = path.relative_to(REPO_ROOT if root is None else root.parent)
            hits.append(str(rel))
    return hits

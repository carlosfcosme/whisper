"""Runtime and static guards: no Hugging Face Hub, offline fixtures only."""

from __future__ import annotations

import re
import sys
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


class HubNetworkError(RuntimeError):
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


def refuse_hub_url(url) -> None:
    if is_hub_url(url):
        raise HubNetworkError(f"tests must not hit the Hugging Face Hub: {url!r}")


def install_hub_urlopen_block() -> None:
    import urllib.request

    original = urllib.request.urlopen

    if getattr(original, "_whisper_hub_guard", False):
        return

    def guarded(url, *args, **kwargs):
        refuse_hub_url(url)
        return original(url, *args, **kwargs)

    guarded._whisper_hub_guard = True
    urllib.request.urlopen = guarded


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

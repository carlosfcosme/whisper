"""Offline bootstrap guards: block WAN/model fetch; allow 127.0.0.1 only.

Stdlib only. No credentials, no kernel modules, no BPF. Importable by file
path so tests do not need torch.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Optional, Set
from urllib.parse import urlparse

_BIND_PATH = Path(__file__).resolve().parent / "bind.py"


def _bind():
    cached = sys.modules.get("whisper.bind") or sys.modules.get("whisper_bind_guard")
    if cached is not None:
        return cached
    try:
        from . import bind as bind_mod

        return bind_mod
    except ImportError:
        spec = importlib.util.spec_from_file_location("whisper_bind_guard", _BIND_PATH)
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError(str(_BIND_PATH))
        sys.modules["whisper_bind_guard"] = module
        spec.loader.exec_module(module)
        return module


class OfflineNetworkError(RuntimeError):
    """WAN, wildcard, or model-weight fetch attempted while offline."""


_state = {
    "depth": 0,
    "urlopen": None,
    "create_connection": None,
    "download": None,
    "whisper": None,
}


def weight_cache_dir() -> Path:
    default = os.path.join(os.path.expanduser("~"), ".cache")
    return Path(os.getenv("XDG_CACHE_HOME", default)) / "whisper"


def weight_files(cache_dir: Optional[os.PathLike] = None) -> List[Path]:
    root = Path(cache_dir) if cache_dir is not None else weight_cache_dir()
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.suffix == ".pt")


def assert_no_new_weights(
    before: Optional[Iterable[Path]] = None,
    cache_dir: Optional[os.PathLike] = None,
) -> None:
    prior: Set[Path] = set(before or [])
    created = [path for path in weight_files(cache_dir) if path not in prior]
    if created:
        raise OfflineNetworkError(
            "offline: model weight files were written: "
            + ", ".join(str(path) for path in created)
        )


def assert_only_loopback_listeners() -> None:
    _bind().assert_only_loopback_listeners()


def _url_host(url) -> Optional[str]:
    if hasattr(url, "host") and url.host:
        return url.host
    if hasattr(url, "full_url"):
        url = url.full_url
    parsed = urlparse(str(url))
    return parsed.hostname


def _url_scheme(url) -> str:
    if hasattr(url, "full_url"):
        url = url.full_url
    if hasattr(url, "type") and url.type:
        return url.type
    return urlparse(str(url)).scheme or ""


def _refuse_host(host, action: str) -> None:
    bind = _bind()
    try:
        bind.require_loopback_host(host)
    except bind.NonLoopbackBindError as exc:
        raise OfflineNetworkError(
            f"offline: blocked {action} to {host!r}; only "
            f"{bind.ALLOWED_BIND_HOST} is allowed"
        ) from exc


def _offline_urlopen(url, *args, **kwargs):
    scheme = _url_scheme(url)
    host = _url_host(url)
    if host in (None, "") and scheme in {"", "file"}:
        return _state["urlopen"](url, *args, **kwargs)
    _refuse_host(host, "urlopen")
    return _state["urlopen"](url, *args, **kwargs)


def _offline_create_connection(address, *args, **kwargs):
    host = address[0] if isinstance(address, tuple) else address
    _refuse_host(host, "connect")
    return _state["create_connection"](address, *args, **kwargs)


def _offline_download(url, root, in_memory=False):
    raise OfflineNetworkError(f"offline: blocked model fetch {url!r}")


def install_offline_guards() -> None:
    """Patch urlopen, create_connection, and whisper._download if imported."""
    if _state["depth"] == 0:
        _state["urlopen"] = urllib.request.urlopen
        _state["create_connection"] = socket.create_connection
        urllib.request.urlopen = _offline_urlopen
        socket.create_connection = _offline_create_connection
        whisper_mod = sys.modules.get("whisper")
        if whisper_mod is not None and hasattr(whisper_mod, "_download"):
            _state["whisper"] = whisper_mod
            _state["download"] = whisper_mod._download
            whisper_mod._download = _offline_download
    _state["depth"] += 1


def restore_offline_guards() -> None:
    if _state["depth"] <= 0:
        return
    _state["depth"] -= 1
    if _state["depth"] > 0:
        return
    if _state["urlopen"] is not None:
        urllib.request.urlopen = _state["urlopen"]
    if _state["create_connection"] is not None:
        socket.create_connection = _state["create_connection"]
    if _state["whisper"] is not None and _state["download"] is not None:
        _state["whisper"]._download = _state["download"]
    _state["urlopen"] = None
    _state["create_connection"] = None
    _state["download"] = None
    _state["whisper"] = None


@contextmanager
def offline_guards():
    """Block WAN/model fetch for the duration of the block."""
    install_offline_guards()
    try:
        yield
    finally:
        restore_offline_guards()

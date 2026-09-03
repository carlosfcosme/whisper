"""Session-level offline test-runner guard.

Fails WAN/model fetches and non-loopback bind/connect. Redirects Whisper,
Hugging Face, and torch caches away from the user home cache so existing
weights are ignored.
"""

import ipaddress
import os
import socket
import tempfile
import urllib.request
from typing import Any, Dict, Optional

from whisper.offline import OfflineDownloadError, apply_offline_env, refuse_hub_url

NETWORK_DISABLED = "Network access is disabled in tests"


class NetworkDisabledError(RuntimeError):
    """Raised when a test attempts WAN fetch or a non-loopback bind/connect."""


_installed = False
_originals: Dict[str, Any] = {}
_cache_root: Optional[str] = None
_RealSocket = socket.socket


def user_whisper_cache() -> str:
    return os.path.realpath(os.path.join(os.path.expanduser("~"), ".cache", "whisper"))


def isolated_cache_root() -> Optional[str]:
    return _cache_root


def _address_host(address: Any) -> Any:
    if isinstance(address, (tuple, list)) and address:
        return address[0]
    return address


def is_loopback_host(host: Any) -> bool:
    if host is None:
        return False
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = str(host).strip()
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _refuse_non_loopback(action: str, address: Any) -> None:
    host = _address_host(address)
    if is_loopback_host(host):
        return
    raise NetworkDisabledError(
        f"{NETWORK_DISABLED}; non-loopback {action} refused: {address!r}"
    )


class GuardedSocket(_RealSocket):
    def connect(self, address):
        _refuse_non_loopback("connect", address)
        return super().connect(address)

    def connect_ex(self, address):
        _refuse_non_loopback("connect", address)
        return super().connect_ex(address)

    def bind(self, address):
        _refuse_non_loopback("bind", address)
        return super().bind(address)

    def sendto(self, data, *args):
        address = args[-1] if args else None
        _refuse_non_loopback("sendto", address)
        return super().sendto(data, *args)


def _denied_fetch(*args, **kwargs):
    raise NetworkDisabledError(
        f"{NETWORK_DISABLED}. Do not download weights or open WAN sockets."
    )


def _guarded_create_connection(address, *args, **kwargs):
    _refuse_non_loopback("connect", address)
    return _originals["create_connection"](address, *args, **kwargs)


def isolate_weight_caches(root: Optional[str] = None) -> str:
    """Point cache env vars at an empty tree so ~/.cache/whisper is unused."""
    global _cache_root
    if root is None:
        root = tempfile.mkdtemp(prefix="whisper-offline-cache-")
    _cache_root = os.path.realpath(root)
    os.makedirs(_cache_root, exist_ok=True)
    hf = os.path.join(_cache_root, "huggingface")
    mapping = {
        "XDG_CACHE_HOME": _cache_root,
        "HF_HOME": hf,
        "HF_HUB_CACHE": os.path.join(hf, "hub"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(hf, "hub"),
        "TRANSFORMERS_CACHE": os.path.join(hf, "transformers"),
        "TORCH_HOME": os.path.join(_cache_root, "torch"),
        "WHISPER_ALLOW_DOWNLOAD": "0",
    }
    for key, value in mapping.items():
        os.environ[key] = value
    apply_offline_env()
    return _cache_root


def refuse_user_weight_cache(path: str) -> None:
    home = user_whisper_cache()
    real = os.path.realpath(path)
    if real == home or real.startswith(home + os.sep):
        raise OfflineDownloadError(f"user weight cache is ignored in tests: {path}")


def install_offline_guard(cache_root: Optional[str] = None) -> str:
    """Install session-wide fetch/bind guards and isolated caches."""
    global _installed
    root = isolate_weight_caches(cache_root)
    if _installed:
        return root

    import whisper

    _originals["socket"] = socket.socket
    _originals["create_connection"] = socket.create_connection
    _originals["urlopen"] = urllib.request.urlopen
    _originals["download"] = whisper._download

    socket.socket = GuardedSocket
    socket.create_connection = _guarded_create_connection
    urllib.request.urlopen = _denied_fetch

    def guarded_download(url, root_dir, in_memory, download=False):
        refuse_hub_url(url)
        refuse_user_weight_cache(root_dir)
        return _originals["download"](url, root_dir, in_memory, download)

    whisper._download = guarded_download

    try:
        import huggingface_hub

        _originals["hf_hub_download"] = getattr(
            huggingface_hub, "hf_hub_download", None
        )
        _originals["snapshot_download"] = getattr(
            huggingface_hub, "snapshot_download", None
        )
        huggingface_hub.hf_hub_download = _denied_fetch
        huggingface_hub.snapshot_download = _denied_fetch
    except ImportError:
        pass

    try:
        import torch

        if hasattr(torch, "hub"):
            _originals["torch_hub_load"] = torch.hub.load
            torch.hub.load = _denied_fetch
            if hasattr(torch.hub, "load_state_dict_from_url"):
                _originals["torch_hub_url"] = torch.hub.load_state_dict_from_url
                torch.hub.load_state_dict_from_url = _denied_fetch
    except ImportError:
        pass

    _installed = True
    return root


def uninstall_offline_guard() -> None:
    global _installed
    if not _installed:
        return
    socket.socket = _originals["socket"]
    socket.create_connection = _originals["create_connection"]
    urllib.request.urlopen = _originals["urlopen"]
    try:
        import whisper

        if "download" in _originals:
            whisper._download = _originals["download"]
    except ImportError:
        pass
    try:
        import huggingface_hub

        if _originals.get("hf_hub_download") is not None:
            huggingface_hub.hf_hub_download = _originals["hf_hub_download"]
        if _originals.get("snapshot_download") is not None:
            huggingface_hub.snapshot_download = _originals["snapshot_download"]
    except ImportError:
        pass
    try:
        import torch

        if "torch_hub_load" in _originals:
            torch.hub.load = _originals["torch_hub_load"]
        if "torch_hub_url" in _originals:
            torch.hub.load_state_dict_from_url = _originals["torch_hub_url"]
    except ImportError:
        pass
    _originals.clear()
    _installed = False

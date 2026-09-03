"""CPU-only, offline, no-store defaults.

Named checkpoints are not downloaded unless WHISPER_OFFLINE=0.
New cache writes are skipped unless WHISPER_NO_STORE=0.
The default inference device is always CPU (CUDA is never implied).
"""

import os
from typing import Optional

DEFAULT_DEVICE = "cpu"

_OFFLINE_ENV = "WHISPER_OFFLINE"
_NO_STORE_ENV = "WHISPER_NO_STORE"
_DEVICE_ENV = "WHISPER_DEVICE"

_FALSEY = {"0", "false", "no", "off"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSEY


def default_device() -> str:
    """Return the default inference device. Always CPU unless WHISPER_DEVICE is set."""
    raw = os.environ.get(_DEVICE_ENV)
    if raw and raw.strip():
        return raw.strip()
    return DEFAULT_DEVICE


def is_offline() -> bool:
    """Whether remote checkpoint downloads are disabled (default: yes)."""
    return _env_flag(_OFFLINE_ENV, default=True)


def is_no_store() -> bool:
    """Whether newly fetched checkpoints must not be written to disk (default: yes)."""
    return _env_flag(_NO_STORE_ENV, default=True)


def default_cache_root() -> str:
    default = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")


def resolve_download_root(download_root: Optional[str] = None) -> str:
    if download_root:
        return download_root
    return default_cache_root()

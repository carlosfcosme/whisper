"""Runtime defaults: CPU, offline, no-store, loopback bind, no Hub."""

import os

DEFAULT_DEVICE = "cpu"
DEFAULT_OFFLINE = True
DEFAULT_NO_STORE = True
BIND_HOST = "127.0.0.1"
BIND_PORT = 8765

HUB_MARKERS = (
    "huggingface.co",
    "hf.co",
    "huggingface.com",
)

OFFLINE_ENV_VARS = (
    "WHISPER_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)

TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE_VALUES


def env_explicit_false(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in FALSE_VALUES


def offline_enabled() -> bool:
    """Offline is on by default. Opt out with WHISPER_ALLOW_DOWNLOAD=1."""
    if env_flag("WHISPER_ALLOW_DOWNLOAD") or env_flag("WHISPER_ALLOW_WEIGHT_FETCH"):
        return False
    if any(env_flag(key) for key in OFFLINE_ENV_VARS):
        return True
    if env_explicit_false("WHISPER_OFFLINE"):
        return False
    return DEFAULT_OFFLINE


def no_store_enabled() -> bool:
    """Do not persist downloaded weights by default."""
    if env_flag("WHISPER_ALLOW_STORE"):
        return False
    if env_explicit_false("WHISPER_NO_STORE"):
        return False
    if env_flag("WHISPER_NO_STORE"):
        return True
    return DEFAULT_NO_STORE


def is_hub_url(url: str) -> bool:
    lowered = (url or "").lower()
    return any(marker in lowered for marker in HUB_MARKERS)


def refuse_remote_download(url: str, dest: str) -> None:
    """Raise if a Hub, offline, or no-store remote weight pull is attempted."""
    if is_hub_url(url):
        raise RuntimeError(
            "no Hub: refusing Hugging Face Hub download ({}); "
            "use a local checkpoint".format(url)
        )
    if offline_enabled():
        raise RuntimeError(
            "offline: no weight pulls; missing local checkpoint {}".format(dest)
        )
    if no_store_enabled():
        raise RuntimeError("no-store: refusing to persist weights at {}".format(dest))

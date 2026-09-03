"""Sovereign defaults: CPU inference, loopback bind, no Hub / weight pulls."""

import os

DEFAULT_DEVICE = "cpu"
BIND_HOST = "127.0.0.1"
BIND_PORT = 8765
ALL_INTERFACES = "0.0.0.0"

HUB_MARKERS = (
    "huggingface.co",
    "hf.co",
    "huggingface.com",
    "cdn-lfs.huggingface.co",
)

OFFLINE_ENV_VARS = (
    "WHISPER_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def offline_enabled() -> bool:
    return any(env_flag(key) for key in OFFLINE_ENV_VARS)


def is_hub_url(url: str) -> bool:
    lowered = (url or "").lower()
    return any(marker in lowered for marker in HUB_MARKERS)


def refuse_remote_download(url: str, dest: str) -> None:
    """Raise if a remote weight pull is forbidden.

    Hugging Face Hub URLs are always refused. Any other remote pull is
    refused when an offline environment flag is set (CI / unit tests).
    """
    if is_hub_url(url):
        raise RuntimeError(
            "no Hub: refusing Hugging Face Hub download ({}); "
            "use a local checkpoint".format(url)
        )
    if offline_enabled():
        raise RuntimeError(
            "offline: no weight pulls; missing local checkpoint {}".format(dest)
        )

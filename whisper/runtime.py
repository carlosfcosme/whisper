"""Runtime policy: CPU default, localhost bind, no Hugging Face Hub."""

import os

DEFAULT_DEVICE = "cpu"
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


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def offline_enabled() -> bool:
    return any(env_flag(key) for key in OFFLINE_ENV_VARS)


def is_hub_url(url: str) -> bool:
    lowered = (url or "").lower()
    return any(marker in lowered for marker in HUB_MARKERS)


def guard_weight_download(url: str, dest: str) -> None:
    """Raise before any remote weight pull that violates Hub/offline policy."""
    if is_hub_url(url):
        raise RuntimeError(
            "no Hub: refusing Hugging Face Hub download ({}); "
            "use a local checkpoint".format(url)
        )
    if offline_enabled():
        raise RuntimeError(
            "offline: no weight pulls; missing local checkpoint {}".format(dest)
        )

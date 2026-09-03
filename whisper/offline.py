"""No Hub and no weight-pull policy.

Cache hits and local checkpoint paths are allowed. Hugging Face Hub URLs
are always refused. Remote weight pulls (Azure CDN included) are refused
unless WHISPER_ALLOW_WEIGHT_DOWNLOAD is set.
"""

import os
from urllib.parse import urlparse

ALLOW_PULL_ENV = "WHISPER_ALLOW_WEIGHT_DOWNLOAD"
OFFLINE_ENV_VARS = (
    "WHISPER_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)

_HUB_HOSTS = frozenset({"huggingface.co", "hf.co", "huggingface.com"})
_HUB_SUFFIXES = (".huggingface.co", ".hf.co")
_TRUTHY = frozenset({"1", "true", "yes", "on"})


class WeightDownloadError(RuntimeError):
    """Raised when a Hub or remote weight pull is refused."""


def _env_truthy(name):
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def offline_enabled():
    return any(_env_truthy(key) for key in OFFLINE_ENV_VARS)


def is_hub_url(url):
    host = (urlparse(url or "").hostname or "").lower().rstrip(".")
    if host:
        return host in _HUB_HOSTS or host.endswith(_HUB_SUFFIXES)
    lowered = (url or "").lower()
    return any(marker in lowered for marker in _HUB_HOSTS)


def weight_pull_allowed():
    if _env_truthy(ALLOW_PULL_ENV):
        return True
    return False


def refuse_remote_download(url, dest=None):
    if is_hub_url(url):
        raise WeightDownloadError(
            "no Hub: refusing Hugging Face Hub download ({})".format(url)
        )
    if weight_pull_allowed():
        return
    suffix = " {}".format(dest) if dest else ""
    raise WeightDownloadError(
        "no weight pull: missing local checkpoint{}".format(suffix)
    )

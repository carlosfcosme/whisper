"""Runtime policy: CPU default, loopback bind, no Hub/weight WAN fetch."""

from .bind import BIND_HOST, require_bind_127_0_0_1

DEFAULT_DEVICE = "cpu"

_HUB_MARKERS = (
    "huggingface.co",
    "hf.co/",
    "hf-mirror.com",
    "huggingface_hub",
    "cas-bridge.xethub",
)
_WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors")
_WEIGHT_HOST_MARKERS = (
    "openaipublic.azureedge.net",
    "whisper/models",
)


def default_device():
    return DEFAULT_DEVICE


def service_bind_host():
    return BIND_HOST


def _url_text(url):
    if hasattr(url, "full_url"):
        return url.full_url
    return str(url)


def is_hub_url(url):
    lowered = _url_text(url).lower()
    return any(marker in lowered for marker in _HUB_MARKERS)


def is_weight_url(url):
    lowered = _url_text(url).lower()
    if any(lowered.endswith(suffix) for suffix in _WEIGHT_SUFFIXES):
        return True
    return any(marker in lowered for marker in _WEIGHT_HOST_MARKERS)


def refuse_forbidden_fetch(url, offline=False):
    """Raise before a Hub or (when offline) weight URL is opened."""
    if is_hub_url(url):
        raise RuntimeError("Hugging Face Hub fetch is refused: %s" % (_url_text(url),))
    if offline:
        raise RuntimeError(
            "WHISPER_OFFLINE is set; refusing to fetch model weights from the network"
        )


__all__ = [
    "BIND_HOST",
    "DEFAULT_DEVICE",
    "default_device",
    "is_hub_url",
    "is_weight_url",
    "refuse_forbidden_fetch",
    "require_bind_127_0_0_1",
    "service_bind_host",
]

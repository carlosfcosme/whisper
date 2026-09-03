"""Runtime policy: CPU default, loopback bind, no Hub/weight WAN fetch."""

DEFAULT_DEVICE = "cpu"
BIND_HOST = "127.0.0.1"

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

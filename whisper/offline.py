"""Refuse Hub/CDN model fetches; only loopback HTTP or a local cache."""

from urllib.parse import urlparse

LOOPBACK_HOST = "127.0.0.1"
LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})


class OfflineFetchError(RuntimeError):
    """Raised when a non-loopback model or Hub download is refused."""


def url_host(url):
    return (urlparse(url).hostname or "").lower()


def is_loopback_url(url):
    return url_host(url) in LOOPBACK_NAMES


def is_hub_host(host):
    host = (host or "").lower()
    return (
        host == "huggingface.co"
        or host.endswith(".huggingface.co")
        or host == "hf.co"
        or host.endswith(".hf.co")
    )


def refuse_remote_fetch(url, download_target):
    """Raise if ``url`` is not a loopback fetch (Hub and CDN included)."""
    host = url_host(url)
    if is_hub_host(host):
        raise OfflineFetchError(
            "Hugging Face Hub downloads are disabled. Place the checkpoint at "
            "{0} or serve it on {1}.".format(download_target, LOOPBACK_HOST)
        )
    if not is_loopback_url(url):
        raise OfflineFetchError(
            "Refusing remote model download from {0!r}. "
            "Place the checkpoint at {1} or serve it on {2}.".format(
                host or url, download_target, LOOPBACK_HOST
            )
        )

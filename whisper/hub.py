"""Refuse Hugging Face Hub downloads.

Official Whisper checkpoints are fetched from OpenAI's Azure CDN, not
the Hub. Named models keep that path. Hub hosts are always rejected.
"""

from urllib.parse import urlparse

HUB_ERROR_MESSAGE = (
    "Hugging Face Hub downloads are disabled. "
    "Use an official model name, a local checkpoint path, or "
    "http://127.0.0.1/... ."
)


class HubError(RuntimeError):
    """Raised when a Hugging Face Hub URL or client is refused."""


def _hostname(url: str) -> str:
    return (urlparse(url).hostname or "").lower().rstrip(".")


def is_hub_host(host: str) -> bool:
    """True for huggingface.co, hf.co, and their CDN subdomains."""
    if not host:
        return False
    normalized = host.lower().rstrip(".")
    if normalized == "hf.co" or normalized.endswith(".hf.co"):
        return True
    if normalized == "huggingface.co" or normalized.endswith(".huggingface.co"):
        return True
    return "huggingface" in normalized


def is_hub_url(url: str) -> bool:
    """True when ``url`` points at the Hugging Face Hub or its CDN."""
    if not url:
        return False
    host = _hostname(url)
    if host:
        return is_hub_host(host)
    lowered = str(url).lower()
    return "huggingface" in lowered or "hf.co/" in lowered


def refuse_hub_url(url: str) -> None:
    """Raise ``HubError`` when ``url`` is a Hub (or Hub CDN) address."""
    if is_hub_url(url):
        host = _hostname(url) or "<unknown>"
        raise HubError("{0} Refused host {1!r}.".format(HUB_ERROR_MESSAGE, host))

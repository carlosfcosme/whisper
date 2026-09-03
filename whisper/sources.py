"""Runtime policy: CPU by default, local sources only, no Hugging Face Hub."""

DEFAULT_DEVICE = "cpu"
DEFAULT_BIND_HOST = "127.0.0.1"

HUB_HINTS = (
    "huggingface.co",
    "huggingface.com",
    "hf.co",
    "hf://",
    "hf-mirror.com",
    "hf_hub:",
    "hub://",
)


def default_device():
    """Return the default inference device (CPU)."""
    return DEFAULT_DEVICE


def is_hub_source(source):
    """True if *source* refers to a Hugging Face Hub URL or protocol."""
    if source is None:
        return False
    text = str(source).strip().lower()
    return any(hint in text for hint in HUB_HINTS)


def reject_hub_source(source):
    """Raise ValueError when *source* is a Hugging Face Hub location."""
    if is_hub_source(source):
        raise ValueError(
            "Hugging Face Hub sources are not supported: {!r}. "
            "Use an official model name or a local checkpoint path.".format(source)
        )


def require_loopback_host(host):
    """Refuse any bind address other than 127.0.0.1."""
    if host != DEFAULT_BIND_HOST:
        raise ValueError(
            "Refusing to bind {!r}; this server only listens on {}".format(
                host, DEFAULT_BIND_HOST
            )
        )
    return host

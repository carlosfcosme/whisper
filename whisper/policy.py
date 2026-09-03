"""Runtime policy: offline default, no Hub, CPU-only, localhost bind."""

import os

DEFAULT_DEVICE = "cpu"
BIND_HOST = "127.0.0.1"
ALLOWED_BIND_HOSTS = frozenset({"127.0.0.1", "::1"})

# Environment variable names encoded in STATUS.md and .cursor/install.sh.
ENV_OFFLINE = "WHISPER_OFFLINE"
ENV_NO_HUB = "WHISPER_NO_HUB"


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def offline_enabled() -> bool:
    """True by default. Set WHISPER_OFFLINE=0 to allow network fetches."""
    return _env_flag(ENV_OFFLINE, default=True)


def hub_disabled() -> bool:
    """True by default. Set WHISPER_NO_HUB=0 to allow Hub/CDN fetches."""
    return _env_flag(ENV_NO_HUB, default=True)


def allow_remote_fetch() -> bool:
    """Remote checkpoint fetch is off unless both offline and no-Hub are disabled."""
    return (not offline_enabled()) and (not hub_disabled())


def refuse_remote_fetch_message(url: str) -> str:
    return (
        "Offline/no-Hub default: refusing to download "
        f"{url}. Set {ENV_OFFLINE}=0 and {ENV_NO_HUB}=0 to fetch."
    )

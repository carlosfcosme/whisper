"""Offline post-install bootstrap: no weight fetch, loopback bind only."""

from __future__ import annotations

import json
import sys
from typing import Dict

from .bind import require_loopback_host
from .defaults import (
    DEFAULT_BIND_HOST,
    DEFAULT_DEVICE,
    DEFAULT_NO_STORE,
    DEFAULT_OFFLINE,
    no_store_enabled,
    offline_enabled,
)


def offline_bootstrap() -> Dict[str, object]:
    """Validate CPU / offline / no-store defaults without downloading weights."""
    if DEFAULT_DEVICE != "cpu":
        raise RuntimeError("bootstrap requires CPU-only DEFAULT_DEVICE")
    if not DEFAULT_OFFLINE or not offline_enabled():
        raise RuntimeError("bootstrap requires offline defaults")
    if not DEFAULT_NO_STORE or not no_store_enabled():
        raise RuntimeError("bootstrap requires no-store defaults")
    bind = require_loopback_host(DEFAULT_BIND_HOST)
    if bind != "127.0.0.1":
        raise RuntimeError(f"bootstrap requires loopback bind, got {bind!r}")
    return {
        "device": DEFAULT_DEVICE,
        "bind": bind,
        "offline": True,
        "no_store": True,
        "weights": False,
        "fetched": False,
    }


def main(argv=None) -> int:
    del argv
    try:
        info = offline_bootstrap()
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(json.dumps(info, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

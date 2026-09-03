#!/usr/bin/env python3
"""Fail CI unless checkpoint fetches are blocked when WHISPER_NO_DOWNLOAD=1."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

AZURE = "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
HUB = "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors"


def _load_hub(path: Path):
    spec = importlib.util.spec_from_file_location("whisper_hub_offline_check", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    hub = _load_hub(root / "whisper" / "hub.py")

    os.environ.pop(hub.NO_DOWNLOAD_ENV, None)
    try:
        hub.assert_can_fetch(AZURE)
    except hub.DownloadBlockedError:
        print(
            "ERROR: official URL blocked without WHISPER_NO_DOWNLOAD", file=sys.stderr
        )
        return 1
    except hub.HubDisabledError:
        print("ERROR: official Azure URL treated as Hub", file=sys.stderr)
        return 1

    os.environ[hub.NO_DOWNLOAD_ENV] = "1"
    try:
        hub.assert_can_fetch(AZURE)
    except hub.DownloadBlockedError:
        pass
    else:
        print(
            "ERROR: official URL fetch allowed when downloads are blocked",
            file=sys.stderr,
        )
        return 1

    try:
        hub.assert_can_fetch(HUB)
    except hub.HubDisabledError:
        pass
    else:
        print("ERROR: Hub URL was allowed", file=sys.stderr)
        return 1

    print("OK: model downloads are blocked when WHISPER_NO_DOWNLOAD=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())

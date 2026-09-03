#!/usr/bin/env python3
"""CI guard: default device is CPU; Hub URLs and cache-miss pulls are refused.

Loads whisper/offline.py by path (stdlib only). No torch, no Hub client,
no weight download.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_offline(root: Path):
    path = root / "whisper" / "offline.py"
    spec = importlib.util.spec_from_file_location("whisper_offline_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    root = repo_root()
    offline = load_offline(root)
    errors = []
    if offline.DEFAULT_DEVICE != "cpu":
        errors.append("DEFAULT_DEVICE is not cpu")
    if offline.default_device() != "cpu":
        errors.append("default_device() is not cpu")
    hub = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"
    try:
        offline.refuse_weight_auto_download(hub)
    except offline.WeightDownloadError:
        pass
    else:
        errors.append("Hub URL was not refused")
    if not offline.is_hf_hub_url(hub):
        errors.append("is_hf_hub_url did not detect Hub")
    if errors:
        sys.stderr.write("ERROR: CPU / offline policy failed\n")
        for message in errors:
            sys.stderr.write("  {}\n".format(message))
        return 1
    sys.stdout.write("OK: CPU default; Hub and weight-pull policy in place\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

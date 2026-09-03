#!/usr/bin/env python3
"""Prove weight downloads (Hub and Azure cache-miss) are refused.

Loads whisper/runtime.py without importing torch or opening a network socket.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"
AZURE_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
)


def _load_runtime():
    spec = importlib.util.spec_from_file_location(
        "whisper_runtime", ROOT / "whisper" / "runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    os.environ.setdefault("WHISPER_NO_WEIGHT_DOWNLOAD", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    runtime = _load_runtime()
    errors = []

    init_text = (ROOT / "whisper" / "__init__.py").read_text(encoding="utf-8")
    if "refuse_weight_auto_download(url)" not in init_text:
        errors.append("whisper/_download is not wired to refuse_weight_auto_download")

    if not runtime.is_hf_hub_url(HF_HUB_URL):
        errors.append("huggingface.co URL was not classified as Hub")

    try:
        runtime.refuse_weight_auto_download(HF_HUB_URL)
        errors.append("Hub URL was not refused")
    except runtime.WeightDownloadError:
        pass

    try:
        runtime.refuse_weight_auto_download(AZURE_URL)
        errors.append("Azure checkpoint URL was not refused")
    except runtime.WeightDownloadError:
        pass

    if errors:
        sys.stderr.write("ERROR: downloads are not blocked:\n")
        for item in errors:
            sys.stderr.write("  - {}\n".format(item))
        return 1

    sys.stdout.write("OK: Hub and cache-miss weight downloads are refused\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail if a test run downloaded model weights or a Hub cache.

Used as a CI post-step after the weight-free pytest invocation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

WEIGHT_SUFFIXES = (
    ".pt",
    ".pth",
    ".bin",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".gguf",
    ".ggml",
)


def _cache_roots() -> List[Path]:
    home = Path.home()
    xdg = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
    hf_home: Optional[Path] = (
        Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else None
    )
    hub_cache: Optional[Path] = (
        Path(os.environ["HUGGINGFACE_HUB_CACHE"])
        if os.environ.get("HUGGINGFACE_HUB_CACHE")
        else None
    )
    roots = [
        xdg / "whisper",
        home / ".cache" / "whisper",
        hf_home,
        hub_cache,
        xdg / "huggingface",
        home / ".cache" / "huggingface",
        home / ".huggingface",
    ]
    return [path for path in roots if path is not None]


def _weight_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    found = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES:
            found.append(path)
    return found


def find_downloads() -> List[str]:
    hits: List[str] = []
    for root in _cache_roots():
        for path in _weight_files(root):
            hits.append(str(path))
    return hits


def main() -> int:
    hits = find_downloads()
    if hits:
        sys.stderr.write("ERROR: weight or Hub download detected after tests:\n")
        for path in hits:
            sys.stderr.write("  {}\n".format(path))
        return 1
    sys.stdout.write("OK: no weight or Hub cache downloads\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

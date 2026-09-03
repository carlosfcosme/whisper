#!/usr/bin/env python3
"""Fail if install or tests wrote model weights into a local cache."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

WEIGHT_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".gguf",
        ".ggml",
        ".bin",
    }
)


def cache_roots() -> List[Path]:
    home = Path.home()
    xdg = os.environ.get("XDG_CACHE_HOME")
    roots: List[Optional[Path]] = [
        home / ".cache" / "whisper",
        Path(xdg) / "whisper" if xdg else None,
        home / ".cache" / "huggingface" / "hub",
        Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else None,
        Path(os.environ["HF_HUB_CACHE"]) if os.environ.get("HF_HUB_CACHE") else None,
    ]
    unique: List[Path] = []
    seen = set()
    for root in roots:
        if root is None:
            continue
        key = str(root.expanduser())
        if key in seen:
            continue
        seen.add(key)
        unique.append(root.expanduser())
    return unique


def find_cached_weights(roots: Optional[Iterable[Path]] = None) -> List[Path]:
    found: List[Path] = []
    for root in roots if roots is not None else cache_roots():
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES:
                found.append(path)
    return sorted(found)


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv
    found = find_cached_weights()
    if found:
        sys.stderr.write("ERROR: install/test path downloaded model weights:\n")
        for path in found:
            sys.stderr.write("  {}\n".format(path))
        return 1
    sys.stdout.write("OK: no model weights in install/test caches\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

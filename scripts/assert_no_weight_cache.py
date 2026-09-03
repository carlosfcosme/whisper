#!/usr/bin/env python3
"""Fail if install or tests wrote model weights into a local cache."""

import os
import sys
from pathlib import Path

WEIGHT_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".bin",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".h5",
        ".hdf5",
        ".gguf",
        ".ggml",
        ".tflite",
        ".pb",
        ".mar",
        ".params",
        ".weights",
    }
)


def cache_roots():
    home = Path.home()
    xdg = os.environ.get("XDG_CACHE_HOME")
    roots = [
        home / ".cache" / "whisper",
        Path(xdg) / "whisper" if xdg else None,
        home / ".cache" / "huggingface" / "hub",
        Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else None,
        Path(os.environ["HF_HUB_CACHE"]) if os.environ.get("HF_HUB_CACHE") else None,
    ]
    seen = set()
    unique = []
    for root in roots:
        if root is None:
            continue
        key = str(root.expanduser())
        if key in seen:
            continue
        seen.add(key)
        unique.append(root.expanduser())
    return unique


def find_cached_weights(roots=None):
    found = []
    for root in roots if roots is not None else cache_roots():
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES:
                found.append(path)
    return sorted(found)


def main(argv=None):
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
    sys.exit(main())

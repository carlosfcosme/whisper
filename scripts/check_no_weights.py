#!/usr/bin/env python3
"""Fail if model weights or large binaries are committed to git."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

WEIGHT_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".bin",
        ".safetensors",
        ".onnx",
        ".ckpt",
        ".h5",
        ".hdf5",
        ".gguf",
        ".tflite",
        ".pb",
        ".msgpack",
        ".ot",
        ".pkl",
    }
)
WEIGHT_DIR_PARTS = frozenset({"cache", ".cache", "weights", "checkpoints"})
ALLOWED_PATHS = frozenset({"whisper/assets/mel_filters.npz"})
MAX_BYTES = 10 * 1024 * 1024


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def is_committed_weight(path: str, size: Optional[int] = None) -> bool:
    norm = normalize_path(path)
    if norm in ALLOWED_PATHS:
        return False
    base = os.path.basename(norm)
    _, ext = os.path.splitext(base)
    if ext.lower() in WEIGHT_SUFFIXES:
        return True
    parts = set(norm.split("/"))
    if parts & WEIGHT_DIR_PARTS:
        return True
    if size is not None and size > MAX_BYTES:
        return True
    return False


def tracked_files() -> List[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    if not result.stdout:
        return []
    return [p.decode("utf-8") for p in result.stdout.split(b"\0") if p]


def file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def violations(
    paths: Optional[Iterable[str]] = None,
) -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    for path in paths if paths is not None else tracked_files():
        size = file_size(path) if os.path.exists(path) else 0
        if not is_committed_weight(path, size):
            continue
        reason = "weight/cache path"
        _, ext = os.path.splitext(os.path.basename(path))
        if ext.lower() in WEIGHT_SUFFIXES:
            reason = f"weight suffix {ext.lower()}"
        elif size > MAX_BYTES:
            reason = f"file larger than {MAX_BYTES} bytes"
        found.append((normalize_path(path), reason))
    return found


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv
    bad = violations()
    if not bad:
        print("no committed weights")
        return 0
    print("committed weights are not allowed:", file=sys.stderr)
    for path, reason in bad:
        print(f"  {path} ({reason})", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

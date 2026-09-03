#!/usr/bin/env python3
"""Fail if the git index tracks model weights or oversized binaries."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors", ".onnx", ".ckpt")
WEIGHT_DIRS = (".cache/", "cache/", "weights/")
MAX_BYTES = 10 * 1024 * 1024


def _tracked_files():
    listed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    return [path for path in listed.stdout.decode().split("\0") if path]


def main() -> int:
    offenders = []
    for rel in _tracked_files():
        if rel.startswith(WEIGHT_DIRS) or rel.endswith(WEIGHT_SUFFIXES):
            offenders.append(f"weight path: {rel}")
            continue
        path = REPO / rel
        if path.is_file() and path.stat().st_size > MAX_BYTES:
            offenders.append(f"oversized ({path.stat().st_size} bytes): {rel}")
    if offenders:
        print("committed weights / oversized files:", file=sys.stderr)
        for line in offenders:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

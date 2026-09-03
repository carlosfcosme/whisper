#!/usr/bin/env python3
"""Fail if model weights or oversized binaries are committed."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SUFFIXES = (
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".gguf",
)
MAX_BYTES = 10 * 1024 * 1024


def _tracked_entries() -> List[Tuple[str, int]]:
    listing = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO_ROOT)
    entries = []
    for raw in listing.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", "surrogateescape")
        path = REPO_ROOT / rel
        size = path.stat().st_size if path.is_file() else 0
        entries.append((rel, size))
    return entries


def reasons_committed_weights(
    entries: Sequence[Tuple[str, int]] | None = None,
) -> List[str]:
    if entries is None:
        entries = _tracked_entries()
    reasons: List[str] = []
    for rel, size in entries:
        lower = rel.lower()
        if any(lower.endswith(suffix) for suffix in WEIGHT_SUFFIXES):
            reasons.append(f"tracked weight file: {rel}")
        if size > MAX_BYTES:
            reasons.append(f"tracked file exceeds 10 MiB: {rel} ({size} bytes)")
    return reasons


def main() -> int:
    reasons = reasons_committed_weights()
    if reasons:
        print("FAIL: committed weights or oversized binaries:")
        for reason in reasons:
            print(f"  - {reason}")
        return 1
    print("OK: no committed weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())

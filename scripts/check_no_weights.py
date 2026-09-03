#!/usr/bin/env python3
"""Fail CI if model weights or large binaries are committed.

Weights belong in a local cache, not in git. Invoked from GitHub Actions
and pre-commit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

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
        ".npz",
    }
)

BINARY_SUFFIXES = frozenset(
    {
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".whl",
        ".egg",
        ".a",
        ".o",
    }
)

# Official Whisper checkpoints start around 75 MiB; existing fixtures stay
# under this limit (largest tracked file is the approach diagram, ~0.9 MiB).
MAX_FILE_BYTES = 10 * 1024 * 1024

# Small non-weight assets that share a weight-like suffix.
ALLOWLIST = frozenset(
    {
        "whisper/assets/mel_filters.npz",
    }
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tracked_files(root: Path) -> List[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def classify(relpath: str, size: int) -> Optional[str]:
    if relpath in ALLOWLIST:
        return None
    suffix = Path(relpath).suffix.lower()
    name = Path(relpath).name.lower()
    if suffix in WEIGHT_SUFFIXES or name.endswith(".safetensors"):
        return f"committed weight file: {relpath} ({size} bytes)"
    if suffix in BINARY_SUFFIXES:
        return f"committed binary artifact: {relpath} ({size} bytes)"
    if size > MAX_FILE_BYTES:
        return f"committed file exceeds {MAX_FILE_BYTES} bytes: {relpath} ({size})"
    return None


def findings(root: Path, paths: Optional[Iterable[str]] = None) -> List[str]:
    issues: List[str] = []
    for relpath in paths if paths is not None else tracked_files(root):
        full = root / relpath
        size = full.stat().st_size if full.is_file() else 0
        reason = classify(relpath, size)
        if reason:
            issues.append(reason)
    return issues


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv  # reserved for future path filters
    root = repo_root()
    issues = findings(root)
    if issues:
        print("Committed model weights / binaries are not allowed:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1
    print("No committed model weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

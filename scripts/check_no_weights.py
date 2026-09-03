#!/usr/bin/env python3
"""CI assertion: no committed model weights or cache directories.

Weights belong in a local cache, not git. Stdlib only. No network.
No secrets.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

WEIGHT_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".gguf",
        ".ggml",
        ".tflite",
        ".pb",
        ".mar",
        ".params",
        ".weights",
    }
)
CACHE_PARTS = frozenset({".cache", "cache", "weights"})
MAX_FILE_BYTES = 10 * 1024 * 1024


def fail(message: str) -> None:
    print("FAIL: {}".format(message), file=sys.stderr)
    raise SystemExit(1)


def tracked_files(root: Path) -> List[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=str(root))
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def classify(relpath: str, size: int) -> Optional[str]:
    posix = relpath.replace("\\", "/")
    parts = set(Path(posix).parts)
    suffix = Path(posix).suffix.lower()
    if parts & CACHE_PARTS:
        return "cache/weight directory path"
    if suffix in WEIGHT_SUFFIXES:
        return "model weight or checkpoint ({})".format(suffix)
    if size > MAX_FILE_BYTES:
        return "large file ({} bytes > {})".format(size, MAX_FILE_BYTES)
    return None


def find_violations(
    root: Path, relative_paths: Optional[Sequence[str]] = None
) -> List[Tuple[str, str]]:
    paths: Iterable[str] = (
        relative_paths if relative_paths is not None else tracked_files(root)
    )
    violations: List[Tuple[str, str]] = []
    for relpath in paths:
        path = root / relpath
        if not path.is_file():
            continue
        size = path.stat().st_size
        reason = classify(relpath, size)
        if reason:
            violations.append((relpath, reason))
    return violations


def main() -> int:
    violations = find_violations(ROOT)
    if violations:
        for relpath, reason in violations:
            print("FAIL: {}: {}".format(relpath, reason), file=sys.stderr)
        raise SystemExit(1)
    print("no-weights: ok (no committed checkpoints or caches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

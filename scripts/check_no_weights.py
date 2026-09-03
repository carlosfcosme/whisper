#!/usr/bin/env python3
"""Fail CI if model weights or large binaries are committed.

``--probe-negative`` plants a checkpoint in a temp tree and fails if the
classifier would miss it. No Hub fetch and no weight download.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

WEIGHT_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".onnx",
        ".safetensors",
        ".ckpt",
        ".ggml",
        ".gguf",
        ".h5",
        ".hdf5",
        ".tflite",
        ".pb",
        ".mlmodel",
        ".weights",
        ".bin",
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

# Official Whisper checkpoints start around 75 MiB. Existing fixtures stay under
# this limit (largest tracked file is notebooks/Multilingual_ASR.ipynb, ~5.7 MiB).
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
    """Return a violation reason, or None if the file is allowed."""
    posix = relpath.replace("\\", "/")
    if posix in ALLOWLIST:
        return None
    suffix = Path(posix).suffix.lower()
    if suffix in WEIGHT_SUFFIXES:
        return "model weight or checkpoint ({})".format(suffix)
    if suffix in BINARY_SUFFIXES:
        return "committed binary ({})".format(suffix)
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
        reason = classify(relpath, path.stat().st_size)
        if reason:
            violations.append((relpath, reason))
    return violations


def probe_negative() -> int:
    """Return 0 if a planted checkpoint is classified as a violation."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        relpath = "models/tiny.pt"
        path = root / relpath
        path.parent.mkdir()
        path.write_bytes(b"not-a-real-checkpoint")
        hits = find_violations(root, relative_paths=[relpath])
        if not hits:
            sys.stderr.write("ERROR: weight checker missed planted checkpoint\n")
            return 1
    sys.stdout.write("OK: negative probe flagged planted checkpoint\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail if model weights or large binaries are committed"
    )
    parser.add_argument(
        "--probe-negative",
        action="store_true",
        help="plant a checkpoint and assert the checker flags it",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.probe_negative:
        return probe_negative()
    root = repo_root()
    violations = find_violations(root)
    if violations:
        sys.stderr.write(
            "ERROR: model weights or large binaries must not be committed:\n"
        )
        for relpath, reason in violations:
            sys.stderr.write("  {}: {}\n".format(relpath, reason))
        sys.stderr.write("Do not add checkpoints to git.\n")
        return 1
    sys.stdout.write("OK: no model weights or oversized binaries in the git tree\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

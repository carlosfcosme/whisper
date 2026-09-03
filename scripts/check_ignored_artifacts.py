#!/usr/bin/env python3
"""Fail CI if weight/cache artifacts are tracked or not gitignored."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

# Paths that must never be committed. git check-ignore must match each one.
IGNORE_SAMPLES: Sequence[str] = (
    "tiny.pt",
    "model.pth",
    "model.safetensors",
    "encoder.bin",
    "weights/tiny.pt",
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    ".huggingface/hub/models--openai--whisper/tiny.pt",
    "hub/tiny.safetensors",
)

TRACKED_GLOBS: Sequence[str] = (
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.bin",
    "*.ckpt",
    "*.onnx",
    ".cache",
    ".cache/**",
    "cache",
    "cache/**",
    "weights",
    "weights/**",
    ".huggingface",
    ".huggingface/**",
    "hub",
    "hub/**",
)


def tracked_artifact_paths(root: Path) -> List[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--"] + list(TRACKED_GLOBS),
        cwd=root,
    )
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def unignored_samples(root: Path) -> List[str]:
    missing: List[str] = []
    for sample in IGNORE_SAMPLES:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", sample],
            cwd=root,
        )
        if result.returncode != 0:
            missing.append(sample)
    return missing


def main() -> int:
    tracked = tracked_artifact_paths(REPO_ROOT)
    missing = unignored_samples(REPO_ROOT)
    failed = False
    if tracked:
        sys.stderr.write("ERROR: weight/cache artifacts must not be tracked:\n")
        for path in tracked:
            sys.stderr.write("  {}\n".format(path))
        failed = True
    if missing:
        sys.stderr.write("ERROR: these weight/cache paths must be gitignored:\n")
        for path in missing:
            sys.stderr.write("  {}\n".format(path))
        failed = True
    if failed:
        return 1
    sys.stdout.write("OK: weight/cache artifacts are ignored and untracked\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

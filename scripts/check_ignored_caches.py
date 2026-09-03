#!/usr/bin/env python3
"""Fail CI if cache or weight paths are not gitignored."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

IGNORED_SAMPLES = (
    ".cache/whisper/tiny.pt",
    ".huggingface/hub/models--openai--whisper/config.json",
    "huggingface/hub/snapshot",
    "hf_cache/models--openai--whisper/config.json",
    ".hub/models--openai--whisper/blobs/abc",
    ".torch/hub/checkpoints/model.pt",
    "model.pt",
    "model.pth",
    "model.safetensors",
    "export/whisper.onnx",
    "weights/encoder.bin",
    "checkpoint.ckpt",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_ignored(root: Path, relpath: str) -> bool:
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "--", relpath],
        cwd=root,
        check=False,
    )
    return proc.returncode == 0


def unignored_samples(
    root: Path, samples: Sequence[str] = IGNORED_SAMPLES
) -> List[str]:
    return [path for path in samples if not is_ignored(root, path)]


def main() -> int:
    root = repo_root()
    missing = unignored_samples(root)
    if missing:
        sys.stderr.write("ERROR: cache/weight paths are not gitignored:\n")
        for path in missing:
            sys.stderr.write("  {}\n".format(path))
        return 1
    sys.stdout.write("OK: cache and weight paths are gitignored\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

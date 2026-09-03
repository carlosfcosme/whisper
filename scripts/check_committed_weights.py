#!/usr/bin/env python3
"""Fail if model weight files are tracked in git.

This check is stdlib-only so CI can run it without installing the package
or downloading checkpoints.
"""

import os
import subprocess
import sys
from typing import List, Optional

WEIGHT_SUFFIXES = (
    ".pt",
    ".pth",
    ".bin",
    ".onnx",
    ".safetensors",
    ".ckpt",
    ".h5",
    ".tflite",
    ".gguf",
    ".ggml",
    ".msgpack",
)

WEIGHT_BASENAMES = (
    "pytorch_model.bin",
    "model.safetensors",
    "adapter_model.bin",
    "adapter_model.safetensors",
    "diffusion_pytorch_model.bin",
    "diffusion_pytorch_model.safetensors",
)


def _git_ls_files(repo_root: str) -> List[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [path for path in result.stdout.decode("utf-8").split("\0") if path]


def is_weight_path(path: str) -> bool:
    name = os.path.basename(path)
    if name in WEIGHT_BASENAMES:
        return True
    lower = path.lower()
    return any(lower.endswith(suffix) for suffix in WEIGHT_SUFFIXES)


def list_committed_weights(repo_root: str) -> List[str]:
    return [path for path in _git_ls_files(repo_root) if is_weight_path(path)]


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = argv[0] if argv else os.getcwd()
    tracked = list_committed_weights(repo_root)
    if not tracked:
        print("OK: no committed model weight files")
        return 0
    print("ERROR: committed model weight files are not allowed:", file=sys.stderr)
    for path in tracked:
        print(f"  {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

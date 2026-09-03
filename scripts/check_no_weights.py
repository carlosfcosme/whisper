#!/usr/bin/env python3
"""Fail if git tracks model-weight files. Used by CI."""

import subprocess
import sys

WEIGHT_SUFFIXES = (
    ".pt",
    ".pth",
    ".onnx",
    ".safetensors",
    ".ckpt",
    ".gguf",
)


def is_weight_path(path):
    lower = path.lower()
    return any(lower.endswith(suffix) for suffix in WEIGHT_SUFFIXES)


def tracked_weight_paths(repo_root="."):
    output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
    )
    paths = [item.decode("utf-8") for item in output.split(b"\0") if item]
    return [path for path in paths if is_weight_path(path)]


def main(argv=None):
    hits = tracked_weight_paths()
    if hits:
        sys.stderr.write("Committed model weight files are not allowed:\n")
        for path in hits:
            sys.stderr.write(f"  {path}\n")
        return 1
    sys.stdout.write("ok: no committed model weights\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

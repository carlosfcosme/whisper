#!/usr/bin/env python3
"""Fail if model weight checkpoints are tracked in git.

Stdlib only so CI can run this before installing PyTorch.
"""

import subprocess
import sys

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
)


def is_committed_weight(path: str) -> bool:
    name = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return any(name.endswith(suffix) for suffix in WEIGHT_SUFFIXES)


def iter_tracked_weights(paths):
    for path in paths:
        if path and is_committed_weight(path):
            yield path


def git_tracked_paths():
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], stderr=subprocess.STDOUT
    )
    if not output:
        return []
    return [part.decode("utf-8") for part in output.split(b"\0") if part]


def main(argv=None):
    del argv
    tracked = list(iter_tracked_weights(git_tracked_paths()))
    if tracked:
        sys.stderr.write("Committed weight files are not allowed:\n")
        for path in tracked:
            sys.stderr.write("  {0}\n".format(path))
        return 1
    sys.stdout.write("No committed weight checkpoints.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

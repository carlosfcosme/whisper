#!/usr/bin/env python3
"""Fail if git-tracked files look like committed model weights.

Standalone (no torch / whisper import) so CI can run it before the
package matrix. Hugging Face Hub artifacts and Whisper checkpoints
are both treated as weights.
"""

from __future__ import print_function

import os
import subprocess
import sys

WEIGHT_SUFFIXES = (
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".ckpt",
    ".onnx",
    ".gguf",
    ".ggml",
    ".h5",
    ".tflite",
    ".msgpack",
)

ALLOWED_PREFIXES = ("whisper/assets/",)

WEIGHT_DIR_MARKERS = (
    ".cache/whisper/",
    "huggingface/",
    "hf_hub/",
    "checkpoints/",
)


def _posix(path):
    return path.replace("\\", "/").lstrip("./")


def is_committed_weight(path):
    """Return True if a tracked path must not be committed."""
    posix = _posix(path)
    if not posix or posix.startswith(".git/"):
        return False
    for prefix in ALLOWED_PREFIXES:
        if posix.startswith(prefix):
            return False
    lower = posix.lower()
    for marker in WEIGHT_DIR_MARKERS:
        if marker in lower:
            return True
    name = posix.rsplit("/", 1)[-1].lower()
    for suffix in WEIGHT_SUFFIXES:
        if name.endswith(suffix):
            return True
    return False


def list_tracked_files(cwd=None):
    cmd = ["git", "ls-files", "-z"]
    try:
        raw = subprocess.check_output(cmd, cwd=cwd)
    except (OSError, subprocess.CalledProcessError):
        return []
    if not raw:
        return []
    return [p.decode("utf-8") for p in raw.split(b"\0") if p]


def find_committed_weights(paths=None, cwd=None):
    if paths is None:
        paths = list_tracked_files(cwd=cwd)
    return [p for p in paths if is_committed_weight(p)]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cwd = os.environ.get("CHECK_NO_WEIGHTS_CWD") or os.getcwd()
    extra = argv
    tracked = list_tracked_files(cwd=cwd)
    paths = list(tracked) + extra
    bad = find_committed_weights(paths=paths, cwd=cwd)
    if bad:
        print("Committed weight files are not allowed:", file=sys.stderr)
        for path in bad:
            print("  {}".format(path), file=sys.stderr)
        return 1
    print("OK: no committed weight files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

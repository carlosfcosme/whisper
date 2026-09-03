"""Fail if model weight files are committed.

This module is stdlib-only so CI can run it as
``python whisper/committed_weights.py`` without installing the package.
"""

import argparse
import os
import subprocess
import sys

WEIGHT_SUFFIXES = (
    ".pt",
    ".pth",
    ".bin",
    ".onnx",
    ".safetensors",
    ".ckpt",
    ".tflite",
    ".gguf",
    ".h5",
    ".hdf5",
    ".msgpack",
    ".ot",
)


def is_weight_path(path):
    """Return True if *path* looks like a committed model checkpoint."""
    name = os.path.basename(path.replace("\\", "/")).lower()
    _, ext = os.path.splitext(name)
    return ext in WEIGHT_SUFFIXES


def find_weight_files(paths):
    """Return tracked paths that look like model weights."""
    hits = []
    for path in paths:
        if not path:
            continue
        normalized = path.replace("\\", "/")
        if is_weight_path(normalized):
            hits.append(normalized)
    return hits


def repo_root(start=None):
    """Locate the git repository root."""
    if start is None:
        start = os.path.dirname(os.path.abspath(__file__))
    current = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.getcwd()
        current = parent


def git_tracked_files(root=None):
    """List files tracked by git in *root*."""
    root = repo_root() if root is None else root
    output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=root,
    )
    return [part.decode("utf-8") for part in output.split(b"\0") if part]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fail if model weight files are committed to git."
    )
    parser.add_argument(
        "--root",
        default=None,
        help="repository root (default: discover from this file or cwd)",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="optional explicit paths; default is git ls-files",
    )
    args = parser.parse_args(argv)

    root = repo_root() if args.root is None else args.root
    files = args.paths if args.paths else git_tracked_files(root)
    hits = find_weight_files(files)
    if hits:
        sys.stderr.write("Committed model weight files are not allowed:\n")
        for path in hits:
            sys.stderr.write("  {}\n".format(path))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

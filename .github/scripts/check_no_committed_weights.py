#!/usr/bin/env python3
"""Fail if model weight files are committed.

Standalone: no torch, no package import, no Hub clients.
"""

from __future__ import print_function

import subprocess
import sys

WEIGHT_PATHSPECS = (
    "*.pt",
    "*.pth",
    "*.safetensors",
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


def committed_weight_paths(repo="."):
    listed = subprocess.check_output(
        ["git", "ls-files", "-z", "--"] + list(WEIGHT_PATHSPECS),
        cwd=repo,
    )
    return [path.decode("utf-8") for path in listed.split(b"\0") if path]


def main():
    paths = committed_weight_paths()
    if paths:
        print("Committed model weight files are forbidden:", file=sys.stderr)
        for path in paths:
            print("  {0}".format(path), file=sys.stderr)
        return 1
    print("OK: no committed model weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())

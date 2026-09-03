#!/usr/bin/env python3
"""Fail if git tracks weight blobs, cache dumps, or coverage reports.

Uses ``git ls-files`` pathspecs (the same globs CI runs). Stdlib only — no
torch, no Hub, no network, no keys.
"""

import subprocess
import sys

WEIGHT_PATHSPECS = (
    "*.pt",
    "*.pth",
    "*.bin",
    "*.onnx",
    "*.safetensors",
    "*.ckpt",
    "*.h5",
    "*.tflite",
    "*.gguf",
    "*.ggml",
)

CACHE_PATHSPECS = (
    ".cache",
    ".cache/**",
    "cache",
    "cache/**",
    "weights",
    "weights/**",
    "checkpoints",
    "checkpoints/**",
    ".huggingface",
    ".huggingface/**",
    "huggingface/hub",
    "huggingface/hub/**",
)

COVERAGE_PATHSPECS = (
    ".coverage",
    ".coverage.*",
    "htmlcov",
    "htmlcov/**",
    "coverage.xml",
    "coverage",
    "coverage/**",
)

PATHSPECS = WEIGHT_PATHSPECS + CACHE_PATHSPECS + COVERAGE_PATHSPECS

# Patterns that .gitignore must contain (ignore-coverage test).
GITIGNORE_PATTERNS = (
    "*.pt",
    "*.pth",
    "*.bin",
    "*.onnx",
    "*.safetensors",
    "*.ckpt",
    "*.gguf",
    "*.ggml",
    ".cache/",
    "cache/",
    "weights/",
    "checkpoints/",
    ".huggingface/",
    "huggingface/hub/",
    ".coverage",
    "htmlcov/",
    "coverage.xml",
)


def git_ls_files(pathspecs, cwd=None):
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--"] + list(pathspecs),
        cwd=cwd,
    )
    if not output:
        return []
    return [part.decode("utf-8") for part in output.split(b"\0") if part]


def main(argv=None):
    del argv
    tracked = git_ls_files(PATHSPECS)
    if tracked:
        sys.stderr.write(
            "git ls-files listed tracked weight/cache/coverage artifacts:\n"
        )
        for path in tracked:
            sys.stderr.write("  {0}\n".format(path))
        return 1
    sys.stdout.write("git ls-files: no tracked weight/cache/coverage artifacts.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

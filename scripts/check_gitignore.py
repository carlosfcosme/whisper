#!/usr/bin/env python3
"""Fail unless .gitignore covers weight/cache paths. Used by CI."""

import subprocess
import sys

MUST_IGNORE = (
    "tiny.pt",
    "model.pth",
    "weights.onnx",
    "model.safetensors",
    "run.ckpt",
    "ggml-tiny.gguf",
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
)

MUST_NOT_IGNORE = (
    "whisper/__init__.py",
    "whisper/assets/mel_filters.npz",
    "tests/jfk.flac",
    "README.md",
    ".github/workflows/test.yml",
)


def check_ignore(path, repo_root="."):
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "--", path],
        cwd=repo_root,
    )
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    raise RuntimeError(f"git check-ignore failed for {path!r} (exit {proc.returncode})")


def verify(repo_root="."):
    missing = [path for path in MUST_IGNORE if not check_ignore(path, repo_root)]
    extra = [path for path in MUST_NOT_IGNORE if check_ignore(path, repo_root)]
    return missing, extra


def main(argv=None):
    missing, extra = verify()
    if missing:
        sys.stderr.write("gitignore does not cover weight/cache paths:\n")
        for path in missing:
            sys.stderr.write(f"  {path}\n")
    if extra:
        sys.stderr.write("gitignore incorrectly ignores source/fixture paths:\n")
        for path in extra:
            sys.stderr.write(f"  {path}\n")
    if missing or extra:
        return 1
    sys.stdout.write("ok: gitignore covers weight/cache paths\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

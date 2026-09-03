#!/usr/bin/env python3
"""Fail CI if cache/weight/secret paths are tracked or not gitignored."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_PATTERNS = (
    ".cache/",
    "cache/",
    "weights/",
    ".huggingface/",
    "hub/",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.bin",
    ".env",
    ".env.*",
)
IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    ".huggingface/hub/tiny.pt",
    "hub/tiny.safetensors",
    "tiny.pt",
    "model.pth",
    "model.safetensors",
    "model.bin",
    ".env",
    ".env.local",
)
LS_SPECS = (
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
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.bin",
    ".env",
    ".env.*",
)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    lines = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    missing = [pattern for pattern in REQUIRED_PATTERNS if pattern not in lines]
    if missing:
        print("patterns missing from .gitignore: %s" % missing, file=sys.stderr)
        return 1

    listed = _git("ls-files", "-z", "--", *LS_SPECS)
    if listed.returncode != 0:
        print(listed.stderr, file=sys.stderr)
        return 1
    tracked = [path for path in listed.stdout.split("\0") if path]
    if tracked:
        print("cache/weight/secret paths must stay untracked:", file=sys.stderr)
        print("\n".join(tracked), file=sys.stderr)
        return 1

    failed = []
    for path in IGNORE_EXAMPLES:
        result = _git("check-ignore", "-q", "--", path)
        if result.returncode != 0:
            failed.append(path)
    if failed:
        print("expected these paths to be gitignored: %s" % failed, file=sys.stderr)
        return 1

    print("ok: gitignore covers cache/weights/secrets; none tracked")
    return 0


if __name__ == "__main__":
    sys.exit(main())

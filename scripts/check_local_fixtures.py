#!/usr/bin/env python3
"""Fail CI if tests rely on anything other than preexisting local fixtures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

# Tracked assets tests may read. No checkpoints. No Hub snapshots.
LOCAL_FIXTURES = (
    "tests/jfk.flac",
    "whisper/assets/multilingual.tiktoken",
    "whisper/assets/gpt2.tiktoken",
    "whisper/normalizers/english.json",
)

# Test modules that load on-disk fixtures (not generated tmp_path toys).
FIXTURE_TEST_MODULES = (
    "tests/test_audio.py",
    "tests/test_transcribe.py",
    "tests/test_tokenizer.py",
    "tests/test_timing.py",
    "tests/test_normalizer.py",
)

WEIGHT_SUFFIXES = (
    ".pt",
    ".pth",
    ".bin",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".gguf",
    ".ggml",
)

REMOTE_MARKERS = (
    "huggingface.co",
    "hf.co",
    "cdn-lfs.huggingface.co",
    "from_pretrained",
    "hf_hub",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tracked_files(root: Path) -> List[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def missing_fixtures(root: Path, fixtures: Sequence[str] = LOCAL_FIXTURES) -> List[str]:
    tracked = set(tracked_files(root))
    missing = []
    for relpath in fixtures:
        path = root / relpath
        if not path.is_file():
            missing.append("{} (missing on disk)".format(relpath))
        elif relpath not in tracked:
            missing.append("{} (not tracked)".format(relpath))
        elif path.suffix.lower() in WEIGHT_SUFFIXES:
            missing.append("{} (weight suffix)".format(relpath))
    return missing


def remote_fixture_hits(root: Path) -> List[str]:
    hits = []
    for relpath in FIXTURE_TEST_MODULES:
        text = (root / relpath).read_text(encoding="utf-8")
        for marker in REMOTE_MARKERS:
            if marker in text:
                hits.append("{}: {}".format(relpath, marker))
    return hits


def main() -> int:
    root = repo_root()
    missing = missing_fixtures(root)
    remote = remote_fixture_hits(root)
    if missing or remote:
        if missing:
            sys.stderr.write("ERROR: preexisting local fixtures are incomplete:\n")
            for item in missing:
                sys.stderr.write("  {}\n".format(item))
        if remote:
            sys.stderr.write("ERROR: fixture tests must not fetch Hub/remote assets:\n")
            for item in remote:
                sys.stderr.write("  {}\n".format(item))
        return 1
    sys.stdout.write("OK: tests use preexisting local fixtures only\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

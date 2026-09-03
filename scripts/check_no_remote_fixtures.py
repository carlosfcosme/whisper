#!/usr/bin/env python3
"""Fail CI if tests reference remote URLs for audio/model fixtures."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
FIXTURES_DIR = TESTS_DIR / "fixtures"

# Quoted http(s) URLs that look like audio or checkpoint assets.
_REMOTE_ASSET = re.compile(
    r"""['\"]https?://[^'\"]+\.(?:wav|flac|mp3|ogg|m4a|pt|pth|bin|safetensors|onnx|ckpt)['\"]""",
    re.IGNORECASE,
)
_HUB_ASSET = re.compile(
    r"""['\"]https?://(?:huggingface\.co|hf\.co)[^'\"]*['\"]""",
    re.IGNORECASE,
)
_LOAD_REMOTE = re.compile(
    r"""(?:load_audio|load_model|fixture_path|local_path|repo_audio_path)\s*\(\s*['\"]https?://"""
)
_ASSIGN_REMOTE = re.compile(
    r"""(?:audio|sample|fixture|model|checkpoint|wav|flac)_path\s*=\s*['\"]https?://"""
)

_SCAN_SUFFIXES = frozenset({".py", ".yml", ".yaml", ".ini", ".toml", ".json", ".cfg"})


def _iter_test_files() -> List[Path]:
    files: List[Path] = []
    if not TESTS_DIR.is_dir():
        return files
    for path in TESTS_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in _SCAN_SUFFIXES:
            files.append(path)
    return sorted(files)


def classify_line(relpath: str, line: str) -> List[str]:
    reasons: List[str] = []
    if _LOAD_REMOTE.search(line):
        reasons.append("load_* / fixture helper called with a remote URL")
    if _ASSIGN_REMOTE.search(line):
        reasons.append("fixture path assigned a remote URL")
    if _REMOTE_ASSET.search(line):
        reasons.append("remote audio/checkpoint URL")
    if _HUB_ASSET.search(line):
        reasons.append("Hugging Face Hub URL used as an asset")
    if relpath.startswith("tests/fixtures/") and _HUB_ASSET.search(line):
        reasons.append("Hub URL in tests/fixtures/")
    return reasons


def find_violations(root: Path = REPO_ROOT) -> List[Tuple[str, int, str, str]]:
    tests_dir = root / "tests"
    hits: List[Tuple[str, int, str, str]] = []
    if not tests_dir.is_dir():
        return hits
    for path in sorted(tests_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SCAN_SUFFIXES:
            continue
        relpath = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for reason in classify_line(relpath, line):
                hits.append((relpath, lineno, reason, line.strip()))
    return hits


def main() -> int:
    violations = find_violations(REPO_ROOT)
    if violations:
        sys.stderr.write(
            "ERROR: tests must use in-repo or tempfile fixture paths; "
            "remote asset URLs are not allowed:\n"
        )
        for relpath, lineno, reason, line in violations:
            sys.stderr.write("  {}:{}: {} | {}\n".format(relpath, lineno, reason, line))
        return 1
    sys.stdout.write("OK: no remote URLs for test audio/model fixtures\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Fail if weight/cache artifacts are not gitignored or are tracked."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
GITIGNORE = REPO_ROOT / ".gitignore"

REQUIRED_PATTERNS = (
    ".cache/",
    "cache/",
    "weights/",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.safetensors",
)

PROBE_PATHS = (
    ".cache/whisper/tiny.pt",
    "cache/tiny.pt",
    "weights/tiny.pt",
    "tiny.pt",
    "model.pth",
    "model.ckpt",
    "model.safetensors",
)


def reasons_gitignore_missing(text: str | None = None) -> List[str]:
    if text is None:
        text = GITIGNORE.read_text(encoding="utf-8")
    lines = {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    reasons: List[str] = []
    for pattern in REQUIRED_PATTERNS:
        if pattern not in lines:
            reasons.append(f"missing gitignore pattern: {pattern}")
    return reasons


def reasons_probes_not_ignored() -> List[str]:
    reasons: List[str] = []
    for probe in PROBE_PATHS:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", probe],
            cwd=REPO_ROOT,
            check=False,
        )
        if result.returncode != 0:
            reasons.append(f"git check-ignore does not match {probe}")
    return reasons


def reasons_tracked_weight_paths() -> List[str]:
    listing = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
    )
    reasons: List[str] = []
    for raw in listing.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", "surrogateescape")
        lower = rel.lower()
        if lower.startswith(".cache/") or lower.startswith("cache/"):
            reasons.append(f"tracked cache path: {rel}")
        if lower.startswith("weights/"):
            reasons.append(f"tracked weights path: {rel}")
        if any(
            lower.endswith(ext)
            for ext in (".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".gguf")
        ):
            reasons.append(f"tracked weight file: {rel}")
    return reasons


def all_reasons() -> List[str]:
    return (
        reasons_gitignore_missing()
        + reasons_probes_not_ignored()
        + reasons_tracked_weight_paths()
    )


def main() -> int:
    reasons = all_reasons()
    if reasons:
        print("FAIL: weight/cache artifacts are not fully gitignored:")
        for reason in reasons:
            print(f"  - {reason}")
        return 1
    print("OK: weight/cache artifacts are gitignored and untracked")
    return 0


if __name__ == "__main__":
    sys.exit(main())

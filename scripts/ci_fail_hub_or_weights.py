#!/usr/bin/env python3
"""CI guard: fail if tests touch HuggingFace Hub or weights are committed.

Standalone (no whisper import) so it can run before the package is installed.
No model-weight download. No Hub access.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

WEIGHT_SUFFIXES = (
    ".pt",
    ".pth",
    ".safetensors",
    ".onnx",
    ".gguf",
    ".ckpt",
    ".h5",
)

HUB_NEEDLES = (
    "huggingface.co",
    "huggingface_hub",
    "from_pretrained",
    "hf_hub",
    "hf.co",
)


def _git_tracked_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
    )
    return [p for p in proc.stdout.decode().split("\0") if p]


def committed_weight_files() -> list[str]:
    hits = []
    for rel in _git_tracked_files():
        path = Path(rel)
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            hits.append(rel)
    return hits


def hub_references_in_tests() -> list[str]:
    hits = []
    tests_root = REPO_ROOT / "tests"
    for path in sorted(tests_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".py", ".sh", ".yml", ".yaml", ".json", ".txt"}:
            continue
        text = path.read_text(errors="replace")
        lowered = text.lower()
        for needle in HUB_NEEDLES:
            if needle.lower() in lowered:
                rel = path.relative_to(REPO_ROOT).as_posix()
                hits.append(f"{rel}: {needle}")
                break
    return hits


def main() -> int:
    weights = committed_weight_files()
    hub = hub_references_in_tests()
    failed = False
    if weights:
        failed = True
        print("FAIL: committed model weight files:")
        for rel in weights:
            print(f"  {rel}")
    if hub:
        failed = True
        print("FAIL: HuggingFace Hub access in tests:")
        for rel in hub:
            print(f"  {rel}")
    if failed:
        return 1
    print("OK: no committed weights; tests do not reference HuggingFace Hub")
    return 0


if __name__ == "__main__":
    sys.exit(main())

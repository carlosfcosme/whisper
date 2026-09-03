#!/usr/bin/env python3
"""CI guards: no 0.0.0.0 binds, no Hub client in tests, no committed weights."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAIL = []

HUB_IMPORTS = (
    "import huggingface_hub",
    "from huggingface_hub",
    "hf_hub_download(",
    "snapshot_download(",
)

WEIGHT_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".onnx", ".ckpt"}


def _fail(message: str) -> None:
    FAIL.append(message)


def check_start_scripts() -> None:
    candidates = [
        ROOT / ".cursor" / "start.sh",
        ROOT / ".cursor" / "environment.json",
        ROOT / "start.sh",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text()
        if "0.0.0.0" in text:
            _fail(f"{path.relative_to(ROOT)} contains 0.0.0.0")


def check_tests_no_hub_client() -> None:
    tests = ROOT / "tests"
    for path in tests.rglob("*.py"):
        if path.name == "test_cpu_default.py":
            # Mentions Hub URLs only to assert they are refused.
            continue
        text = path.read_text()
        for needle in HUB_IMPORTS:
            if needle in text:
                _fail(f"{path.relative_to(ROOT)} uses Hub client {needle!r}")


def check_no_tracked_weights() -> None:
    if not (ROOT / ".git").is_dir():
        return
    listed = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    for rel in listed:
        suffix = Path(rel).suffix.lower()
        if suffix in WEIGHT_SUFFIXES:
            _fail(f"tracked weight file: {rel}")
        parts = Path(rel).parts
        if ".huggingface" in parts:
            _fail(f"tracked Hub cache path: {rel}")


def main() -> int:
    check_start_scripts()
    check_tests_no_hub_client()
    check_no_tracked_weights()
    if FAIL:
        print("CI guards failed:")
        for item in FAIL:
            print(f"  - {item}")
        return 1
    print("CI guards passed: 127.0.0.1 bind, no Hub client in tests, no weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())

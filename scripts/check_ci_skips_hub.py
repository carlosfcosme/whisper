#!/usr/bin/env python3
"""Fail if CI can reach the Hugging Face Hub."""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "test.yml"
REQUIRED_ENV = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)
FORBIDDEN = (
    "huggingface.co",
    "huggingface_hub",
    "hf_hub_download",
    "hf.co/",
)


def main() -> int:
    text = WORKFLOW.read_text(encoding="utf-8")
    errors = []
    for name in REQUIRED_ENV:
        if not re.search(rf"{name}:\s*[\"']?1[\"']?", text):
            errors.append(f"whisper-test must set {name}=1")
    for token in FORBIDDEN:
        if token in text:
            errors.append(f"workflow must not reference {token}")
    if "whisper-test:" not in text:
        errors.append("whisper-test job is missing")
    if errors:
        print("CI must skip the Hugging Face Hub:", file=sys.stderr)
        for line in errors:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

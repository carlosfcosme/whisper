#!/usr/bin/env python3
"""Fail if CI can reach the Hugging Face Hub or pull named-model weights."""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "test.yml"
REQUIRED_ENV = (
    "WHISPER_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)
FORBIDDEN = (
    "huggingface.co",
    "huggingface_hub",
    "hf_hub_download",
    "snapshot_download",
    "from_pretrained",
    "hf.co/",
    "openaipublic.azureedge.net",
)


def main() -> int:
    text = WORKFLOW.read_text(encoding="utf-8")
    errors = []
    if "whisper-test:" not in text:
        errors.append("whisper-test job is missing")
    for name in REQUIRED_ENV:
        if not re.search(r"{}:\s*[\"']?1[\"']?".format(name), text):
            errors.append("whisper-test must set {}=1".format(name))
    for token in FORBIDDEN:
        if token in text:
            errors.append("workflow must not reference {}".format(token))
    if "-k 'not test_transcribe'" not in text:
        errors.append("pytest must skip test_transcribe (no named-model download)")
    if "scripts/assert_no_weight_cache.py" not in text:
        errors.append("whisper-test must run scripts/assert_no_weight_cache.py")
    if "tests/test_no_weight_fetch.py" not in text:
        errors.append("whisper-test must run tests/test_no_weight_fetch.py")
    if errors:
        sys.stderr.write("CI must skip the Hugging Face Hub and weight downloads:\n")
        for line in errors:
            sys.stderr.write("  {}\n".format(line))
        return 1
    sys.stdout.write("OK: CI skips Hub and named-model weight downloads\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

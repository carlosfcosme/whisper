#!/usr/bin/env python3
"""Fail CI if the test workflow would download Hub or model weights."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"


def main() -> int:
    text = WORKFLOW.read_text(encoding="utf-8")
    errors = []
    if "HF_HUB_OFFLINE" not in text:
        errors.append("whisper-test must set HF_HUB_OFFLINE")
    if "WHISPER_OFFLINE" not in text:
        errors.append("whisper-test must set WHISPER_OFFLINE")
    if "test_transcribe[tiny]" in text or "test_transcribe[tiny.en]" in text:
        errors.append("CI must not select test_transcribe[tiny] (weight pull)")
    if (
        "-k 'not test_transcribe'" not in text
        and '-k "not test_transcribe"' not in text
    ):
        errors.append("CI pytest must exclude test_transcribe")
    if errors:
        sys.stderr.write("error: CI would download Hub/weights:\n")
        for item in errors:
            sys.stderr.write("  {0}\n".format(item))
        return 1
    sys.stdout.write("OK: CI does not pull Hub or test weights\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

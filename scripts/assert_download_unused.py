#!/usr/bin/env python3
"""CI helper: default path must not Hub or download weights.

Modes:
  --check-workflow  static scan of .github/workflows/test.yml (no torch)
  --probe           import whisper, refuse named load, assert download unused
  --cache DIR       fail if DIR contains .pt / .pth / Hub cache files
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"
TOKEN_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_workflow() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    if "test_transcribe[tiny]" in text or "test_transcribe[tiny.en]" in text:
        _fail("CI must not select test_transcribe[tiny] (that path pulls weights)")
    if "WHISPER_OFFLINE" not in text:
        _fail("CI must set WHISPER_OFFLINE")
    if "HF_HUB_OFFLINE" not in text:
        _fail("CI must set HF_HUB_OFFLINE")
    for name in TOKEN_NAMES:
        if name in text:
            _fail(f"CI must not use Hub token env {name}")
    if "huggingface.co" in text.lower() and "HF_HUB_OFFLINE" not in text:
        _fail("CI mentions Hub without HF_HUB_OFFLINE")
    print("OK: workflow is offline (no tiny weight pull, no Hub token)")


def probe() -> None:
    os.environ.setdefault("WHISPER_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import tempfile

    import whisper
    from whisper.offline import (
        OfflineError,
        assert_download_unused,
        reset_download_usage,
    )

    reset_download_usage()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            whisper.load_model("tiny", download_root=tmp)
        except OfflineError:
            pass
        else:
            _fail("load_model('tiny') must refuse on the offline path")
        leftover = list(Path(tmp).rglob("*.pt"))
        if leftover:
            _fail(f"probe wrote checkpoints: {leftover}")
    assert_download_unused("probe")
    print("OK: download unused (named load refused, no urlopen)")


def check_cache(cache_dir: str) -> None:
    root = Path(cache_dir)
    if not root.exists():
        print("OK: cache dir absent")
        return
    bad = []
    for pattern in ("*.pt", "*.pth", "*.safetensors", "*.bin"):
        bad.extend(root.rglob(pattern))
    for name in ("huggingface", "hub"):
        hit = root / name
        if hit.exists():
            bad.append(hit)
    if bad:
        _fail(f"weight/Hub cache present: {bad}")
    print("OK: cache has no weight or Hub files")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-workflow", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--cache", metavar="DIR")
    args = parser.parse_args(argv)
    if not (args.check_workflow or args.probe or args.cache):
        parser.error("specify --check-workflow, --probe, and/or --cache")
    if args.check_workflow:
        check_workflow()
    if args.probe:
        probe()
    if args.cache:
        check_cache(args.cache)


if __name__ == "__main__":
    main()

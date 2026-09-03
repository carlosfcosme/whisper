#!/usr/bin/env python3
"""CI guard: fail if bind is not 127.0.0.1, Hub is used, or weights are pulled."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"
WILDCARD = "0.0.0.0"
TOKEN_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")
FORBIDDEN = (
    "docker-compose",
    "compose.yaml",
    "compose.yml",
    "spark.yaml",
    "spark.yml",
    "--live true",
    "--live=true",
    "Field-Brain",
    "fieldbrain",
    "bpf",
    "kprobe",
)


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_bind() -> None:
    start = ROOT / ".cursor" / "start.sh"
    if not start.is_file():
        _fail("missing .cursor/start.sh")
    text = start.read_text(encoding="utf-8")
    if "--host 127.0.0.1" not in text:
        _fail("start.sh must bind --host 127.0.0.1")
    if WILDCARD in text:
        _fail("start.sh must not contain 0.0.0.0")
    for path in list(ROOT.glob(".cursor/*.sh")) + [
        ROOT / ".cursor" / "environment.json"
    ]:
        if path.is_file() and WILDCARD in path.read_text(encoding="utf-8"):
            _fail(f"{path.relative_to(ROOT)} contains {WILDCARD}")
    print("OK: bind 127.0.0.1")


def check_workflow() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    if "test_transcribe[tiny]" in text or "test_transcribe[tiny.en]" in text:
        _fail("CI must not select test_transcribe[tiny] (that pulls weights)")
    if "WHISPER_OFFLINE" not in text or "HF_HUB_OFFLINE" not in text:
        _fail("CI must set WHISPER_OFFLINE and HF_HUB_OFFLINE")
    for name in TOKEN_NAMES:
        if name in text:
            _fail(f"CI must not use Hub token env {name}")
    for token in FORBIDDEN:
        if token.lower() in text.lower() and token != "bpf":
            _fail(f"forbidden token in workflow: {token}")
    print("OK: workflow offline / no Hub tokens")


def check_tree() -> None:
    listed = subprocess.check_output(["git", "ls-files", "-z"], cwd=str(ROOT)).split(
        b"\0"
    )
    tracked = []
    for raw in listed:
        if not raw:
            continue
        name = raw.decode("utf-8", "replace")
        lower = name.lower()
        if any(lower.endswith(s) for s in (".pt", ".pth", ".safetensors")):
            if not name.startswith("whisper/assets/"):
                tracked.append(name)
        if any(
            part in lower
            for part in (
                "docker-compose",
                "compose.yaml",
                "spark.yaml",
                "spark.yml",
                "field-brain",
            )
        ):
            tracked.append(name)
    if tracked:
        _fail(f"forbidden tracked paths: {tracked}")
    device = (ROOT / "whisper" / "device.py").read_text(encoding="utf-8")
    if 'DEFAULT_DEVICE = "cpu"' not in device:
        _fail("DEFAULT_DEVICE must be cpu")
    print("OK: no tracked weights / no compose / no Spark / CPU default")


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
    print("OK: download unused")


def check_cache(cache_dir: str) -> None:
    root = Path(cache_dir)
    if not root.exists():
        print("OK: cache dir absent")
        return
    bad = []
    for pattern in ("*.pt", "*.pth", "*.safetensors"):
        bad.extend(root.rglob(pattern))
    if bad:
        _fail(f"weight files in cache: {bad}")
    print("OK: cache has no weights")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--bind", action="store_true")
    parser.add_argument("--workflow", action="store_true")
    parser.add_argument("--tree", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--cache", metavar="DIR")
    args = parser.parse_args(argv)
    if args.all:
        args.bind = args.workflow = args.tree = True
    if not any([args.bind, args.workflow, args.tree, args.probe, args.cache]):
        parser.error("specify a check")
    if args.bind:
        check_bind()
    if args.workflow:
        check_workflow()
    if args.tree:
        check_tree()
    if args.probe:
        probe()
    if args.cache:
        check_cache(args.cache)


if __name__ == "__main__":
    main()

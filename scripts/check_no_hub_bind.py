#!/usr/bin/env python3
"""Static CI guard: bind 127.0.0.1, no Hub dependency, no tracked weights."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WILDCARD = "0.0.0.0"
HUB_TOKENS = ("huggingface_hub", "huggingface.co", "hf.co")
WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors", ".onnx", ".bin")


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def _scan_start_scripts() -> None:
    paths = list(ROOT.glob(".cursor/*.sh"))
    paths.extend(ROOT.glob("start*.sh"))
    paths.append(ROOT / ".cursor" / "environment.json")
    hits = []
    for path in paths:
        if not path.is_file():
            continue
        if WILDCARD in path.read_text(encoding="utf-8"):
            hits.append(str(path.relative_to(ROOT)))
    if hits:
        _fail(f"{WILDCARD} must not appear in start scripts: {hits}")


def _scan_pyproject() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    for token in HUB_TOKENS:
        if token in text:
            _fail(f"pyproject.toml must not depend on Hub ({token})")


def _scan_tracked_weights() -> None:
    listed = subprocess.check_output(["git", "ls-files", "-z"], cwd=str(ROOT)).split(
        b"\0"
    )
    tracked = []
    for raw in listed:
        if not raw:
            continue
        name = raw.decode("utf-8", "replace")
        lower = name.lower()
        if any(lower.endswith(suffix) for suffix in WEIGHT_SUFFIXES):
            if name.startswith("whisper/assets/"):
                continue
            tracked.append(name)
        if lower.startswith((".cache/", "cache/", "weights/", ".huggingface/")):
            tracked.append(name)
    if tracked:
        _fail(f"tracked weight/cache paths: {tracked}")


def _scan_device_default() -> None:
    device = (ROOT / "whisper" / "device.py").read_text(encoding="utf-8")
    if 'DEFAULT_DEVICE = "cpu"' not in device:
        _fail("whisper.device.DEFAULT_DEVICE must be cpu")
    init = (ROOT / "whisper" / "__init__.py").read_text(encoding="utf-8")
    if "cuda if torch.cuda.is_available()" in init:
        _fail("load_model must not auto-select CUDA")
    transcribe = (ROOT / "whisper" / "transcribe.py").read_text(encoding="utf-8")
    if 'default="cuda" if torch.cuda.is_available()' in transcribe:
        _fail("CLI --device must not auto-select CUDA")


def main() -> None:
    _scan_start_scripts()
    _scan_pyproject()
    _scan_tracked_weights()
    _scan_device_default()
    print("OK: bind 127.0.0.1, no Hub, CPU default, no tracked weights")


if __name__ == "__main__":
    main()

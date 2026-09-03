#!/usr/bin/env python3
"""Fail CI if the default load path can pull weights or hit the Hub."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_offline():
    path = repo_root() / "whisper" / "offline.py"
    spec = importlib.util.spec_from_file_location("whisper_offline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _download_calls_refuse_before_urlopen(src: str) -> None:
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_download":
            refuse_line = None
            urlopen_line = None
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Name)
                    and child.id == "refuse_weight_auto_download"
                ):
                    refuse_line = child.lineno
                elif isinstance(child, ast.Attribute) and child.attr == "urlopen":
                    urlopen_line = child.lineno
            if refuse_line is None:
                raise SystemExit("_download must call refuse_weight_auto_download")
            if urlopen_line is None:
                raise SystemExit("_download is missing urlopen (unexpected)")
            if refuse_line > urlopen_line:
                raise SystemExit("_download must refuse auto-download before urlopen")
            return
    raise SystemExit("_download not found in whisper/__init__.py")


def main() -> int:
    root = repo_root()
    offline = load_offline()
    for key in (
        offline.ALLOW_WEIGHT_DOWNLOAD_ENV,
        offline.OFFLINE_ENV,
        offline.NO_WEIGHT_DOWNLOAD_ENV,
    ):
        os.environ.pop(key, None)
    if offline.weight_auto_download_allowed():
        print(
            "weight_auto_download_allowed() is True; default must be offline",
            file=sys.stderr,
        )
        return 1

    init_src = (root / "whisper" / "__init__.py").read_text()
    _download_calls_refuse_before_urlopen(init_src)
    if "huggingface.co" in init_src or "hf.co" in init_src:
        print("whisper/__init__.py must not contain Hub download URLs", file=sys.stderr)
        return 1

    yml = (root / ".github" / "workflows" / "test.yml").read_text()
    for needle in ("HF_HUB_OFFLINE", "WHISPER_OFFLINE", "WHISPER_NO_WEIGHT_DOWNLOAD"):
        if needle not in yml:
            print(f"CI workflow missing offline env var {needle}", file=sys.stderr)
            return 1
    if "test_transcribe[tiny]" in yml:
        print("CI still selects test_transcribe (would pull weights)", file=sys.stderr)
        return 1

    print("Default load path is offline; CI does not pull weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

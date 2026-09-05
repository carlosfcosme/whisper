#!/usr/bin/env python3
"""Fail CI if the default load path can pull weights or CI still fetches them."""

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
        print("default must refuse weight auto-download", file=sys.stderr)
        return 1

    src = (root / "whisper" / "__init__.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_download":
            refuse_line = urlopen_line = None
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Name)
                    and child.id == "refuse_weight_auto_download"
                ):
                    refuse_line = child.lineno
                elif isinstance(child, ast.Attribute) and child.attr == "urlopen":
                    urlopen_line = child.lineno
            if (
                refuse_line is None
                or urlopen_line is None
                or refuse_line > urlopen_line
            ):
                print("_download must refuse before urlopen", file=sys.stderr)
                return 1
            break
    else:
        print("_download not found", file=sys.stderr)
        return 1

    yml = (root / ".github" / "workflows" / "test.yml").read_text()
    for needle in ("WHISPER_OFFLINE", "WHISPER_NO_WEIGHT_DOWNLOAD", "HF_HUB_OFFLINE"):
        if needle not in yml:
            print(f"CI missing {needle}", file=sys.stderr)
            return 1
    if "test_transcribe[tiny]" in yml:
        print("CI still selects test_transcribe (would pull weights)", file=sys.stderr)
        return 1
    print("Offline no-weight-download CI guard OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

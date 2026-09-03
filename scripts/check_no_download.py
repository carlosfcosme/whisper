#!/usr/bin/env python3
"""Fail CI if weight downloads can still reach the network.

Loads ``whisper.offline`` without torch, asserts every negative network
fixture is refused, and checks that ``whisper/_download`` never calls
``urlopen``.
"""

from __future__ import annotations

import ast
import importlib.util
import tempfile
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]


def _load_offline():
    path = ROOT / "whisper" / "offline.py"
    spec = importlib.util.spec_from_file_location("whisper_offline_ci", path)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _download_calls_urlopen(source: str) -> bool:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "_download":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr == "urlopen":
                return True
            if isinstance(child, ast.Name) and child.id == "urlopen":
                return True
        return False
    raise SystemExit("_download() is missing from whisper/__init__.py")


def find_urlopen_in_download() -> List[str]:
    source = (ROOT / "whisper" / "__init__.py").read_text(encoding="utf-8")
    if _download_calls_urlopen(source):
        return ["whisper/__init__.py:_download"]
    return []


def runtime_checks() -> None:
    offline = _load_offline()
    urls = list(offline.iter_forbidden_network_urls())
    if len(urls) < 3:
        raise SystemExit("offline module must expose negative network fixtures")

    for url in urls:
        try:
            offline.refuse_weight_fetch(url)
        except offline.WeightDownloadError:
            continue
        raise SystemExit("refuse_weight_fetch accepted %r" % url)

    loopback = "http://127.0.0.1:9/health"
    try:
        offline.refuse_weight_fetch(loopback)
    except offline.WeightDownloadError:
        raise SystemExit("loopback health URL must not be classified as a weight fetch")

    with tempfile.TemporaryDirectory() as tmp:
        planted = Path(tmp) / "tiny.pt"
        planted.write_bytes(b"not-a-checkpoint")
        # Local paths are not remote fetches.
        try:
            offline.refuse_weight_fetch(str(planted))
        except offline.WeightDownloadError:
            raise SystemExit("local path must not be refused as a network fetch")


def main() -> int:
    hits = find_urlopen_in_download()
    if hits:
        print("weight download still calls urlopen:")
        for hit in hits:
            print("  %s" % hit)
        return 1
    runtime_checks()
    print("ok: no-download runtime, negative network fixtures refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

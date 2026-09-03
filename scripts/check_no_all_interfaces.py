#!/usr/bin/env python3
"""Fail if production paths bind all interfaces, or if that bind is accepted."""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("whisper", "scripts", ".cursor")
FORBIDDEN = "0.0.0.0"
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache"}
# This checker names the forbidden bind; do not scan this file.
SELF = Path(__file__).resolve()


def _iter_files(root: Path):
    if root.is_file():
        yield root
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _scan_production_paths():
    hits = []
    for root_name in SCAN_ROOTS:
        root = REPO / root_name
        if not root.exists():
            continue
        for path in _iter_files(root):
            if path.resolve() == SELF:
                continue
            if SKIP_PARTS.intersection(path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if FORBIDDEN in text:
                hits.append(str(path.relative_to(REPO)))
    return hits


def _load_serve():
    path = REPO / "whisper" / "serve.py"
    spec = importlib.util.spec_from_file_location("whisper_serve_ci", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bind_is_refused() -> None:
    serve = _load_serve()
    if serve.DEFAULT_HOST != "127.0.0.1":
        raise SystemExit(f"DEFAULT_HOST must be 127.0.0.1, got {serve.DEFAULT_HOST!r}")
    try:
        httpd = serve.create_server(host=FORBIDDEN, port=0)
    except serve.BindError:
        return
    httpd.server_close()
    raise SystemExit("create_server accepted an all-interfaces bind")


def main() -> int:
    hits = _scan_production_paths()
    if hits:
        print(
            "all-interfaces bind is not allowed in production paths:", file=sys.stderr
        )
        for hit in hits:
            print(f"  {hit}", file=sys.stderr)
        return 1
    _bind_is_refused()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail CI if the package downloads checkpoints from Hugging Face Hub."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

HUB_MARKERS = (
    "huggingface.co",
    "huggingface_hub",
    "hf_hub",
    "from_pretrained",
)
SCAN_FILES = (
    Path("pyproject.toml"),
    Path("requirements.txt"),
)


def _load_hub(path: Path):
    spec = importlib.util.spec_from_file_location("whisper_hub_check", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package_sources(root: Path) -> list[Path]:
    return sorted((root / "whisper").rglob("*.py"))


def _imports_hub(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "huggingface" in alias.name:
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if "huggingface" in node.module:
                hits.append(node.module)
    return hits


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    hub = _load_hub(root / "whisper" / "hub.py")

    official = "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
    try:
        hub.assert_not_hub_url(official)
    except hub.HubDisabledError:
        print("ERROR: official Azure URL was rejected", file=sys.stderr)
        return 1

    for url in (
        "https://huggingface.co/openai/whisper-tiny",
        "https://hf.co/openai/whisper-tiny",
        "https://cdn-lfs.huggingface.co/repos/whisper/model.safetensors",
    ):
        try:
            hub.assert_not_hub_url(url)
        except hub.HubDisabledError:
            continue
        print(f"ERROR: Hub URL was allowed: {url}", file=sys.stderr)
        return 1

    for path in _package_sources(root):
        if path.name == "hub.py":
            continue
        imports = _imports_hub(path)
        if imports:
            print(
                f"ERROR: Hugging Face import in {path}: {imports}",
                file=sys.stderr,
            )
            return 1
        text = path.read_text(encoding="utf-8")
        for marker in HUB_MARKERS:
            if marker in text:
                print(
                    f"ERROR: {marker!r} appears in {path}",
                    file=sys.stderr,
                )
                return 1

    for rel in SCAN_FILES:
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for marker in ("huggingface_hub", "transformers"):
            if marker in text:
                print(
                    f"ERROR: {marker!r} listed in {path}",
                    file=sys.stderr,
                )
                return 1

    print("OK: package does not use Hugging Face Hub")
    return 0


if __name__ == "__main__":
    sys.exit(main())

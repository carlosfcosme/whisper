#!/usr/bin/env python3
"""Fail if GitHub Actions cache paths include model weights.

Allowed cache targets are pip and pre-commit only. ~/.cache/whisper,
Hugging Face caches, and checkpoint globs must not be cached.
"""

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / ".github" / "workflows"
WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors", ".onnx", ".ckpt")
RUNTIME_DIRS = (
    Path.home() / ".cache" / "whisper",
    Path.home() / ".cache" / "huggingface",
    REPO / "weights",
    REPO / ".cache",
    REPO / "cache",
)

_CACHE_USES = re.compile(r"uses:\s*actions/cache(?:/save|/restore)?@")
_PATH_KEY = re.compile(r"^(\s+)path:\s*(?:\|\s*)?(\S.*)?$")
_KEY_LINE = re.compile(r"^\s+[A-Za-z_][\w-]*:")


def _is_forbidden_cache_path(path):
    lowered = path.strip().lower().replace("\\", "/")
    if not lowered:
        return False
    if lowered.startswith("${{") and "pip-cache" in lowered:
        return False
    if "huggingface" in lowered or "hf_home" in lowered:
        return True
    if re.search(r"(^|/)weights(/|$)", lowered):
        return True
    if any(
        token in lowered
        for token in ("*.pt", "*.pth", "*.safetensors", "*.onnx", "*.ckpt")
    ):
        return True
    if re.search(r"(^|/)(\.cache/)?whisper(/|$)", lowered):
        return True
    if lowered in {"~/.cache", "~/.cache/", "${{ env.home }}/.cache"}:
        return True
    return False


def _cache_paths_from_text(text):
    paths = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not _CACHE_USES.search(lines[i]):
            i += 1
            continue
        j = i + 1
        while j < len(lines):
            match = _PATH_KEY.match(lines[j])
            if match:
                indent, inline = match.group(1), (match.group(2) or "").strip()
                if inline and inline != "|":
                    paths.append(inline)
                    break
                j += 1
                while j < len(lines):
                    line = lines[j]
                    if not line.strip():
                        j += 1
                        continue
                    if not line.startswith(indent + "  "):
                        break
                    if _KEY_LINE.match(line) and not line.strip().startswith("-"):
                        break
                    paths.append(line.strip())
                    j += 1
                break
            if lines[j].lstrip().startswith("- ") and "uses:" in lines[j]:
                break
            j += 1
        i = j if j > i else i + 1
    return paths


def check_workflows():
    offenders = []
    for path in sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        for cache_path in _cache_paths_from_text(text):
            if _is_forbidden_cache_path(cache_path):
                offenders.append("{}: {}".format(path.relative_to(REPO), cache_path))
    if offenders:
        sys.stderr.write("GitHub Actions cache must not include weights:\n")
        for line in offenders:
            sys.stderr.write("  {}\n".format(line))
        return 1
    return 0


def check_runtime():
    found = []
    for root in RUNTIME_DIRS:
        if not root.exists():
            continue
        for item in root.rglob("*"):
            if item.is_file() and item.suffix in WEIGHT_SUFFIXES:
                found.append(str(item))
    if found:
        sys.stderr.write("weight files present in CI cache directories:\n")
        for line in found:
            sys.stderr.write("  {}\n".format(line))
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="scan local cache dirs for checkpoint files",
    )
    args = parser.parse_args(argv)
    code = check_workflows()
    if args.runtime:
        code = code or check_runtime()
    if code == 0:
        sys.stdout.write("OK: CI cache paths do not include model weights\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail if GitHub Actions can hit the Hub or download named checkpoints."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "test.yml"

REQUIRED_ENV = (
    "WHISPER_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)

FORBIDDEN_TOKENS = (
    "huggingface.co",
    "huggingface_hub",
    "hf_hub_download",
    "snapshot_download",
    "huggingface-cli",
    "from_pretrained(",
    "test_transcribe[tiny]",
    "test_transcribe[tiny.en]",
)

_CACHE_USES = re.compile(r"uses:\s*actions/cache(?:/save|/restore)?@")
_PATH_KEY = re.compile(r"^(\s+)path:\s*(?:\|\s*)?(\S.*)?$")
_KEY_LINE = re.compile(r"^\s+[A-Za-z_][\w-]*:")


def _is_forbidden_cache_path(path: str) -> bool:
    lowered = path.strip().lower().replace("\\", "/")
    if not lowered:
        return False
    if "pip-cache" in lowered:
        return False
    if "huggingface" in lowered or "hf_home" in lowered or "hf_hub" in lowered:
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
    if lowered in {"~/.cache", "~/.cache/"}:
        return True
    return False


def _cache_paths_from_text(text: str) -> List[str]:
    paths: List[str] = []
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


def main() -> int:
    text = WORKFLOW.read_text(encoding="utf-8")
    errors: List[str] = []
    if "whisper-test:" not in text:
        errors.append("whisper-test job is missing")
    for name in REQUIRED_ENV:
        if not re.search(rf"{name}:\s*[\"']?1[\"']?", text):
            errors.append("whisper-test must set {}=1".format(name))
    for token in FORBIDDEN_TOKENS:
        if token in text:
            errors.append("workflow must not reference {}".format(token))
    if "not requires_weights" not in text:
        errors.append("pytest must exclude requires_weights")
    if "assert_no_hub_fetch.py" not in text:
        errors.append("whisper-test must run assert_no_hub_fetch.py")
    if "assert_no_weight_cache.py" not in text:
        errors.append("whisper-test must run assert_no_weight_cache.py")
    for cache_path in _cache_paths_from_text(text):
        if _is_forbidden_cache_path(cache_path):
            errors.append("actions/cache must not include {}".format(cache_path))
    if errors:
        sys.stderr.write("ERROR: CI must stay offline and never hit the Hub:\n")
        for line in errors:
            sys.stderr.write("  {}\n".format(line))
        return 1
    sys.stdout.write("OK: CI is offline (no Hub, no named-weight download)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

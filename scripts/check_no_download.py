#!/usr/bin/env python3
"""CI guard: no Hub client and no remote model-weight download.

Loads whisper.runtime by path (stdlib only). No torch, no network.
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Tuple

HUB_IMPORTS = (
    "huggingface_hub",
    "from_pretrained",
    "hf_hub_download",
    "snapshot_download",
)

HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/model.pt"
AZURE_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_runtime(root: Path):
    path = root / "whisper" / "runtime.py"
    spec = importlib.util.spec_from_file_location("whisper_runtime_ci", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def scan_hub_imports(root: Path) -> List[Tuple[str, str]]:
    hits: List[Tuple[str, str]] = []
    whisper_dir = root / "whisper"
    for path in whisper_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(root)).replace("\\", "/")
        for token in HUB_IMPORTS:
            if token in text:
                hits.append((rel, token))
    return hits


def check_policy(runtime) -> List[str]:
    errors: List[str] = []
    os.environ["WHISPER_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not runtime.is_hub_url(HUB_URL):
        errors.append("is_hub_url missed a Hugging Face Hub URL")
    if runtime.is_hub_url(AZURE_URL):
        errors.append("official Azure checkpoint URL was classified as Hub")
    try:
        runtime.guard_weight_download(HUB_URL, "/tmp/model.pt")
        errors.append("guard_weight_download accepted a Hub URL")
    except RuntimeError as exc:
        if "no Hub" not in str(exc):
            errors.append("Hub refusal message missing: {!r}".format(exc))
    try:
        runtime.guard_weight_download(AZURE_URL, "/tmp/tiny.pt")
        errors.append("offline guard allowed a remote weight pull")
    except RuntimeError as exc:
        if "no weight pulls" not in str(exc) and "offline" not in str(exc):
            errors.append("offline refusal message missing: {!r}".format(exc))
    return errors


def main() -> int:
    root = repo_root()
    hits = scan_hub_imports(root)
    runtime = load_runtime(root)
    errors = check_policy(runtime)
    if hits or errors:
        sys.stderr.write("ERROR: network / model download is not blocked:\n")
        for rel, token in hits:
            sys.stderr.write("  {}: {}\n".format(rel, token))
        for err in errors:
            sys.stderr.write("  {}\n".format(err))
        return 1
    sys.stdout.write("OK: Hub and remote weight downloads are blocked\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

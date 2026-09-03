#!/usr/bin/env python3
"""CI guard: refuse Hugging Face Hub and remote weight pulls.

Loads whisper/policy.py by path (stdlib only). No torch, no keys.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_policy(root: Path):
    path = root / "whisper" / "policy.py"
    spec = importlib.util.spec_from_file_location("whisper_policy_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_policy(root: Path) -> list:
    errors = []
    policy = load_policy(root)
    if policy.DEFAULT_DEVICE != "cpu":
        errors.append("DEFAULT_DEVICE is not cpu")
    if not policy.is_hub_url("https://huggingface.co/openai/whisper"):
        errors.append("huggingface.co was not classified as Hub")
    if not policy.is_hub_url("https://hf.co/models/openai/whisper"):
        errors.append("hf.co was not classified as Hub")
    azure = "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
    if policy.is_hub_url(azure):
        errors.append("Azure CDN was misclassified as Hub")
    dest = str(root / "tiny.pt")
    try:
        policy.refuse_remote_download("https://huggingface.co/openai/whisper", dest)
        errors.append("Hub URL was not refused")
    except RuntimeError as exc:
        if "Hub" not in str(exc):
            errors.append("Hub refusal message missing: {}".format(exc))
    os.environ["WHISPER_OFFLINE"] = "1"
    try:
        policy.refuse_remote_download(azure, dest)
        errors.append("offline Azure pull was not refused")
    except RuntimeError as exc:
        if "offline" not in str(exc):
            errors.append("offline refusal message missing: {}".format(exc))
    return errors


def main() -> int:
    errors = check_policy(repo_root())
    if errors:
        sys.stderr.write("ERROR: Hub / weight-pull policy failed\n")
        for message in errors:
            sys.stderr.write("  {}\n".format(message))
        return 1
    sys.stdout.write("OK: Hub refused; offline weight pulls refused; CPU default\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

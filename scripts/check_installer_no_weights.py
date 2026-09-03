#!/usr/bin/env python3
"""CI assertion: the Cloud Agent installer must not download weights.

Stdlib only. No network. Localhost only. No secrets.
Fails if install.sh would pull checkpoints or bind a public interface.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / ".cursor" / "install.sh"
ENVIRONMENT_JSON = ROOT / ".cursor" / "environment.json"

WEIGHT_SUFFIXES = (".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".onnx")
# Small in-repo assets that share a weight-like suffix.
ALLOWLIST = frozenset(
    {
        "whisper/assets/mel_filters.npz",
        "whisper/assets/gpt2.tiktoken",
        "whisper/assets/multilingual.tiktoken",
    }
)

FORBIDDEN_IN_INSTALLER = (
    "load_model",
    "_download",
    "azureedge",
    "huggingface.co",
    "hf.co",
    "0.0.0.0",
    "wget ",
    "curl ",
)

SECRET_PATTERNS = (
    "API_KEY",
    "SECRET_KEY",
    "PRIVATE_KEY",
    "BEGIN RSA",
    "sk-",
)


def without_hash_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def tracked_files() -> list[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def check_install_sh() -> None:
    if not INSTALL_SH.is_file():
        fail("missing .cursor/install.sh")
    text = INSTALL_SH.read_text(encoding="utf-8")
    code = without_hash_comments(text)

    for token in FORBIDDEN_IN_INSTALLER:
        if token in code:
            fail(
                f".cursor/install.sh must not {token!r} (would pull weights or bind WAN)"
            )

    if re.search(r"(?m)^\s*whisper\b", code):
        fail(".cursor/install.sh must not invoke the whisper CLI (downloads weights)")

    if not re.search(r"(?m)^export WHISPER_NO_WEIGHT_DOWNLOAD=", text):
        fail("install.sh must export WHISPER_NO_WEIGHT_DOWNLOAD (default 1)")
    if not re.search(r"(?m)^export WHISPER_LOCALHOST_ONLY=", text):
        fail("install.sh must export WHISPER_LOCALHOST_ONLY (default 1)")

    if "import whisper" not in code:
        fail("install.sh readiness check must import whisper (not load_model)")
    if "XDG_CACHE_HOME" not in text:
        fail("install.sh must isolate XDG_CACHE_HOME for the import check")
    if "must not download model weights" not in text:
        fail("install.sh must fail if isolated cache contains checkpoints")


def check_environment_json() -> None:
    if not ENVIRONMENT_JSON.is_file():
        fail("missing .cursor/environment.json")
    raw = ENVIRONMENT_JSON.read_text(encoding="utf-8")
    data = json.loads(raw)
    if "ports" in data:
        fail("environment.json must not publish ports (localhost only)")
    if "0.0.0.0" in raw:
        fail("environment.json must not mention 0.0.0.0")
    if data.get("install") != "bash .cursor/install.sh":
        fail("environment.json install must be bash .cursor/install.sh")


def check_cursor_scripts_localhost_only() -> None:
    for path in sorted((ROOT / ".cursor").glob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        code = without_hash_comments(text) if path.suffix == ".sh" else text
        if "0.0.0.0" in code:
            fail(f"{path.relative_to(ROOT)} must not bind 0.0.0.0")
        for pattern in SECRET_PATTERNS:
            if pattern in text:
                fail(f"{path.relative_to(ROOT)} must not contain secrets ({pattern})")


def check_no_tracked_weights() -> None:
    for rel in tracked_files():
        if rel in ALLOWLIST:
            continue
        lower = rel.lower()
        if any(lower.endswith(suffix) for suffix in WEIGHT_SUFFIXES):
            fail(f"tracked weight/binary is not allowed: {rel}")


def main() -> int:
    check_install_sh()
    check_environment_json()
    check_cursor_scripts_localhost_only()
    check_no_tracked_weights()
    print("installer-no-weights: ok (no default weight download, localhost only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

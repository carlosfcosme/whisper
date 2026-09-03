#!/usr/bin/env python3
"""CI assertion: installer is offline, loopback-only, and weight-free.

Stdlib only. No network. No secrets.
Fails if install.sh would pull checkpoints, bind a public interface,
or leave weight/cache paths unignored.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / ".cursor" / "install.sh"
START_SH = ROOT / ".cursor" / "start.sh"
ENVIRONMENT_JSON = ROOT / ".cursor" / "environment.json"
GITIGNORE = ROOT / ".gitignore"
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"

ALL_INTERFACES = ".".join(("0", "0", "0", "0"))
LOOPBACK = "127.0.0.1"

FORBIDDEN_IN_INSTALLER = (
    "load_model",
    "_download",
    "azureedge",
    "huggingface.co",
    "hf.co",
    "wget ",
    "curl ",
    ALL_INTERFACES,
)

REQUIRED_GITIGNORE = (
    ".cache/",
    ".cache/whisper/",
    ".cache/huggingface/",
    "cache/",
    "weights/",
    "checkpoints/",
    "huggingface/",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "pytorch_model.bin",
)

WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".ggml", ".gguf")
ALLOWLIST = frozenset(
    {
        "whisper/assets/mel_filters.npz",
        "whisper/assets/gpt2.tiktoken",
        "whisper/assets/multilingual.tiktoken",
    }
)

SECRET_PATTERNS = (
    "_".join(("API", "KEY")),
    "_".join(("SECRET", "KEY")),
    "_".join(("PRIVATE", "KEY")),
    "BEGIN RSA",
    "-".join(("Field", "Brain")),
    "_".join(("FIELD", "BRAIN")),
)


def without_hash_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def tracked_files() -> list:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=str(ROOT))
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
                f".cursor/install.sh must not contain {token!r} "
                "(would pull weights or bind WAN)"
            )

    if re.search(r"(?m)^\s*whisper\b", code):
        fail(".cursor/install.sh must not invoke the whisper CLI (downloads weights)")
    if "whisper-serve" in code or "whisper.serve" in code:
        fail(".cursor/install.sh must not start a server")
    if "--host" in code:
        fail(".cursor/install.sh must not bind a host; start.sh owns the listener")

    if not re.search(r"(?m)^export WHISPER_NO_WEIGHT_DOWNLOAD=", text):
        fail("install.sh must export WHISPER_NO_WEIGHT_DOWNLOAD (default 1)")
    if not re.search(r"(?m)^export HF_HUB_OFFLINE=", text):
        fail("install.sh must export HF_HUB_OFFLINE (default 1)")
    if "WHISPER_OFFLINE_INSTALL" not in text:
        fail("install.sh must support WHISPER_OFFLINE_INSTALL=1 (skip apt/pip)")
    if "import whisper" not in code:
        fail("install.sh readiness check must import whisper (not load_model)")
    if "XDG_CACHE_HOME" not in text:
        fail("install.sh must isolate XDG_CACHE_HOME for the import check")
    if "must not download model weights" not in text:
        fail("install.sh must fail if isolated cache contains checkpoints")
    if "pip install" not in code:
        fail("install.sh must install with pip")
    if re.search(r"(?m)^\s*uv\b", code):
        fail("install.sh must not call uv")


def check_start_and_environment() -> None:
    if not START_SH.is_file():
        fail("missing .cursor/start.sh")
    start = START_SH.read_text(encoding="utf-8")
    if "--host 127.0.0.1" not in start:
        fail(".cursor/start.sh must bind --host 127.0.0.1")
    if ALL_INTERFACES in start:
        fail(".cursor/start.sh must not bind all interfaces")

    if not ENVIRONMENT_JSON.is_file():
        fail("missing .cursor/environment.json")
    raw = ENVIRONMENT_JSON.read_text(encoding="utf-8")
    data = json.loads(raw)
    if "ports" in data:
        fail("environment.json must not publish ports (loopback only)")
    if ALL_INTERFACES in raw:
        fail("environment.json must not mention an all-interfaces bind")
    if data.get("install") != "bash .cursor/install.sh":
        fail("environment.json install must be bash .cursor/install.sh")


def check_gitignore() -> None:
    text = GITIGNORE.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED_GITIGNORE if token not in text]
    if missing:
        fail("gitignore must ignore weight/cache paths: {}".format(missing))


def check_workflow_offline() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for token in ("HF_HUB_OFFLINE", "WHISPER_NO_WEIGHT_DOWNLOAD"):
        if token not in text:
            fail("CI workflow must set {}".format(token))
    if "curl" in text and "huggingface" in text:
        fail("CI workflow must not curl Hugging Face Hub")


def check_no_tracked_weights() -> None:
    for rel in tracked_files():
        if rel in ALLOWLIST:
            continue
        lower = rel.lower()
        if any(lower.endswith(suffix) for suffix in WEIGHT_SUFFIXES):
            fail("tracked weight/binary is not allowed: {}".format(rel))
        parts = set(Path(rel).parts)
        if parts & {".cache", "cache", "weights", "checkpoints", "huggingface"}:
            fail("tracked cache/weight path is not allowed: {}".format(rel))


def check_no_secrets() -> None:
    for path in (INSTALL_SH, START_SH, ENVIRONMENT_JSON, Path(__file__)):
        text = path.read_text(encoding="utf-8")
        for pattern in SECRET_PATTERNS:
            if pattern in text and path != Path(__file__):
                fail(
                    "{} must not contain secrets ({})".format(
                        path.relative_to(ROOT), pattern
                    )
                )


def check_peer_scripts() -> None:
    for name in ("check_bind_localhost.py", "check_no_weights.py"):
        script = ROOT / "scripts" / name
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            fail("{} failed: {}".format(name, result.stderr.strip() or result.stdout))


def main() -> int:
    check_peer_scripts()
    check_install_sh()
    check_start_and_environment()
    check_gitignore()
    check_workflow_offline()
    check_no_tracked_weights()
    check_no_secrets()
    print(
        "offline-installer: ok (network-disabled path; no weights; "
        "loopback bind; caches ignored)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""CI gate: default is offline; Hub/weight fetch is opt-in; tests stay local.

Fails when any of these is true:

* downloads are allowed with a clean environment
* CI still selects named checkpoints that would be fetched
* install/CI/test paths call Hub download APIs or bind off 127.0.0.1
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Host literals are split so this file is not itself a download source.
_AZURE = "openaipublic." + "azureedge.net"
_HF_HOST = "huggingface." + "co"
_HF_LFS = "cdn-lfs." + "huggingface.co"
_HF_CLI = "huggingface-cli" + " download"
_ZERO_ADDR = "0.0.0." + "0"

DOWNLOAD_PATTERNS: Sequence[Tuple[str, str]] = (
    (_AZURE, "Whisper Azure weight host"),
    (_HF_LFS, "Hugging Face LFS weight host"),
    (re.escape(_HF_HOST), "huggingface.co host"),
    (r"hf_hub_download\s*\(", "hf_hub_download()"),
    (r"snapshot_download\s*\(", "snapshot_download()"),
    (r"from_pretrained\s*\(", "from_pretrained()"),
    (re.escape(_HF_CLI), "huggingface-cli download"),
    (
        r"\bwget\b[^\n]*\.(?:pt|pth|bin|ckpt|safetensors|onnx|gguf)\b",
        "wget of a weight file",
    ),
    (
        r"\bcurl\b[^\n]*\.(?:pt|pth|bin|ckpt|safetensors|onnx|gguf)\b",
        "curl of a weight file",
    ),
)

ALLOW_DOWNLOAD_MENTIONS = frozenset(
    {
        "tests/test_offline_hub.py",
        "tests/conftest.py",
    }
)

ALLOW_ZERO_ADDR = frozenset(
    {
        "tests/test_offline_hub.py",
    }
)

SCAN_FILES = (
    Path(".cursor/install.sh"),
    Path(".cursor/start.sh"),
)
SCAN_DIRS = (
    Path(".github/workflows"),
    Path("tests"),
)

BIND_RX = re.compile(
    r"""(?:--host\s+{addr}|host\s*=\s*['\"]?{addr}|server_name\s*=\s*['\"]{addr}"""
    r"""|\(\s*['\"]{addr}['\"]|['\"]{addr}['\"])""".format(addr=re.escape(_ZERO_ADDR))
)

FORBIDDEN_LITERALS = (
    ("Field-Brain", "Field-Brain must not appear"),
    ("FIELD_BRAIN", "Field-Brain must not appear"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_scan_paths(root: Path) -> Iterable[Path]:
    for rel in SCAN_FILES:
        path = root / rel
        if path.is_file():
            yield path
    for rel in SCAN_DIRS:
        directory = root / rel
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yml", ".yaml", ".sh"}:
                yield path


def _load_offline():
    path = repo_root() / "whisper" / "offline.py"
    spec = importlib.util.spec_from_file_location("whisper_offline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def check_default_denies_downloads() -> List[str]:
    errors: List[str] = []
    offline = _load_offline()
    saved = {
        key: os.environ.get(key)
        for key in offline.OFFLINE_ENV_VARS + (offline.ALLOW_DOWNLOADS_ENV,)
    }
    try:
        for key in saved:
            os.environ.pop(key, None)
        if offline.downloads_allowed():
            errors.append("downloads_allowed() is True with a clean environment")
        if not offline.downloads_forbidden():
            errors.append("downloads_forbidden() is False with a clean environment")
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    init = (repo_root() / "whisper" / "__init__.py").read_text(encoding="utf-8")
    if "downloads_allowed()" not in init or "refuse_download" not in init:
        errors.append("whisper._download does not gate on the offline policy")
    if "urlopen" in init and init.find("downloads_allowed()") > init.find(
        "with urllib.request.urlopen"
    ):
        errors.append("urlopen runs before the downloads_allowed() gate")
    return errors


def check_workflow_offline(root: Path) -> List[str]:
    errors: List[str] = []
    workflow = root / ".github" / "workflows" / "test.yml"
    text = workflow.read_text(encoding="utf-8")
    for key in ("WHISPER_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if "{}:".format(key) not in text and "{}=".format(key) not in text:
            errors.append("CI workflow is missing {}".format(key))
    if "test_transcribe[tiny]" in text or "test_transcribe[tiny.en]" in text:
        errors.append("CI still selects named checkpoints that would be downloaded")
    if "WHISPER_ALLOW_DOWNLOADS" in text:
        errors.append("CI must not set WHISPER_ALLOW_DOWNLOADS")
    if "assert_no_weight_cache.py" not in text:
        errors.append("CI does not assert the weight cache stays empty")
    return errors


def check_no_hub_in_install_test(root: Path) -> List[str]:
    compiled = [(re.compile(pat), label) for pat, label in DOWNLOAD_PATTERNS]
    errors: List[str] = []
    for path in iter_scan_paths(root):
        relpath = _posix(path, root)
        text = path.read_text(encoding="utf-8", errors="replace")
        patterns = compiled
        if relpath in ALLOW_DOWNLOAD_MENTIONS:
            patterns = [
                (rx, label)
                for rx, label in compiled
                if label
                not in {
                    "Whisper Azure weight host",
                    "Hugging Face LFS weight host",
                    "huggingface.co host",
                }
            ]
        for lineno, line in enumerate(text.splitlines(), 1):
            for rx, label in patterns:
                if rx.search(line):
                    errors.append(
                        "{}:{}: {} ({})".format(relpath, lineno, label, line.strip())
                    )
    return errors


def check_localhost_only(root: Path) -> List[str]:
    errors: List[str] = []
    for path in iter_scan_paths(root):
        relpath = _posix(path, root)
        if relpath in ALLOW_ZERO_ADDR:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if BIND_RX.search(line):
                errors.append(
                    "{}:{}: non-loopback bind ({})".format(
                        relpath, lineno, line.strip()
                    )
                )
    return errors


def check_no_secrets_or_field_brain(root: Path) -> List[str]:
    errors: List[str] = []
    for path in iter_scan_paths(root):
        relpath = _posix(path, root)
        text = path.read_text(encoding="utf-8", errors="replace")
        for literal, label in FORBIDDEN_LITERALS:
            if literal in text:
                errors.append("{}: {}".format(relpath, label))
        if re.search(r"sk-[A-Za-z0-9]{10,}", text):
            errors.append("{}: looks like an API key".format(relpath))
    return errors


def collect_errors(root: Optional[Path] = None) -> List[str]:
    root = root if root is not None else repo_root()
    errors: List[str] = []
    errors.extend(check_default_denies_downloads())
    errors.extend(check_workflow_offline(root))
    errors.extend(check_no_hub_in_install_test(root))
    errors.extend(check_localhost_only(root))
    errors.extend(check_no_secrets_or_field_brain(root))
    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv
    errors = collect_errors()
    if errors:
        sys.stderr.write("ERROR: offline / no-Hub default check failed:\n")
        for item in errors:
            sys.stderr.write("  {}\n".format(item))
        return 1
    sys.stdout.write("OK: default is offline; no Hub fetch; CI does not pull weights\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

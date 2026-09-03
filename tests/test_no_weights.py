"""Committed-weight CI guard. No Hub. No download. No secrets."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPO / "scripts" / "check_no_weights.py"
HUB_SCRIPT = REPO / "scripts" / "check_no_hub.py"

DOWNLOAD_HELPERS = (
    "import huggingface_hub",
    "from huggingface_hub",
    "hf_hub_download(",
    "snapshot_download(",
    "hf_hub_url(",
    "cached_download(",
    "from_pretrained(",
)


def test_ci_no_weights_script_passes():
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "no-weights: ok" in result.stdout


def test_gitignore_covers_weights_and_caches():
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    for token in (
        ".cache/",
        "cache/",
        "weights/",
        ".huggingface/",
        "*.pt",
        "*.safetensors",
        "*.bin",
        ".env",
    ):
        assert token in text


def test_ci_no_hub_script_passes():
    result = subprocess.run(
        [sys.executable, str(HUB_SCRIPT)],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "no-hub: ok" in result.stdout


def test_download_helpers_unused_in_application_sources():
    hits = []
    for base in (REPO / "whisper", REPO / ".cursor"):
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for token in DOWNLOAD_HELPERS:
                if token in text:
                    hits.append("{}: {}".format(path.relative_to(REPO), token))
    assert hits == [], "download helpers must stay unused: {}".format(hits)


def test_no_field_brain_and_no_keys():
    forbidden = ("Field-Brain", "FIELD_BRAIN", "API_KEY", "SECRET_KEY", "BEGIN RSA")
    for path in (
        REPO / "whisper" / "__init__.py",
        REPO / "whisper" / "bind.py",
        REPO / "whisper" / "serve.py",
        REPO / "scripts" / "check_no_weights.py",
        REPO / ".cursor" / "start.sh",
    ):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text


def test_hub_offline_env_is_set():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_huggingface_and_cdn_urlopen_are_blocked():
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen("https://huggingface.co")
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )


def test_committed_weight_suffix_fails_check(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    import check_no_weights

    planted = tmp_path / "sneaky.pt"
    planted.write_bytes(b"not-a-real-checkpoint")
    violations = check_no_weights.find_violations(
        tmp_path, relative_paths=["sneaky.pt"]
    )
    assert violations == [("sneaky.pt", "model weight or checkpoint (.pt)")]

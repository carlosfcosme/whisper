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
    for token in (".cache/", "cache/", "weights/", "*.pt", "*.safetensors", ".env"):
        assert token in text


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


def test_no_field_brain_and_no_keys():
    forbidden = ("Field-Brain", "FIELD_BRAIN", "API_KEY", "SECRET_KEY", "BEGIN RSA")
    for path in (
        REPO / "whisper" / "bind.py",
        REPO / "whisper" / "serve.py",
        REPO / ".cursor" / "start.sh",
        REPO / "scripts" / "check_bind_localhost.py",
        REPO / "scripts" / "check_no_weights.py",
    ):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text

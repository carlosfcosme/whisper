"""Prove tests never download model weights. No torch. No Hub."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


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


def test_download_helper_uses_urlopen():
    source = (REPO / "whisper" / "__init__.py").read_text(encoding="utf-8")
    assert "urllib.request.urlopen" in source
    assert "def _download(" in source


def test_ci_pytest_does_not_run_transcribe():
    workflow = (REPO / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    assert "-k 'not test_transcribe'" in workflow
    assert "test_transcribe[tiny]" not in workflow
    assert "XDG_CACHE_HOME" in workflow


def test_transcribe_loads_cached_file_only():
    source = (REPO / "tests" / "test_transcribe.py").read_text(encoding="utf-8")
    assert "requires_local_weights" in source
    assert "pytest.skip" in source
    assert "load_model(cached)" in source
    assert "load_model(model_name)" not in source

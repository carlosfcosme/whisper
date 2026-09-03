"""Tests must not use the Hugging Face Hub."""

import ast
import os
from pathlib import Path

import pytest

import whisper

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"

HUB_TOKENS = (
    "huggingface",
    "hf_hub",
    "huggingface_hub",
    "from_pretrained",
    "hf.co",
)


def _python_test_files():
    skip = {"test_no_hub.py", "test_local_fixtures.py"}
    return sorted(
        path
        for path in TESTS_DIR.glob("test_*.py")
        if path.is_file() and path.name not in skip
    )


def test_hub_offline_env_is_set():
    assert os.environ.get("WHISPER_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_DATASETS_OFFLINE") == "1"


def test_load_model_does_not_fetch_when_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="offline"):
        whisper.load_model("tiny", device="cpu")
    assert list(tmp_path.rglob("*.pt")) == []
    assert list(tmp_path.rglob("*.pth")) == []


def test_ci_does_not_download_weights():
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text()
    assert "WHISPER_OFFLINE" in workflow
    assert "HF_HUB_OFFLINE" in workflow
    assert "test_transcribe[tiny]" not in workflow
    assert (
        "-k 'not test_transcribe'" in workflow or '-k "not test_transcribe"' in workflow
    )
    assert "Fail if CI downloaded weights" in workflow
    assert "local-fixtures" in workflow
    assert "python3 tests/local_fixtures.py" in workflow


def test_test_sources_do_not_reference_the_hub():
    hits = []
    for path in _python_test_files():
        text = path.read_text().lower()
        for token in HUB_TOKENS:
            if token in text:
                hits.append("{0}: {1}".format(path.name, token))
    assert hits == [], "tests must not reference the Hub: {0}".format(hits)


def test_test_sources_do_not_import_hub_modules():
    forbidden = {"huggingface_hub", "datasets", "transformers"}
    imported = []
    for path in _python_test_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in forbidden:
                    imported.append("{0}: {1}".format(path.name, name))
    assert imported == [], "tests must not import Hub clients: {0}".format(imported)

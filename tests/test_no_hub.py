"""Tests must not use the Hugging Face Hub."""

import ast
import os
from pathlib import Path

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
    return sorted(
        path
        for path in TESTS_DIR.glob("test_*.py")
        if path.is_file() and path.name != "test_no_hub.py"
    )


def test_hub_offline_env_is_set():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_DATASETS_OFFLINE") == "1"


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


def test_ci_skips_named_model_downloads():
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text()
    assert "-k 'not test_transcribe'" in workflow
    assert "test_transcribe[tiny]" not in workflow
    assert "test_transcribe[tiny.en]" not in workflow

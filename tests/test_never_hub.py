"""Unit tests must not contact the Hugging Face Hub.

Whisper loads official checkpoints from Azure
(openaipublic.azureedge.net in whisper/__init__.py) and ships tokenizer
vocab under whisper/assets/. Hugging Face Hub downloads are out of scope
for this suite.
"""

import ast
import os
import sys
import types
import urllib.request
from pathlib import Path

import pytest
from hub_offline import (
    HUB_OFFLINE_ENV,
    install_hub_client_guard,
    is_huggingface_hub_host,
    refuse_hub_download,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "whisper"
TESTS_ROOT = Path(__file__).resolve().parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
FORBIDDEN_MODULES = ("huggingface_hub", "transformers", "datasets")
HUB_URLS = (
    "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin",
    "https://hf.co/openai/whisper-tiny",
    "https://cdn-lfs.huggingface.co/repos/openai/whisper-tiny",
)
WEIGHT_URLS = (
    "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt",
    "file:///tmp/tiny.pt",
)


def _python_files(root: Path):
    return sorted(
        path
        for path in root.rglob("*.py")
        if ".venv" not in path.parts and "__pycache__" not in path.parts
    )


def _imported_names(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_hub_offline_env_is_set():
    missing = [name for name in HUB_OFFLINE_ENV if os.environ.get(name) != "1"]
    assert missing == [], f"Hub-offline env vars must be 1: {missing}"


def test_ci_sets_hub_offline_env():
    workflow = WORKFLOW.read_text()
    for name in HUB_OFFLINE_ENV:
        assert f"{name}:" in workflow, f"CI must set {name}"


def test_package_source_does_not_import_hub_clients():
    offenders = []
    for path in _python_files(PACKAGE_ROOT):
        imported = _imported_names(path) & set(FORBIDDEN_MODULES)
        if imported:
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {sorted(imported)}")
    assert offenders == [], f"package source imports Hub clients: {offenders}"


def test_unit_tests_do_not_import_hub_clients():
    offenders = []
    for path in _python_files(TESTS_ROOT):
        imported = _imported_names(path) & set(FORBIDDEN_MODULES)
        if imported:
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {sorted(imported)}")
    assert offenders == [], f"unit tests import Hub clients: {offenders}"


@pytest.mark.parametrize("url", HUB_URLS)
def test_urlopen_blocks_huggingface_hub(url):
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        urllib.request.urlopen(url)


def test_urlopen_blocks_huggingface_request_object():
    request = urllib.request.Request(HUB_URLS[0])
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        urllib.request.urlopen(request)


@pytest.mark.parametrize("url", WEIGHT_URLS)
def test_urlopen_blocks_weight_pulls(url):
    with pytest.raises(RuntimeError, match="weights|remote hosts"):
        urllib.request.urlopen(url)


def test_is_huggingface_hub_host_matches_cdn_and_aliases():
    assert is_huggingface_hub_host("huggingface.co")
    assert is_huggingface_hub_host("cdn-lfs.huggingface.co")
    assert is_huggingface_hub_host("cas-bridge.xethub.hf.co")
    assert not is_huggingface_hub_host("openaipublic.azureedge.net")
    assert not is_huggingface_hub_host("127.0.0.1")


def test_hub_client_guard_patches_download_helpers(monkeypatch):
    fake = types.ModuleType("huggingface_hub")

    def _should_not_run(*args, **kwargs):
        raise AssertionError("Hub client must not download")

    fake.hf_hub_download = _should_not_run
    fake.snapshot_download = _should_not_run
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    install_hub_client_guard()
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        fake.hf_hub_download(repo_id="openai/whisper-tiny", filename="config.json")
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        fake.snapshot_download(repo_id="openai/whisper-tiny")
    assert fake.hf_hub_download is refuse_hub_download


def test_huggingface_hub_download_is_offline():
    huggingface_hub = pytest.importorskip("huggingface_hub")
    with pytest.raises(Exception):
        huggingface_hub.hf_hub_download(
            repo_id="openai/whisper-tiny",
            filename="config.json",
        )

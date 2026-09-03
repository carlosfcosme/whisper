"""CI and unit tests must stay offline and never hit the Hugging Face Hub."""

import ast
import hashlib
import http.client
import importlib.util
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

import whisper

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "whisper"
TESTS_ROOT = Path(__file__).resolve().parent

HUB_OFFLINE_ENV = (
    "WHISPER_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)
FORBIDDEN_MODULES = ("huggingface_hub", "transformers", "datasets")
HUB_URLS = (
    "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin",
    "https://hf.co/openai/whisper-tiny",
)
WEIGHT_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/tiny.pt"
)


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(script: str, *args: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
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
    assert missing == [], "Hub-offline env vars must be 1: {}".format(missing)
    assert os.environ.get("HF_TOKEN") is None
    assert os.environ.get("HUGGING_FACE_HUB_TOKEN") is None
    assert whisper.weights_download_forbidden() is True


def test_package_source_does_not_import_hub_clients():
    offenders = []
    for path in _python_files(PACKAGE_ROOT):
        imported = _imported_names(path) & set(FORBIDDEN_MODULES)
        if imported:
            offenders.append(
                "{}: {}".format(path.relative_to(REPO_ROOT), sorted(imported))
            )
    assert offenders == [], "package source imports Hub clients: {}".format(offenders)


def test_unit_tests_do_not_import_hub_clients():
    offenders = []
    for path in _python_files(TESTS_ROOT):
        imported = _imported_names(path) & set(FORBIDDEN_MODULES)
        if imported:
            offenders.append(
                "{}: {}".format(path.relative_to(REPO_ROOT), sorted(imported))
            )
    assert offenders == [], "unit tests import Hub clients: {}".format(offenders)


@pytest.mark.parametrize("url", HUB_URLS)
def test_urlopen_blocks_huggingface_hub(url):
    with pytest.raises(RuntimeError, match="Hub"):
        urllib.request.urlopen(url)


def test_https_connection_to_hub_is_blocked():
    with pytest.raises(RuntimeError, match="Hub"):
        http.client.HTTPSConnection("huggingface.co")


def test_azure_weight_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(WEIGHT_URL)


def test_is_hub_url_detects_hub_and_mirrors():
    assert whisper.is_hub_url(HUB_URLS[0])
    assert whisper.is_hub_url("https://hf.co/openai/whisper-tiny")
    assert whisper.is_hub_url("https://cas-bridge.xethub.hf.co/blob")
    assert not whisper.is_hub_url(WEIGHT_URL)
    assert not whisper.is_hub_url("https://example.invalid/tiny.pt")


def test_download_refuses_when_offline(tmp_path):
    fake_url = (
        "https://example.invalid/"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/"
        "tiny.pt"
    )
    with pytest.raises(RuntimeError, match="offline"):
        whisper._download(fake_url, str(tmp_path), False)
    assert list(tmp_path.glob("*.pt")) == []


def test_download_refuses_hub_url(tmp_path, monkeypatch):
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with pytest.raises(RuntimeError, match="Hub"):
        whisper._download(HUB_URLS[0], str(tmp_path), False)
    assert list(tmp_path.glob("*")) == []


def test_offline_download_uses_cached_checkpoint(tmp_path):
    payload = b"cached-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    url = "https://example.invalid/whisper/models/{}/tiny.pt".format(digest)
    target = tmp_path / "tiny.pt"
    target.write_bytes(payload)
    assert whisper._download(url, str(tmp_path), in_memory=False) == str(target)


def test_load_model_offline_does_not_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="offline"):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.rglob("*.pt")) == []


def test_check_scripts_pass_on_this_tree():
    for script, extra in (
        ("check_gitignore_caches.py", ()),
        ("check_no_weights.py", ()),
        ("check_ci_offline.py", ()),
        ("assert_no_weight_cache.py", ()),
        ("assert_no_hub_fetch.py", ("--require-offline-env",)),
    ):
        result = _run(script, *extra)
        assert result.returncode == 0, result.stderr
        assert "OK:" in result.stdout


def test_assert_no_hub_fetch_requires_offline_env(monkeypatch):
    hub = _load_script("assert_no_hub_fetch.py")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    assert "HF_HUB_OFFLINE" in hub.offline_env_missing()
    assert hub.main(["--require-offline-env"]) == 1


def test_assert_no_hub_fetch_flags_cache(tmp_path):
    hub = _load_script("assert_no_hub_fetch.py")
    planted = tmp_path / "hub"
    planted.mkdir()
    (planted / "snapshot.bin").write_bytes(b"blob")
    assert hub.find_hub_artifacts([planted]) == [planted / "snapshot.bin"]


def test_assert_no_weight_cache_flags_planted_checkpoint(tmp_path):
    cache = _load_script("assert_no_weight_cache.py")
    planted = tmp_path / "whisper"
    planted.mkdir()
    (planted / "tiny.pt").write_bytes(b"not-weights")
    assert cache.find_cached_weights([planted]) == [planted / "tiny.pt"]


def test_check_no_weights_classifies_extensions():
    check = _load_script("check_no_weights.py")
    assert check.classify("models/tiny.pt", 100) is not None
    assert check.classify("weights/model.safetensors", 10) is not None
    assert check.classify("export/model.onnx", 10) is not None
    assert check.classify("whisper/assets/mel_filters.npz", 4271) is None
    assert check.classify("tests/jfk.flac", 1_152_693) is None
    assert check.classify("README.md", 800) is None
    assert check.classify("README.md", check.MAX_FILE_BYTES + 1) is not None


def test_check_ci_offline_flags_hub_and_tiny_download():
    checker = _load_script("check_ci_offline.py")
    assert checker._is_forbidden_cache_path("~/.cache/whisper")
    assert checker._is_forbidden_cache_path("~/.cache/huggingface")
    assert checker._is_forbidden_cache_path("weights/")
    assert checker._is_forbidden_cache_path("*.pt")
    assert not checker._is_forbidden_cache_path("${{ steps.pip-cache.outputs.dir }}")
    assert not checker._is_forbidden_cache_path("~/.cache/pre-commit")


def test_install_sh_does_not_download_weights():
    text = (REPO_ROOT / ".cursor" / "install.sh").read_text()
    code = "\n".join(
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    assert "load_model" not in code
    assert "_download" not in code
    assert "azureedge" not in code
    assert "huggingface.co" not in code
    assert "API_KEY" not in code
    assert "SECRET" not in text
    assert "WHISPER_OFFLINE" in text
    assert "HF_HUB_OFFLINE" in text

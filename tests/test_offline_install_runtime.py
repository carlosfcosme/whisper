"""Offline install and runtime: no weight download, loopback bind required."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from conftest import DownloadHelperInvoked

import whisper
from whisper.serve import BindError, main, require_bind_127_0_0_1

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL = REPO_ROOT / ".cursor" / "install.sh"
START = REPO_ROOT / ".cursor" / "start.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"

WEIGHT_FETCH_TOKENS = (
    "load_model(",
    "_download(",
    "hf_hub_download",
    "from_pretrained",
    "huggingface.co",
    "hf.co/",
    "openaipublic",
    "azureedge",
)


def test_install_script_does_not_fetch_weights():
    assert INSTALL.is_file()
    text = INSTALL.read_text(encoding="utf-8")
    assert "pip install" in text
    assert "HF_HUB_OFFLINE" in text
    assert "TRANSFORMERS_OFFLINE" in text
    assert "WHISPER_OFFLINE" in text
    for token in WEIGHT_FETCH_TOKENS:
        assert token not in text, token
    assert "import whisper, torch" in text
    assert "load_model" not in text


def test_start_script_requires_loopback_and_stays_offline():
    assert START.is_file()
    text = START.read_text(encoding="utf-8")
    assert "--host 127.0.0.1" in text
    assert "0.0.0.0" not in text
    assert "HF_HUB_OFFLINE" in text
    for token in WEIGHT_FETCH_TOKENS:
        assert token not in text, token


def test_runtime_named_model_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="offline|no weight pulls|no Hub"):
        whisper.load_model("tiny")
    assert list(tmp_path.rglob("*.pt")) == []
    assert list(tmp_path.rglob("*.pth")) == []


def test_runtime_urlopen_hub_fails():
    import urllib.request

    with pytest.raises(DownloadHelperInvoked, match="download helper"):
        urllib.request.urlopen("https://huggingface.co/openai/whisper-tiny")


def test_runtime_serve_requires_loopback():
    require_bind_127_0_0_1("127.0.0.1")
    with pytest.raises(BindError, match="127.0.0.1"):
        require_bind_127_0_0_1("0.0.0.0")
    assert main(["--host", "0.0.0.0", "--port", "0"]) == 2


def test_ci_offline_install_runtime_job_and_weights_ignored():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "offline-install-runtime:" in text
    assert "tests/test_offline_install_runtime.py" in text
    assert "git check-ignore" in text
    assert "scripts/check_no_weights.py" in text
    assert "HF_HUB_OFFLINE" in text
    assert "TRANSFORMERS_OFFLINE" in text
    assert "-k 'not test_transcribe'" in text
    assert "test_transcribe[tiny]" not in text
    assert "huggingface.co" not in text
    assert "load_model" not in text


def test_ci_gitignore_ignores_weight_paths():
    planted = [
        "cache/whisper/tiny.pt",
        ".cache/whisper/tiny.pt",
        "leaked.pt",
        "model.pth",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "-v", "--stdin"],
        cwd=str(REPO_ROOT),
        input="\n".join(planted) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    ignored = {line.split()[-1] for line in result.stdout.splitlines() if line}
    for path in planted:
        assert path in ignored, (path, result.stdout)


def test_install_probe_stays_offline():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert whisper.offline.offline_enabled()

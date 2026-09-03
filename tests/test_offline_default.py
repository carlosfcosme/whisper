"""Offline-default tests: fail network/model fetch; require local fixtures.

No whisper import. The CI offline-default job is pytest-only (no torch).
"""

import importlib.util
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent

pytestmark = [pytest.mark.localhost_only, pytest.mark.offline_default]


def _load_bind():
    path = ROOT / "whisper" / "bind.py"
    spec = importlib.util.spec_from_file_location("whisper_bind_offline_default", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _offline_check():
    path = ROOT / "scripts" / "check_default_offline.py"
    spec = importlib.util.spec_from_file_location("check_default_offline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_network_model_fetch_fails():
    """WAN / CDN model URLs must not be fetched on the default path."""
    with pytest.raises(RuntimeError, match="localhost-only"):
        urllib.request.urlopen("https://example.invalid/whisper/models/tiny.pt")


def test_huggingface_hub_url_fetch_fails():
    with pytest.raises(RuntimeError, match="localhost-only"):
        urllib.request.urlopen(
            "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"
        )


def test_non_loopback_ip_fetch_fails():
    with pytest.raises(RuntimeError, match="localhost-only"):
        urllib.request.urlopen("http://10.0.0.1/tiny.pt")


def test_local_audio_fixture_exists(sample_audio_path):
    path = Path(sample_audio_path)
    assert path.is_file()
    assert path.name == "jfk.flac"
    assert path.stat().st_size > 0
    parsed = urlparse(str(path))
    assert parsed.scheme in {"", "file"}
    assert not str(path).startswith(("http://", "https://"))
    assert path.read_bytes()[:4] == b"fLaC"


def test_local_file_url_is_allowed(sample_audio_path):
    """Local fixtures may be opened; only WAN fetches are refused."""
    uri = Path(sample_audio_path).resolve().as_uri()
    with urllib.request.urlopen(uri) as resp:
        assert resp.read(4) == b"fLaC"


def test_local_tokenizer_fixtures_exist():
    assets = ROOT / "whisper" / "assets"
    required = ("gpt2.tiktoken", "multilingual.tiktoken")
    for name in required:
        path = assets / name
        assert path.is_file(), f"missing local tokenizer fixture {path}"
        assert path.stat().st_size > 0
        assert not str(path).startswith(("http://", "https://"))


def test_bind_requires_127_0_0_1():
    bind = _load_bind()
    assert bind.BIND_HOST == "127.0.0.1"
    assert bind.require_bind_127_0_0_1("127.0.0.1") == "127.0.0.1"
    with pytest.raises(bind.BindError, match="127.0.0.1"):
        bind.require_bind_127_0_0_1("10.0.0.1")
    with pytest.raises(bind.BindError, match="127.0.0.1"):
        bind.require_bind_127_0_0_1("localhost")


def test_start_script_binds_127_0_0_1():
    text = (ROOT / ".cursor" / "start.sh").read_text()
    assert "--host 127.0.0.1" in text
    assert "WHISPER_OFFLINE" in text


def test_grep_loopback_bind_passes():
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "grep_loopback_bind.sh")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK:" in result.stdout


def test_ci_wires_offline_default_suite():
    checker = _offline_check()
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    reasons = checker.reasons_offline_default_job_incomplete(workflow)
    assert reasons == [], reasons
    block = checker.job_block(workflow, "offline-default")
    assert "test_offline_default.py" in block
    assert "grep_loopback_bind.sh" in block
    assert "WHISPER_OFFLINE" in block


def test_check_fails_when_ci_omits_offline_default_suite():
    checker = _offline_check()
    poisoned = """
jobs:
  offline-default:
    runs-on: ubuntu-latest
    steps:
      - run: echo skip
"""
    reasons = checker.reasons_offline_default_job_incomplete(poisoned)
    assert reasons, "checker must fail when CI drops the offline-default suite"
    assert any("test_offline_default.py" in r for r in reasons)
    assert any("grep_loopback_bind.sh" in r for r in reasons)
    assert any("WHISPER_OFFLINE" in r for r in reasons)

"""Ticket 4: CPU-only / offline / no-store defaults."""

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

import whisper
from whisper.runtime import (
    BIND_HOST,
    DEFAULT_DEVICE,
    DEFAULT_NO_STORE,
    DEFAULT_OFFLINE,
    is_hub_url,
    no_store_enabled,
    offline_enabled,
    refuse_remote_download,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
AZURE_TINY = "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"

POLICY_ENV = (
    "WHISPER_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "WHISPER_NO_STORE",
    "WHISPER_ALLOW_DOWNLOAD",
    "WHISPER_ALLOW_WEIGHT_FETCH",
    "WHISPER_ALLOW_STORE",
)


def _clear_policy_env(monkeypatch):
    for key in POLICY_ENV:
        monkeypatch.delenv(key, raising=False)


def _load_assert_script():
    path = REPO_ROOT / "scripts" / "assert_no_weight_download.py"
    spec = importlib.util.spec_from_file_location("assert_no_weight_download", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cpu_offline_no_store_are_defaults(monkeypatch):
    _clear_policy_env(monkeypatch)
    assert DEFAULT_DEVICE == "cpu"
    assert DEFAULT_OFFLINE is True
    assert DEFAULT_NO_STORE is True
    assert whisper.DEFAULT_DEVICE == "cpu"
    assert whisper.DEFAULT_OFFLINE is True
    assert whisper.DEFAULT_NO_STORE is True
    assert offline_enabled() is True
    assert no_store_enabled() is True
    assert BIND_HOST == "127.0.0.1"
    assert BIND_HOST != "0.0.0.0"


def test_offline_default_refuses_download_without_env(monkeypatch, tmp_path):
    _clear_policy_env(monkeypatch)
    dest = tmp_path / "tiny.pt"
    with pytest.raises(RuntimeError, match="offline"):
        refuse_remote_download(AZURE_TINY, str(dest))
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_no_store_refuses_persist_when_download_allowed(monkeypatch, tmp_path):
    _clear_policy_env(monkeypatch)
    monkeypatch.setenv("WHISPER_ALLOW_DOWNLOAD", "1")
    dest = tmp_path / "tiny.pt"
    assert offline_enabled() is False
    assert no_store_enabled() is True
    with pytest.raises(RuntimeError, match="no-store"):
        refuse_remote_download(AZURE_TINY, str(dest))
    assert not dest.exists()


def test_opt_out_flags(monkeypatch):
    _clear_policy_env(monkeypatch)
    monkeypatch.setenv("WHISPER_ALLOW_DOWNLOAD", "1")
    assert offline_enabled() is False
    _clear_policy_env(monkeypatch)
    monkeypatch.setenv("WHISPER_OFFLINE", "0")
    assert offline_enabled() is False
    _clear_policy_env(monkeypatch)
    monkeypatch.setenv("WHISPER_ALLOW_STORE", "1")
    assert no_store_enabled() is False
    _clear_policy_env(monkeypatch)
    monkeypatch.setenv("WHISPER_NO_STORE", "0")
    assert no_store_enabled() is False


def test_refuse_passes_when_download_and_store_allowed(monkeypatch):
    _clear_policy_env(monkeypatch)
    monkeypatch.setenv("WHISPER_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WHISPER_ALLOW_STORE", "1")
    refuse_remote_download(AZURE_TINY, "/tmp/tiny.pt")


def test_hub_urls_always_refused(monkeypatch):
    _clear_policy_env(monkeypatch)
    monkeypatch.setenv("WHISPER_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WHISPER_ALLOW_STORE", "1")
    for url in (
        "https://huggingface.co/openai/whisper-tiny",
        "https://hf.co/openai/whisper-tiny",
    ):
        assert is_hub_url(url)
        with pytest.raises(RuntimeError, match="Hugging Face Hub"):
            refuse_remote_download(url, "/tmp/x.pt")
    assert not is_hub_url(AZURE_TINY)


def test_load_model_default_does_not_store_weights(monkeypatch, tmp_path):
    _clear_policy_env(monkeypatch)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    with pytest.raises(RuntimeError, match="offline|no-store|no Hub"):
        whisper.load_model("tiny", device="cpu", download_root=str(tmp_path))
    leftover = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert leftover == []


def test_download_miss_does_not_mkdir_or_urlopen(monkeypatch, tmp_path):
    root = tmp_path / "missing-cache"

    def _boom_mkdir(*args, **kwargs):
        raise AssertionError("os.makedirs must not run on a refused download")

    def _boom_urlopen(*args, **kwargs):
        raise AssertionError("urlopen must not run on a refused download")

    monkeypatch.setattr(os, "makedirs", _boom_mkdir)
    monkeypatch.setattr("urllib.request.urlopen", _boom_urlopen)
    with pytest.raises(RuntimeError, match="offline|no-store|no Hub"):
        whisper._download(
            "https://openaipublic.azureedge.net/main/whisper/models/deadbeef/tiny.pt",
            str(root),
            in_memory=False,
        )
    assert not root.exists()


def test_download_cache_hit_does_not_fetch(tmp_path, monkeypatch):
    payload = b"local-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "tiny.pt"
    dest.write_bytes(payload)
    url = "https://openaipublic.azureedge.net/main/whisper/models/{}/tiny.pt".format(
        digest
    )

    def _boom(*args, **kwargs):
        raise AssertionError("urlopen must not run on a checksummed cache hit")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    result = whisper._download(url, str(tmp_path), in_memory=False)
    assert result == str(dest)


def test_ci_is_cpu_offline_and_skips_weight_pulls():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES" in workflow
    assert "WHISPER_OFFLINE" in workflow
    assert "WHISPER_NO_STORE" in workflow
    assert "HF_HUB_OFFLINE" in workflow
    assert "-k 'not test_transcribe'" in workflow
    assert "test_transcribe[tiny]" not in workflow
    assert "assert_no_weight_download.py" in workflow
    assert "+cpu" in workflow


def test_no_keys_and_no_field_brain():
    runtime = (REPO_ROOT / "whisper" / "runtime.py").read_text(encoding="utf-8")
    serve = (REPO_ROOT / "whisper" / "serve.py").read_text(encoding="utf-8")
    init = (REPO_ROOT / "whisper" / "__init__.py").read_text(encoding="utf-8")
    for text in (runtime, serve, init):
        assert "FIELD_BRAIN" not in text
        assert "Field-Brain" not in text
        assert "API_KEY" not in text
        assert "OPENAI_API_KEY" not in text


def test_assert_no_weight_download_clean(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    module = _load_assert_script()
    assert module.find_downloads() == []
    assert module.main() == 0


def test_assert_no_weight_download_detects_cache(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    xdg = tmp_path / "xdg"
    (xdg / "whisper").mkdir(parents=True)
    (xdg / "whisper" / "tiny.pt").write_bytes(b"checkpoint")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    module = _load_assert_script()
    hits = module.find_downloads()
    assert any(path.endswith("tiny.pt") for path in hits)
    assert module.main() == 1

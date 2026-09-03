import importlib.util
import urllib.request
from pathlib import Path

import pytest

import whisper
from whisper.runtime import (
    BIND_HOST,
    default_device,
    is_hub_url,
    is_offline,
    is_weight_url,
    refuse_forbidden_fetch,
    service_bind_host,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_check_no_hub_fetch():
    path = REPO_ROOT / "scripts" / "check_no_hub_fetch.py"
    spec = importlib.util.spec_from_file_location("check_no_hub_fetch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


check_no_hub_fetch = _load_check_no_hub_fetch()


def test_runtime_cpu_and_loopback():
    assert whisper.DEFAULT_DEVICE == "cpu"
    assert default_device() == "cpu"
    assert BIND_HOST == "127.0.0.1"
    assert service_bind_host() == "127.0.0.1"


def test_hub_urls_are_detected():
    assert is_hub_url("https://huggingface.co/openai/whisper-tiny")
    assert is_hub_url("https://hf.co/openai/whisper-tiny")
    assert not is_hub_url("http://127.0.0.1:8765/health")


def test_weight_urls_are_detected():
    assert is_weight_url(
        "https://openaipublic.azureedge.net/main/whisper/models/abc/tiny.pt"
    )
    assert is_weight_url("https://example.invalid/model.safetensors")
    assert not is_weight_url("http://127.0.0.1:8765/health")


def test_refuse_hub_fetch_always():
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        refuse_forbidden_fetch("https://huggingface.co/openai/whisper-tiny")


def test_refuse_weight_fetch_when_offline():
    with pytest.raises(RuntimeError, match="offline"):
        refuse_forbidden_fetch(
            "https://openaipublic.azureedge.net/main/whisper/models/abc/tiny.pt",
            offline=True,
        )


def test_urlopen_hub_is_blocked_in_tests():
    with pytest.raises(RuntimeError, match="Hub/weight"):
        urllib.request.urlopen("https://huggingface.co/openai/whisper-tiny")


def test_urlopen_azure_weight_is_blocked_in_tests():
    with pytest.raises(RuntimeError, match="Hub/weight"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/abc/tiny.pt"
        )


def test_huggingface_hub_import_is_blocked():
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        __import__("huggingface_hub")


def test_load_model_named_checkpoint_does_not_hit_network(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def boom(*_args, **_kwargs):
        raise AssertionError("urlopen must not run for a named checkpoint in tests")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="offline|Hub/weight|refusing"):
        whisper.load_model("tiny")


def test_ci_script_finds_no_hub_apis():
    assert check_no_hub_fetch.find_hub_api_uses(REPO_ROOT) == []
    assert check_no_hub_fetch.main() == 0


def test_ci_workflow_fails_on_hub_fetch_and_weights():
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    assert "no-committed-weights" in workflow
    assert "no-hub-fetch" in workflow
    assert "scripts/check_no_hub_fetch.py" in workflow
    assert "scripts/check_no_weights.py" in workflow
    assert "WHISPER_OFFLINE" in workflow


def test_tests_run_offline():
    assert is_offline()

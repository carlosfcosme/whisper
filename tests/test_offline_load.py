"""Default load path is offline: no Hub / urlretrieve / requests / urlopen."""

import hashlib
import sys
import types
import urllib.request

import pytest

import whisper
from whisper.runtime import (
    ALLOW_WEIGHT_DOWNLOAD_ENV,
    WeightDownloadError,
    weight_auto_download_allowed,
)

HF_HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"


def _patch_download_helpers(monkeypatch):
    """Patch network helpers. Any call fails the test (download was attempted)."""
    invoked = []

    def _mark(name):
        def boom(*args, **kwargs):
            invoked.append(name)
            raise AssertionError(
                "%s must not be invoked on the default offline load path" % name
            )

        return boom

    monkeypatch.setattr(urllib.request, "urlopen", _mark("urlopen"))
    monkeypatch.setattr(urllib.request, "urlretrieve", _mark("urlretrieve"))

    requests_mod = sys.modules.get("requests")
    if requests_mod is None:
        requests_mod = types.ModuleType("requests")
        sys.modules["requests"] = requests_mod
    monkeypatch.setattr(requests_mod, "get", _mark("requests.get"), raising=False)
    monkeypatch.setattr(requests_mod, "post", _mark("requests.post"), raising=False)

    hub = sys.modules.get("huggingface_hub")
    if hub is None:
        hub = types.ModuleType("huggingface_hub")
        sys.modules["huggingface_hub"] = hub
    monkeypatch.setattr(hub, "hf_hub_download", _mark("hf_hub_download"), raising=False)
    monkeypatch.setattr(
        hub, "snapshot_download", _mark("snapshot_download"), raising=False
    )
    return invoked


def test_default_is_offline_without_opt_in(monkeypatch):
    monkeypatch.delenv(ALLOW_WEIGHT_DOWNLOAD_ENV, raising=False)
    monkeypatch.delenv("WHISPER_NO_WEIGHT_DOWNLOAD", raising=False)
    monkeypatch.delenv("WHISPER_CPU_ONLY", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    assert weight_auto_download_allowed() is False


def test_hf_hub_offline_forces_offline_even_with_opt_in(monkeypatch):
    monkeypatch.setenv(ALLOW_WEIGHT_DOWNLOAD_ENV, "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert weight_auto_download_allowed() is False


def test_default_load_model_does_not_call_download_helpers(monkeypatch, tmp_path):
    monkeypatch.delenv(ALLOW_WEIGHT_DOWNLOAD_ENV, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    invoked = _patch_download_helpers(monkeypatch)

    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))

    assert invoked == []
    assert list(tmp_path.iterdir()) == []


def test_default_download_does_not_call_helpers_without_ci_flags(monkeypatch, tmp_path):
    monkeypatch.delenv(ALLOW_WEIGHT_DOWNLOAD_ENV, raising=False)
    monkeypatch.delenv("WHISPER_NO_WEIGHT_DOWNLOAD", raising=False)
    monkeypatch.delenv("WHISPER_CPU_ONLY", raising=False)
    monkeypatch.delenv("CI", raising=False)
    invoked = _patch_download_helpers(monkeypatch)

    with pytest.raises(WeightDownloadError):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), in_memory=False)

    assert invoked == []


def test_hf_hub_url_does_not_call_download_helpers(monkeypatch, tmp_path):
    invoked = _patch_download_helpers(monkeypatch)

    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        whisper._download(HF_HUB_URL, str(tmp_path), in_memory=False)

    assert invoked == []


def test_cache_hit_is_not_a_download(monkeypatch, tmp_path):
    payload = b"cached-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "tiny.pt"
    target.write_bytes(payload)
    url = "https://openaipublic.azureedge.net/main/whisper/models/%s/tiny.pt" % digest
    invoked = _patch_download_helpers(monkeypatch)

    assert whisper._download(url, str(tmp_path), in_memory=False) == str(target)
    assert invoked == []

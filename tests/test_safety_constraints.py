"""Safety gates: CPU default, offline Hub, no committed weights, 127.0.0.1."""

import importlib.util
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest
import torch

import whisper
from whisper.model import ModelDimensions, Whisper

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "{}.py".format(name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_device_is_cpu():
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_load_model_defaults_to_cpu(tmp_path):
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=16,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=1,
        n_vocab=50,
        n_text_ctx=16,
        n_text_state=32,
        n_text_head=4,
        n_text_layer=1,
    )
    model = Whisper(dims)
    ckpt = tmp_path / "toy.pt"
    torch.save({"dims": dims.__dict__, "model_state_dict": model.state_dict()}, ckpt)
    loaded = whisper.load_model(str(ckpt))
    assert loaded.device.type == "cpu"


def test_hub_offline_env_is_set():
    assert os.environ.get("WHISPER_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert whisper.weights_download_forbidden() is True


def test_remote_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen("https://huggingface.co")


def test_azure_weight_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )


def test_download_refuses_when_offline(tmp_path):
    fake_url = (
        "https://example.invalid/"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/"
        "tiny.pt"
    )
    with pytest.raises(RuntimeError, match="offline"):
        whisper._download(fake_url, str(tmp_path), False)


def test_check_no_weights_classifies_extensions():
    check = _load_script("check_no_weights")
    assert check.classify("models/tiny.pt", 100) is not None
    assert check.classify("weights/model.safetensors", 10) is not None
    assert check.classify("export/model.onnx", 10) is not None
    assert check.classify("libfoo.so", 100) is not None
    assert check.classify("whisper/assets/mel_filters.npz", 4271) is None
    assert check.classify("tests/jfk.flac", 1_152_693) is None
    assert check.classify("README.md", 800) is None
    assert check.classify("README.md", check.MAX_FILE_BYTES + 1) is not None


def test_check_no_weights_passes_on_this_repo():
    script = REPO_ROOT / "scripts" / "check_no_weights.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout


def test_install_test_policy_passes_on_this_repo():
    script = REPO_ROOT / "scripts" / "check_install_test_policy.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout


def test_install_test_policy_flags_weight_download(tmp_path):
    policy = _load_script("check_install_test_policy")
    cursor = tmp_path / ".cursor"
    cursor.mkdir()
    weight_name = "tiny" + ".pt"
    (cursor / "install.sh").write_text(
        "curl -O https://{host}/models/{name}\n".format(
            host=policy._AZURE, name=weight_name
        ),
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    hits = policy.find_download_violations(tmp_path)
    assert hits
    assert any("tiny.pt" in item or "curl" in item for item in hits)


def test_install_test_policy_flags_non_loopback_bind(tmp_path):
    policy = _load_script("check_install_test_policy")
    (tmp_path / ".cursor").mkdir()
    (tmp_path / ".cursor" / "start.sh").write_text(
        "python3 -m http.server --host {}\n".format(policy.ZERO_ADDR),
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    hits = policy.find_bind_violations(tmp_path)
    assert hits
    assert any(policy.ZERO_ADDR in item for item in hits)


def test_assert_no_weight_cache_passes():
    script = REPO_ROOT / "scripts" / "assert_no_weight_cache.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout


def test_assert_no_weight_cache_flags_planted_checkpoint(tmp_path):
    cache = _load_script("assert_no_weight_cache")
    planted = tmp_path / "whisper"
    planted.mkdir()
    (planted / "tiny.pt").write_bytes(b"not-weights")
    found = cache.find_cached_weights([planted])
    assert found == [planted / "tiny.pt"]


def test_ci_integration_script_passes():
    script = REPO_ROOT / "scripts" / "ci_integration.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "no committed weights" in result.stdout
    assert "cpu" in result.stdout
    assert "127.0.0.1" in result.stdout

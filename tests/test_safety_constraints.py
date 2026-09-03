import importlib.util
import os
import subprocess
import sys
import threading
import urllib.request
from http.client import HTTPConnection
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


def test_download_refuses_when_offline(tmp_path):
    fake_url = (
        "https://example.invalid/"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/"
        "tiny.pt"
    )
    with pytest.raises(RuntimeError, match="offline"):
        whisper._download(fake_url, str(tmp_path), False)


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
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "demo_server.py").write_text(
        "HOST = '127.0.0.1'\n", encoding="utf-8"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    hits = policy.find_download_violations(tmp_path)
    assert hits, "expected a download violation in a fake install.sh"
    assert any("tiny.pt" in item or "wget" in item or "curl" in item for item in hits)


def test_install_test_policy_flags_non_loopback_bind(tmp_path):
    policy = _load_script("check_install_test_policy")
    (tmp_path / ".cursor").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "demo_server.py").write_text(
        'ThreadingHTTPServer(("{}", 7860), Handler)\n'.format(policy.ZERO_ADDR),
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


def test_demo_server_defaults_to_loopback():
    demo = _load_script("demo_server")
    assert demo.DEFAULT_HOST == "127.0.0.1"
    with pytest.raises(ValueError, match="127.0.0.1"):
        demo.validate_host("0.0.0.0")
    with pytest.raises(ValueError, match="127.0.0.1"):
        demo.make_server("0.0.0.0", 7860)


def test_demo_server_binds_loopback_and_serves():
    demo = _load_script("demo_server")
    server = demo.make_server(demo.DEFAULT_HOST, 0)
    host, port = server.server_address[:2]
    assert host == "127.0.0.1"
    thread = threading.Thread(target=server.handle_request)
    thread.daemon = True
    thread.start()
    try:
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/")
        response = conn.getresponse()
        body = response.read()
        conn.close()
        assert response.status == 200
        assert b"127.0.0.1" in body
    finally:
        server.server_close()
        thread.join(timeout=5)

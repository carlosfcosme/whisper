import json
import os
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from whisper.serve import (
    DEFAULT_DEVICE,
    DEFAULT_HOST,
    ServeConfig,
    build_parser,
    create_server,
    is_hub_ref,
    is_remote_url,
    load_local_model,
    require_local_path,
    require_loopback_host,
)


def test_cli_defaults_are_localhost_cpu():
    args = build_parser().parse_args([])
    assert args.host == DEFAULT_HOST == "127.0.0.1"
    assert args.device == DEFAULT_DEVICE == "cpu"
    assert args.model is None


def test_require_loopback_host():
    assert require_loopback_host("127.0.0.1") == "127.0.0.1"
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback_host("0.0.0.0")
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback_host("localhost")


def test_rejects_hub_refs():
    assert is_hub_ref("openai/whisper-tiny")
    assert is_hub_ref("https://huggingface.co/openai/whisper-tiny")
    assert is_hub_ref("hf://openai/whisper-tiny")
    assert is_hub_ref("https://hf.co/openai/whisper-tiny")
    assert not is_hub_ref("tiny")
    assert not is_hub_ref("/tmp/model.pt")
    assert not is_hub_ref("checkpoints/tiny.pt")


def test_rejects_remote_urls():
    assert is_remote_url("https://example.com/audio.wav")
    assert not is_remote_url("/tmp/audio.wav")


def test_require_local_path_rejects_hub_and_urls(tmp_path):
    local = tmp_path / "model.pt"
    local.write_bytes(b"not-a-real-checkpoint")
    assert require_local_path(str(local), "checkpoint") == str(local.resolve())

    with pytest.raises(ValueError, match="Hub"):
        require_local_path("openai/whisper-tiny", "checkpoint")
    with pytest.raises(ValueError, match="remote"):
        require_local_path("https://example.com/model.pt", "checkpoint")
    with pytest.raises(FileNotFoundError):
        require_local_path(str(tmp_path / "missing.pt"), "checkpoint")


def test_load_local_model_rejects_hub_before_download():
    with pytest.raises(ValueError, match="Hub"):
        load_local_model("openai/whisper-tiny")
    with pytest.raises(ValueError, match="Hub"):
        load_local_model("https://huggingface.co/openai/whisper-tiny")


def test_create_server_rejects_non_loopback():
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_server(ServeConfig(host="0.0.0.0", port=0))


def test_health_binds_127_and_reports_cpu():
    server = create_server(host="127.0.0.1", port=0, device="cpu")
    host, port = server.server_address
    assert host == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/health") as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["status"] == "ok"
        assert payload["host"] == "127.0.0.1"
        assert payload["device"] == "cpu"
        assert payload["hub"] is False
        assert payload["model"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _DummyModel:
    def transcribe(self, audio, **kwargs):
        return {"text": "hello from local server", "language": "en"}


def test_transcribe_uses_local_audio_and_rejects_hub(tmp_path):
    server = create_server(host="127.0.0.1", port=0, device="cpu")
    server.model = _DummyModel()
    _, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    try:
        request = Request(
            f"http://127.0.0.1:{port}/transcribe",
            data=json.dumps({"audio": str(audio)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["text"] == "hello from local server"

        hub_request = Request(
            f"http://127.0.0.1:{port}/transcribe",
            data=json.dumps({"audio": "openai/whisper-tiny"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc:
            urlopen(hub_request)
        assert exc.value.code == 400
        error = json.loads(exc.value.read().decode("utf-8"))
        assert "Hub" in error["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_serve_module_does_not_use_hub():
    serve_path = os.path.join(os.path.dirname(__file__), "..", "whisper", "serve.py")
    source = open(serve_path, encoding="utf-8").read()
    assert "import huggingface_hub" not in source
    assert "from huggingface_hub" not in source
    assert "hf_hub_download" not in source
    assert "from_pretrained" not in source

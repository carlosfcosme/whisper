import json
import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from whisper.server import (
    DEFAULT_MODEL,
    DEFAULT_PORT,
    ServerState,
    build_parser,
    create_server,
)
from whisper.sources import DEFAULT_BIND_HOST, default_device


class FakeModel:
    def transcribe(self, audio, **kwargs):
        return {
            "text": "hello from fake model",
            "language": "en",
            "segments": [],
        }


@contextmanager
def running_server(state=None, host="127.0.0.1"):
    if state is None:
        state = ServerState(model=FakeModel())
    httpd = create_server(host, 0, state=state)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def _json(url, data=None, content_type="application/octet-stream"):
    headers = {}
    if data is not None:
        headers["Content-Type"] = content_type
    request = Request(url, data=data, headers=headers)
    with urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_server_parser_defaults_are_cpu_and_loopback():
    args = build_parser().parse_args([])
    assert args.host == "127.0.0.1"
    assert args.device == "cpu"
    assert args.model == DEFAULT_MODEL
    assert args.port == DEFAULT_PORT


def test_server_state_defaults_to_cpu_and_rejects_hub():
    state = ServerState()
    assert state.device == default_device()
    assert state.model_name == "tiny"
    with pytest.raises(ValueError, match="Hub"):
        ServerState(model_name="https://huggingface.co/openai/whisper-tiny")
    with pytest.raises(ValueError, match="Hub"):
        ServerState(download_root="https://huggingface.co/openai/whisper-tiny")


def test_create_server_rejects_non_loopback():
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_server("0.0.0.0", 0, state=ServerState(model=FakeModel()))


def _fail_if_loaded():
    raise RuntimeError("model should not load for /health")


def test_health_binds_127_0_0_1_without_loading_weights():
    state = ServerState(model_name="tiny", device=None, loader=_fail_if_loaded)
    with running_server(state) as httpd:
        host, port = httpd.server_address[:2]
        assert host == DEFAULT_BIND_HOST
        status, payload = _json("http://127.0.0.1:{}/health".format(port))
        assert status == 200
        assert payload["status"] == "ok"
        assert payload["device"] == "cpu"
        assert payload["host"] == "127.0.0.1"
        assert payload["hub"] is False
        assert payload["model"] == "tiny"


def test_transcribe_with_injected_model():
    with running_server() as httpd:
        _, port = httpd.server_address[:2]
        status, payload = _json(
            "http://127.0.0.1:{}/transcribe".format(port),
            data=b"fake-audio-bytes",
        )
        assert status == 200
        assert payload["text"] == "hello from fake model"
        assert payload["language"] == "en"


def test_transcribe_empty_body_is_400():
    with running_server() as httpd:
        _, port = httpd.server_address[:2]
        request = Request(
            "http://127.0.0.1:{}/transcribe".format(port),
            data=b"",
            method="POST",
            headers={"Content-Length": "0"},
        )
        with pytest.raises(HTTPError) as excinfo:
            urlopen(request, timeout=5)
        assert excinfo.value.code == 400

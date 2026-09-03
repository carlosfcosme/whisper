import urllib.request

import pytest

import whisper
from whisper.runtime import is_offline, refuse_forbidden_fetch


def test_tests_run_offline():
    assert is_offline()


def test_named_checkpoint_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("WHISPER_OFFLINE", "1")

    def boom(*_args, **_kwargs):
        raise AssertionError("urlopen must not run for a named checkpoint")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="offline|Hub/weight|refusing"):
        whisper.load_model("tiny")


def test_urlopen_weight_url_is_blocked():
    with pytest.raises(RuntimeError, match="Hub/weight"):
        urllib.request.urlopen(
            "https://" + "openaipublic.azureedge.net/main/whisper/models/abc/tiny.pt"
        )


def test_urlopen_hub_is_blocked():
    with pytest.raises(RuntimeError, match="Hub/weight"):
        urllib.request.urlopen("https://" + "huggingface.co" + "/openai/whisper-tiny")


def test_refuse_hub_fetch_always():
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        refuse_forbidden_fetch("https://" + "huggingface.co" + "/openai/whisper-tiny")


def test_huggingface_hub_import_is_blocked():
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        __import__("huggingface_hub")

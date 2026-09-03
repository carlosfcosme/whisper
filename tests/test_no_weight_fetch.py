"""Named-model loads must not fetch checkpoints in tests."""

import pytest

HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/model.pt"


def test_download_fails_when_urlopen_is_monkeypatched(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import urllib.request

    import whisper

    def _fail(url, *args, **kwargs):
        raise RuntimeError("network-call monkeypatch: weight fetch forbidden")

    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    monkeypatch.setattr(whisper.urllib.request, "urlopen", _fail)
    with pytest.raises(RuntimeError, match="monkeypatch"):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), False)
    assert list(tmp_path.glob("*.pt")) == []


def test_load_model_cache_miss_does_not_fetch(tmp_path, forbid_network_calls):
    pytest.importorskip("torch")
    import whisper

    with pytest.raises(RuntimeError, match="network"):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert not any(path.suffix == ".pt" for path in tmp_path.rglob("*"))


def test_download_hub_url_is_blocked(tmp_path, forbid_network_calls):
    pytest.importorskip("torch")
    import whisper

    with pytest.raises(RuntimeError, match="network"):
        whisper._download(HUB_URL, str(tmp_path), False)
    assert not any(path.suffix == ".pt" for path in tmp_path.rglob("*"))


def test_huggingface_hub_download_is_blocked():
    huggingface_hub = pytest.importorskip("huggingface_hub")
    with pytest.raises(RuntimeError, match="forbidden"):
        huggingface_hub.hf_hub_download(
            repo_id="openai/whisper-tiny",
            filename="config.json",
        )

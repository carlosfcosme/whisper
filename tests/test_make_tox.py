from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_makefile_and_tox_encode_offline_localhost_ci():
    makefile = (ROOT / "Makefile").read_text()
    tox_ini = (ROOT / "tox.ini").read_text()
    for text in (makefile, tox_ini):
        assert "WHISPER_OFFLINE" in text
        assert "HF_HUB_OFFLINE" in text
        assert "WHISPER_BIND_HOST" in text
        assert "127.0.0.1" in text
        assert "not test_transcribe" in text
        assert "not requires_weights" in text


def test_makefile_has_offline_targets():
    makefile = (ROOT / "Makefile").read_text()
    assert "test-offline" in makefile
    assert "tox-offline" in makefile
    assert "ci-offline" in makefile

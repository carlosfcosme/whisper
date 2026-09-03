import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_status_encodes_offline_no_hub_cpu_localhost():
    text = (ROOT / "STATUS.md").read_text()
    lowered = text.lower()
    assert "offline" in lowered
    assert "no hub" in lowered
    assert "cpu-only" in lowered or "cpu" in lowered
    assert "127.0.0.1" in text
    assert "WHISPER_OFFLINE=1" in text
    assert "WHISPER_NO_HUB=1" in text


def test_environment_json_encodes_offline_cpu_localhost():
    env = json.loads((ROOT / ".cursor" / "environment.json").read_text())
    assert env["install"] == "bash .cursor/install.sh"
    assert "offline" in env["name"]
    assert "cpu" in env["name"]
    assert "localhost" in env["name"]


def test_install_and_policy_env_encode_gates():
    install = (ROOT / ".cursor" / "install.sh").read_text()
    policy = (ROOT / ".cursor" / "whisper-policy.env").read_text()
    for blob in (install, policy):
        assert "WHISPER_OFFLINE=1" in blob
        assert "WHISPER_NO_HUB=1" in blob
        assert "HF_HUB_OFFLINE=1" in blob
    assert "127.0.0.1" in policy
    assert "WHISPER_DEVICE=cpu" in policy
    assert "torch==2.5.1+cpu" in install
    assert "127.0.0.1" in install

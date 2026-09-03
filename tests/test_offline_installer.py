"""Offline installer tests: network disabled, no model download.

Loopback bind only. Weight/cache paths stay gitignored. Stdlib-first so
the bind-and-weights CI job can run this without torch.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO / ".cursor" / "install.sh"
START_SH = REPO / ".cursor" / "start.sh"
ENVIRONMENT_JSON = REPO / ".cursor" / "environment.json"
CHECK_SCRIPT = REPO / "scripts" / "check_offline_installer.py"
ALL_INTERFACES = ".".join(("0", "0", "0", "0"))
LOOPBACK = "127.0.0.1"
HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"
CDN_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
)


def _load_offline():
    spec = importlib.util.spec_from_file_location(
        "whisper_offline_installer_isolated", REPO / "whisper" / "offline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _without_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _whisper_importable() -> bool:
    try:
        import importlib

        importlib.import_module("torch")
        importlib.import_module("whisper")
        return True
    except ImportError:
        return False


def _unshare_net(argv, *, env=None, cwd=None):
    if shutil.which("unshare") is None:
        pytest.skip("unshare is not available")
    cmd = ["unshare", "--net", "--map-root-user"] + argv
    probe = subprocess.run(
        ["unshare", "--net", "--map-root-user", "true"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("unshare --net --map-root-user is not permitted")
    return subprocess.run(
        cmd,
        cwd=cwd or REPO,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_offline_installer_ci_script_passes():
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "offline-installer: ok" in result.stdout


def test_install_sh_is_offline_and_weight_free():
    text = INSTALL_SH.read_text(encoding="utf-8")
    code = _without_comments(text)
    assert "export WHISPER_NO_WEIGHT_DOWNLOAD=" in text
    assert "export HF_HUB_OFFLINE=" in text
    assert "WHISPER_OFFLINE_INSTALL" in text
    assert "XDG_CACHE_HOME" in text
    assert "import whisper" in code
    assert "load_model" not in code
    assert "_download" not in code
    assert "pip install" in code
    assert "huggingface.co" not in code
    assert "azureedge" not in code
    assert ALL_INTERFACES not in code


def test_install_sh_does_not_bind_or_start_a_server():
    code = _without_comments(INSTALL_SH.read_text(encoding="utf-8"))
    assert "whisper-serve" not in code
    assert "whisper.serve" not in code
    assert "--host" not in code
    assert ALL_INTERFACES not in code


def test_start_sh_and_environment_are_loopback_only():
    start = START_SH.read_text(encoding="utf-8")
    assert "--host 127.0.0.1" in start
    assert ALL_INTERFACES not in start

    raw = ENVIRONMENT_JSON.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert "ports" not in data
    assert ALL_INTERFACES not in raw
    assert data["install"] == "bash .cursor/install.sh"


def test_gitignore_ignores_weights_and_caches():
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    for token in (
        ".cache/",
        ".cache/whisper/",
        ".cache/huggingface/",
        "cache/",
        "weights/",
        "checkpoints/",
        "huggingface/",
        "*.pt",
        "*.safetensors",
        "pytorch_model.bin",
    ):
        assert token in text


def test_no_tracked_weight_files():
    listed = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO)
    paths = [p for p in listed.decode("utf-8", "surrogateescape").split("\0") if p]
    suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".ggml", ".gguf"}
    bad = [
        path
        for path in paths
        if Path(path).suffix.lower() in suffixes
        or set(Path(path).parts)
        & {".cache", "cache", "weights", "checkpoints", "huggingface"}
    ]
    assert bad == []


def test_package_tree_has_no_weight_payloads():
    hits = [
        str(path.relative_to(REPO))
        for path in (REPO / "whisper").rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".pt", ".pth", ".safetensors", ".ggml"}
    ]
    assert hits == []


def test_network_and_model_fetch_are_monkeypatched_to_fail(
    isolated_cache, loopback_bind
):
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("WHISPER_NO_WEIGHT_DOWNLOAD") == "1"
    assert loopback_bind == LOOPBACK
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(HUB_URL)
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(CDN_URL)
    with pytest.raises(RuntimeError, match="forbidden"):
        socket.create_connection(("huggingface.co", 443), timeout=1)
    assert list(isolated_cache.rglob("*")) == [] or all(
        path.is_dir() for path in isolated_cache.rglob("*")
    )


def test_offline_module_refuses_hub_and_cdn_without_network(isolated_cache):
    offline = _load_offline()
    with pytest.raises(offline.WeightDownloadError, match="Hub"):
        offline.refuse_weight_network_pull(HUB_URL)
    with pytest.raises(offline.WeightDownloadError, match="disabled"):
        offline.refuse_weight_network_pull(CDN_URL)
    assert list(isolated_cache.iterdir()) == []


def test_unshare_net_blocks_wan():
    result = _unshare_net(
        [
            sys.executable,
            "-c",
            "import socket; socket.create_connection(('1.1.1.1', 80), 1)",
        ]
    )
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "huggingface" not in combined.lower()


def test_offline_install_sh_under_disabled_network(isolated_cache):
    if not _whisper_importable():
        pytest.skip("whisper/torch not installed in this job")
    env = os.environ.copy()
    env["WHISPER_OFFLINE_INSTALL"] = "1"
    env["WHISPER_NO_WEIGHT_DOWNLOAD"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["XDG_CACHE_HOME"] = str(isolated_cache)
    result = _unshare_net(["bash", str(INSTALL_SH)], env=env, cwd=REPO)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "whisper environment ready" in result.stdout
    leftover = [
        path
        for path in isolated_cache.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".pt", ".pth", ".safetensors", ".bin"}
    ]
    assert leftover == []


def test_named_model_does_not_download_under_unshare(isolated_cache):
    if not _whisper_importable():
        pytest.skip("whisper/torch not installed in this job")
    script = r"""
import os, sys
from pathlib import Path
cache = Path(os.environ["XDG_CACHE_HOME"])
from whisper.offline import WeightDownloadError
import whisper
raised = False
try:
    whisper.load_model("tiny", download_root=str(cache / "whisper"))
except WeightDownloadError:
    raised = True
if not raised:
    sys.exit(3)
leftover = [p for p in cache.rglob("*") if p.suffix in {".pt", ".pth", ".bin"}]
if leftover:
    sys.exit(4)
"""
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(isolated_cache)
    env["WHISPER_NO_WEIGHT_DOWNLOAD"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["CI"] = "1"
    result = _unshare_net([sys.executable, "-c", script], env=env)
    assert result.returncode == 0, result.stderr + result.stdout
    leftover = [
        path
        for path in isolated_cache.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".bin"}
    ]
    assert leftover == []


def test_loopback_bind_policy_from_installer_surface(loopback_bind):
    spec = importlib.util.spec_from_file_location(
        "whisper_bind_installer_isolated", REPO / "whisper" / "bind.py"
    )
    bind = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bind)
    assert bind.require_bind_127_0_0_1(None) == loopback_bind
    with pytest.raises(bind.BindError):
        bind.require_bind_127_0_0_1("")
    with pytest.raises(bind.BindError):
        bind.require_bind_127_0_0_1(ALL_INTERFACES)


def test_installer_paths_have_no_keys_or_field_brain():
    forbidden = (
        "-".join(("Field", "Brain")),
        "_".join(("FIELD", "BRAIN")),
        "_".join(("API", "KEY")),
        "_".join(("SECRET", "KEY")),
        "BEGIN RSA",
    )
    for path in (
        INSTALL_SH,
        START_SH,
        ENVIRONMENT_JSON,
        REPO / "whisper" / "offline.py",
        REPO / "whisper" / "bind.py",
    ):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, (path, token)

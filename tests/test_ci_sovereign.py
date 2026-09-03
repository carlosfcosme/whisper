"""CI tests: no weight fetch, local fixtures only, 127.0.0.1, ignored caches."""

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relpath: str):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sovereign = _load("whisper_sovereign_ci", "whisper/sovereign.py")
check_no_weights = _load("check_no_weights", "scripts/check_no_weights.py")
check_ignored_caches = _load("check_ignored_caches", "scripts/check_ignored_caches.py")
check_local_fixtures = _load("check_local_fixtures", "scripts/check_local_fixtures.py")
check_loopback_bind = _load("check_loopback_bind", "scripts/check_loopback_bind.py")


def test_no_weight_fetch_named_checkpoint_is_refused(tmp_path):
    dest = str(tmp_path / "tiny.pt")
    with pytest.raises(RuntimeError, match="no Hub"):
        sovereign.refuse_remote_download(
            "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt", dest
        )
    with pytest.raises(RuntimeError, match="no weight pulls"):
        sovereign.refuse_remote_download(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt", dest
        )


def test_no_weight_fetch_ci_env_and_workflow():
    assert os.environ.get("WHISPER_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    assert "HF_HUB_OFFLINE" in workflow
    assert "WHISPER_OFFLINE" in workflow
    assert "not test_transcribe" in workflow
    assert "check_no_weights.py" in workflow


def test_no_weight_fetch_transcribe_requires_preexisting_local_weights():
    source = (REPO_ROOT / "tests" / "test_transcribe.py").read_text(encoding="utf-8")
    assert "requires_local_weights" in source
    assert "must not download" in source
    assert "jfk.flac" in source
    download = (REPO_ROOT / "whisper" / "__init__.py").read_text(encoding="utf-8")
    assert "refuse_remote_download" in download


def test_preexisting_local_fixtures_only():
    assert check_local_fixtures.missing_fixtures(REPO_ROOT) == []
    assert check_local_fixtures.remote_fixture_hits(REPO_ROOT) == []
    assert check_local_fixtures.main() == 0
    jfk = REPO_ROOT / "tests" / "jfk.flac"
    assert jfk.is_file()
    tracked = subprocess.check_output(
        ["git", "ls-files", "--", "tests/jfk.flac"], cwd=REPO_ROOT, text=True
    ).strip()
    assert tracked == "tests/jfk.flac"


def test_audio_and_transcribe_use_only_jfk_fixture():
    for relpath in ("tests/test_audio.py", "tests/test_transcribe.py"):
        text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert "jfk.flac" in text
        assert ".pt" not in text
        assert "huggingface" not in text


def test_bind_127_0_0_1_ci_guard():
    assert sovereign.BIND_HOST == "127.0.0.1"
    assert check_loopback_bind.grep_all_interfaces(REPO_ROOT) == []
    check_loopback_bind.assert_bind_paths_loopback_only(REPO_ROOT)
    assert check_loopback_bind.main() == 0
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    assert "git grep -nF '0.0.0.0'" in workflow
    assert "loopback-bind" in workflow
    start = (REPO_ROOT / ".cursor" / "start.sh").read_text(encoding="utf-8")
    assert "127.0.0.1" in start
    assert "0.0.0.0" not in start


def test_ignored_caches_are_not_tracked():
    assert check_no_weights.find_violations(REPO_ROOT) == []
    assert check_ignored_caches.unignored_samples(REPO_ROOT) == []
    assert check_ignored_caches.main() == 0
    assert check_no_weights.main() == 0


@pytest.mark.parametrize(
    "relpath",
    [
        ".cache/whisper/tiny.pt",
        ".huggingface/hub/models--openai--whisper/config.json",
        "hf_cache/models--openai--whisper/config.json",
        ".hub/models--openai--whisper/blobs/abc",
        "model.safetensors",
        "tiny.pt",
    ],
)
def test_git_check_ignore_cache_and_weight_paths(relpath):
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "--", relpath],
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, relpath


def test_ci_job_runs_all_four_guards():
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    assert "sovereign-ci" in workflow
    assert "scripts/check_no_weights.py" in workflow
    assert "scripts/check_ignored_caches.py" in workflow
    assert "scripts/check_local_fixtures.py" in workflow
    assert "scripts/check_loopback_bind.py" in workflow
    assert "tests/test_ci_sovereign.py" in workflow

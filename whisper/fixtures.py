"""Local-only test fixture paths and executable no-WAN guards.

Fixture audio and assets must be in-repo files or tempfiles. HTTP(S) and
Hugging Face Hub URLs are refused so pytest never needs WAN. Autouse
fixtures also block weight downloads and restrict binds to 127.0.0.1.
"""

import os
import tempfile
import urllib.request
import wave
from typing import Union
from urllib.parse import urlparse

from .bind import BIND_HOST, BindError, bind_tcp, require_bind_host

PathLike = Union[str, os.PathLike]

REMOTE_SCHEMES = ("http://", "https://")
HUB_MARKERS = ("huggingface.co", "hf.co/")
_OFFLINE_VALUES = frozenset({"1", "true", "yes", "on"})


class RemoteFixtureError(ValueError):
    """Raised when a fixture path is an HTTP(S) or Hub URL."""


def is_remote_fixture_url(path: PathLike) -> bool:
    """True for http(s) URLs and Hugging Face Hub / hf.co addresses."""
    text = str(path).strip().lower()
    if text.startswith(REMOTE_SCHEMES):
        return True
    return any(marker in text for marker in HUB_MARKERS)


def assert_local_fixture(path: PathLike, must_exist: bool = True) -> str:
    """Return an absolute local path, or raise ``RemoteFixtureError``.

    Remote http(s) and Hugging Face URLs are always rejected. When
    ``must_exist`` is true the path must already be a regular file
    (in-repo asset or a tempfile that has been written).
    """
    if is_remote_fixture_url(path):
        raise RemoteFixtureError(
            "Fixture path must be local (in-repo or tempfile); refused {0!r}".format(
                str(path)
            )
        )
    resolved = os.path.abspath(os.path.expanduser(str(path)))
    if is_remote_fixture_url(resolved):
        raise RemoteFixtureError(
            "Fixture path must be local (in-repo or tempfile); refused {0!r}".format(
                resolved
            )
        )
    if must_exist and not os.path.isfile(resolved):
        raise RemoteFixtureError(
            "Fixture path is not a local file: {0}".format(resolved)
        )
    return resolved


def write_tiny_wav(
    path: PathLike, seconds: float = 0.05, sample_rate: int = 16000
) -> str:
    """Write a tiny silent WAV to ``path`` and return the local file path."""
    target = assert_local_fixture(path, must_exist=False)
    frames = int(sample_rate * seconds)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with wave.open(target, "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)
    return assert_local_fixture(target, must_exist=True)


class WeightDownloadError(RuntimeError):
    """Raised when a model-weight pull is refused (offline / no-WAN)."""


def offline_enabled() -> bool:
    """True when ``WHISPER_OFFLINE`` requests no weight downloads."""
    return os.environ.get("WHISPER_OFFLINE", "").strip().lower() in _OFFLINE_VALUES


def _request_url(url) -> str:
    if isinstance(url, str):
        return url
    getter = getattr(url, "get_full_url", None)
    if callable(getter):
        return getter()
    return getattr(url, "full_url", str(url))


def is_loopback_or_file_url(url: str) -> bool:
    """True for ``file:`` URLs and ``http(s)://127.0.0.1/...``."""
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return True
    if parsed.scheme not in ("http", "https"):
        return False
    return (parsed.hostname or "") == BIND_HOST


def refuse_weight_pull(url: str) -> None:
    """Raise ``WeightDownloadError`` for WAN / Hub / CDN weight URLs."""
    if is_loopback_or_file_url(url):
        return
    raise WeightDownloadError(
        "WHISPER_OFFLINE: refusing weight download from {0!r}".format(url)
    )


def install_weight_download_guard(monkeypatch) -> None:
    """Monkeypatch ``urlopen`` so WAN weight pulls fail closed."""
    original = urllib.request.urlopen

    def guarded(url, *args, **kwargs):
        refuse_weight_pull(_request_url(url))
        return original(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded)


def run_executable_checks() -> int:
    """CI entry: exercise local fixtures, bind 127.0.0.1, refuse WAN pulls."""
    os.environ.setdefault("WHISPER_OFFLINE", "1")
    handle, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(handle)
    try:
        written = write_tiny_wav(wav_path)
        assert_local_fixture(written)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    sock = bind_tcp(0)
    try:
        host, port = sock.getsockname()
        if host != BIND_HOST or port <= 0:
            print("FAIL: bind_tcp did not listen on 127.0.0.1", flush=True)
            return 1
    finally:
        sock.close()

    try:
        require_bind_host("0.0.0.0")
    except BindError:
        pass
    else:
        print("FAIL: wildcard bind was accepted", flush=True)
        return 1

    try:
        refuse_weight_pull(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )
    except WeightDownloadError:
        pass
    else:
        print("FAIL: WAN weight URL was accepted", flush=True)
        return 1

    print("OK: executable no-WAN fixtures (no weight pull, bind 127.0.0.1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_executable_checks())

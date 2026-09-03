"""Offline / localhost / CPU helpers.

Stdlib only so CI can run ``python3 whisper/offline.py --check`` without
installing torch. Fixtures are local files. Hugging Face Hub is refused.
Sockets bind 127.0.0.1. Device defaults to CPU.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from typing import List, Optional, Sequence
from urllib.parse import urlparse

LOCALHOST = "127.0.0.1"
WEIGHT_GLOBS = ("*.pt", "*.pth", "*.ckpt", "*.safetensors")
HUB_HOST_SUFFIXES = ("huggingface.co", "hf.co", "hf-mirror.com")
REMOTE_SCHEMES = ("http", "https", "hf", "huggingface")
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


class DownloadHelperCalled(AssertionError):
    """Raised when a network download helper is invoked while offline / in tests."""


class BindNotLoopback(ValueError):
    """Raised when a bind/listen host is not 127.0.0.1 (or loopback)."""


# In-repo fixtures. Resolved from the git/checkout root — never Hub URLs.
OFFLINE_FIXTURES = (
    "tests/jfk.flac",
    "whisper/assets/mel_filters.npz",
    "whisper/assets/gpt2.tiktoken",
    "whisper/assets/multilingual.tiktoken",
    "whisper/normalizers/english.json",
)


def offline_enabled() -> bool:
    """Offline is the default. Opt in to CDN pulls with WHISPER_OFFLINE=0
    or WHISPER_ALLOW_DOWNLOAD=1. Hugging Face Hub is never allowed.
    """
    offline = os.environ.get("WHISPER_OFFLINE")
    if offline is not None and offline.strip() != "":
        return offline.strip().lower() not in _FALSY
    allow = os.environ.get("WHISPER_ALLOW_DOWNLOAD", "").strip().lower()
    if allow in _TRUTHY:
        return False
    return True


def downloads_allowed() -> bool:
    return not offline_enabled()


def banned_download(*_args, **_kwargs):
    raise DownloadHelperCalled(
        "download helper called (offline by default; no Hub; no weight pull)"
    )


def default_device() -> str:
    """CPU-only inference path. CUDA availability never selects the device.

    Set WHISPER_DEVICE to override. CUDA is never required.
    """
    return os.environ.get("WHISPER_DEVICE", "cpu") or "cpu"


def assert_loopback(host: str) -> str:
    """Return a loopback address or raise BindNotLoopback."""
    raw = (host or "").strip()
    if raw in {LOCALHOST, "localhost"}:
        return LOCALHOST
    if raw == "::1":
        return "::1"
    raise BindNotLoopback(f"bind/listen must be 127.0.0.1 (loopback), got {host!r}")


def bind_host() -> str:
    """Return the loopback address helpers must bind. Never 0.0.0.0."""
    host = os.environ.get("WHISPER_BIND", LOCALHOST).strip() or LOCALHOST
    return assert_loopback(host)


def bind_localhost(port: int = 0) -> socket.socket:
    """Bind a TCP socket to 127.0.0.1 (or ::1). Caller closes the socket."""
    host = bind_host()
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    bound = sock.getsockname()[0]
    if bound not in {LOCALHOST, "::1"}:
        sock.close()
        raise BindNotLoopback(f"socket bound {bound!r}, not 127.0.0.1")
    return sock


def listen_localhost(port: int = 0, backlog: int = 1) -> socket.socket:
    """Bind and listen on 127.0.0.1 only. Caller closes the socket."""
    sock = bind_localhost(port)
    sock.listen(backlog)
    bound = sock.getsockname()[0]
    if bound not in {LOCALHOST, "::1"}:
        sock.close()
        raise BindNotLoopback(f"listen bound {bound!r}, not 127.0.0.1")
    return sock


def is_hub_url(value: str) -> bool:
    raw = (value or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    if lowered.startswith("hf://") or lowered.startswith("huggingface://"):
        return True
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower().rstrip(".")
    if host:
        for suffix in HUB_HOST_SUFFIXES:
            if host == suffix or host.endswith("." + suffix):
                return True
    return any(part in lowered for part in HUB_HOST_SUFFIXES)


def is_remote_url(value: str) -> bool:
    parsed = urlparse(value or "")
    return parsed.scheme.lower() in REMOTE_SCHEMES


def require_local_path(path: str, *, must_exist: bool = True) -> str:
    """Return an absolute local filesystem path. Rejects Hub and http(s)."""
    if is_hub_url(path) or is_remote_url(path):
        raise ValueError(
            f"fixture must be a local path (no Hub, no network pull): {path!r}"
        )
    resolved = os.path.abspath(path)
    if must_exist and not os.path.isfile(resolved):
        raise FileNotFoundError(resolved)
    return resolved


def git_root(start: Optional[str] = None) -> Optional[str]:
    start = start or os.getcwd()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return out.decode().strip()


def resolve_offline_fixture(relpath: str, root: Optional[str] = None) -> str:
    if os.path.isabs(relpath) or is_remote_url(relpath) or is_hub_url(relpath):
        return require_local_path(relpath)
    base = root or git_root() or os.getcwd()
    return require_local_path(os.path.join(base, relpath))


def offline_fixture_paths(root: Optional[str] = None) -> List[str]:
    return [resolve_offline_fixture(rel, root=root) for rel in OFFLINE_FIXTURES]


def guard_download_url(url: str) -> None:
    """Refuse Hub always. Refuse every remote pull unless downloads are opted in."""
    if is_hub_url(url):
        raise RuntimeError(f"Hugging Face Hub downloads are disabled: {url}")
    if offline_enabled() and is_remote_url(url):
        raise RuntimeError(f"offline mode: refusing network pull: {url}")


def committed_weight_files(root: Optional[str] = None) -> List[str]:
    """Tracked weight artifacts (``.pt`` / ``.pth`` / ``.ckpt`` / ``.safetensors``)."""
    cwd = root or git_root() or os.getcwd()
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "-z", "--", *WEIGHT_GLOBS],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return _walk_weight_files(cwd)
    return [p.decode() for p in out.split(b"\0") if p]


def _walk_weight_files(root: str) -> List[str]:
    hits: List[str] = []
    suffixes = tuple(g[1:] for g in WEIGHT_GLOBS)  # ".pt", ...
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", ".venv", "venv"}]
        for name in filenames:
            if name.endswith(suffixes):
                hits.append(os.path.relpath(os.path.join(dirpath, name), root))
    return sorted(hits)


def assert_no_committed_weights(root: Optional[str] = None) -> None:
    hits = committed_weight_files(root)
    if hits:
        listed = "\n".join(hits)
        raise AssertionError(f"committed model weights are forbidden:\n{listed}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args in ([], ["--check"], ["check-weights"]):
        hits = committed_weight_files()
        if hits:
            print("Committed model weights are forbidden:")
            for hit in hits:
                print(hit)
            return 1
        print("ok: no committed weight files")
        return 0
    print("usage: python3 whisper/offline.py --check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

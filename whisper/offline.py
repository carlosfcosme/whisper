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

# In-repo fixtures. Resolved from the git/checkout root — never Hub URLs.
OFFLINE_FIXTURES = (
    "tests/jfk.flac",
    "whisper/assets/mel_filters.npz",
    "whisper/assets/gpt2.tiktoken",
    "whisper/assets/multilingual.tiktoken",
    "whisper/normalizers/english.json",
)


def offline_enabled() -> bool:
    return os.environ.get("WHISPER_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def default_device() -> str:
    """CPU is the default. Set WHISPER_DEVICE (e.g. cuda) to override."""
    return os.environ.get("WHISPER_DEVICE", "cpu") or "cpu"


def bind_host() -> str:
    """Return the loopback address helpers must bind. Never 0.0.0.0."""
    host = os.environ.get("WHISPER_BIND", LOCALHOST).strip() or LOCALHOST
    allowed = {"127.0.0.1", "localhost", "::1"}
    if host not in allowed:
        raise ValueError(f"WHISPER_BIND must be loopback ({LOCALHOST}), got {host!r}")
    return LOCALHOST if host == "localhost" else host


def bind_localhost(port: int = 0) -> socket.socket:
    """Bind a TCP socket to 127.0.0.1 (or ::1). Caller closes the socket."""
    host = bind_host()
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
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
    """Refuse Hub always. Refuse every remote pull when WHISPER_OFFLINE=1."""
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

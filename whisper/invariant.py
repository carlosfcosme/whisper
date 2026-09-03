"""Offline invariant: localhost binds, no bootstrap weight pull, gitignored caches.

Stdlib-only when run as ``python3 whisper/invariant.py`` (no torch, no WAN).
"""

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "tiny.pt",
    "model.pth",
)
LS_FILES_PATHSPECS = (
    ".cache",
    ".cache/**",
    "cache",
    "cache/**",
    "weights",
    "weights/**",
    "*.pt",
    "*.pth",
)
_BOOTSTRAP_CALLS = frozenset({"urlopen", "_download", "load_model"})


def _load_sibling(name):
    """Load a whisper/*.py sibling without importing the torch package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "whisper_{0}_invariant".format(name), PACKAGE_DIR / "{0}.py".format(name)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bind():
    return _load_sibling("bind")


def _git(*args):
    return subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )


def check_bind_localhost():
    """Every service bind must be 127.0.0.1."""
    errors = []
    bind = _load_bind()
    host = bind.bind_host()
    if host != "127.0.0.1":
        errors.append("bind_host() returned {0}".format(host))
    if bind.bind_host("localhost") != "127.0.0.1":
        errors.append("bind_host('localhost') did not resolve to 127.0.0.1")
    sock = bind.listen()
    try:
        bound = sock.getsockname()[0]
        if bound != "127.0.0.1":
            errors.append("listen() bound {0}".format(bound))
    finally:
        sock.close()
    refused = tuple(bind.WILDCARD_HOSTS) + ("8.8.8.8", "1.2.3.4")
    for bad in refused:
        try:
            bind.bind_host(bad)
            errors.append("bind_host({0!r}) was accepted".format(bad))
        except ValueError:
            pass
    return errors


def check_cache_gitignored():
    """Weight and cache paths must be gitignored and untracked."""
    errors = []
    lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    patterns = _load_sibling("offline").GITIGNORED_WEIGHT_PATTERNS
    missing = [p for p in patterns if p not in lines]
    if missing:
        errors.append("gitignore missing {0}".format(missing))

    listed = _git("ls-files", "-z", "--", *LS_FILES_PATHSPECS)
    if listed.returncode != 0:
        errors.append("git ls-files failed: {0}".format(listed.stderr.strip()))
    else:
        tracked = [path for path in listed.stdout.split("\0") if path]
        if tracked:
            errors.append("tracked weight/cache paths: {0}".format(tracked))

    for path in IGNORE_EXAMPLES:
        if _git("check-ignore", "-q", "--", path).returncode != 0:
            errors.append("not gitignored: {0}".format(path))
    return errors


def _call_name(node):
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def check_no_bootstrap_download():
    """Package import must not call urlopen / _download / load_model."""
    errors = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if path.name == "__main__.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                name = _call_name(child)
                if name in _BOOTSTRAP_CALLS:
                    errors.append("{0}: module-level {1}()".format(path.name, name))
    return errors


def check_offline_invariant():
    """Run all offline invariant checks. Returns a list of error strings."""
    errors = []
    errors.extend(check_bind_localhost())
    errors.extend(check_cache_gitignored())
    errors.extend(check_no_bootstrap_download())
    return errors


def main(argv=None):
    del argv
    errors = check_offline_invariant()
    if errors:
        for item in errors:
            sys.stderr.write("{0}\n".format(item))
        return 1
    sys.stdout.write("offline invariant OK\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Load whisper stdlib modules without importing torch."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]

_BIND_STANDALONE = None


def load_bind_standalone():
    """Load ``whisper/bind.py`` without installing a ``whisper`` package stub."""
    global _BIND_STANDALONE
    if _BIND_STANDALONE is not None:
        return _BIND_STANDALONE
    path = ROOT / "whisper" / "bind.py"
    spec = importlib.util.spec_from_file_location("_whisper_bind_standalone", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _BIND_STANDALONE = module
    return module


def load_stdlib_modules(*names: str) -> Dict[str, object]:
    """Load bind/serve. Uses the real package when torch is already imported."""
    existing = sys.modules.get("whisper")
    if existing is not None and getattr(existing, "__file__", None):
        return {name: importlib.import_module("whisper.%s" % name) for name in names}

    pkg_dir = ROOT / "whisper"
    pkg = types.ModuleType("whisper")
    pkg.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
    pkg.__package__ = "whisper"
    sys.modules["whisper"] = pkg
    loaded: Dict[str, object] = {}
    for name in names:
        full = "whisper.%s" % name
        if full in sys.modules:
            loaded[name] = sys.modules[full]
            continue
        path = pkg_dir / ("%s.py" % name)
        spec = importlib.util.spec_from_file_location(full, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded

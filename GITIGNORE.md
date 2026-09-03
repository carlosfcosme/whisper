# Gitignore caches

[`.gitignore`](.gitignore) lists local caches so they stay untracked.
This file documents those patterns. It does not download model weights
and does not require secrets.

## Inventory

| Pattern | Cache | Produced by |
| --- | --- | --- |
| `__pycache__/` | Bytecode directories | Importing or running Python modules |
| `*.py[cod]` | `.pyc`, `.pyo`, `.pyd` | CPython, optimized bytecode, extensions |
| `*$py.class` | Jython class files | Jython |
| `*.egg-info` | setuptools metadata | `pip install -e .` / `pip install -e ".[dev]"` |
| `.pytest_cache` | pytest cache | `pytest` (dev extra and CI) |
| `.ipynb_checkpoints` | Jupyter autosave | Opening `notebooks/*.ipynb` |

OS / editor entries in the same file (not tool caches): `thumbs.db`,
`.DS_Store`, `.idea`.

## Confirm

```bash
git check-ignore -v __pycache__/x.pyc foo.pyc
git check-ignore -v openai_whisper.egg-info/PKG-INFO
git check-ignore -v .pytest_cache/v/cache
git check-ignore -v .ipynb_checkpoints/x.ipynb
```

Each command should print the matching `.gitignore` line.

## Out of tree (not a `.gitignore` pattern)

`whisper.load_model()` writes checkpoints to `$XDG_CACHE_HOME/whisper` when
`XDG_CACHE_HOME` is set, otherwise `~/.cache/whisper`. Those paths are
outside this checkout by default. Do not commit `.pt` files. Do not add
API keys, tokens, or `.env` files.

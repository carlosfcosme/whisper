# Installer

The Cloud Agent environment installs Python packages with **pip**, not **uv**.

`.cursor/environment.json` runs `bash .cursor/install.sh`. That script calls
system `pip` (`/usr/bin/pip`) with `--break-system-packages` so wheels land
in the user site. GitHub Actions and the README Setup snippets also use
`pip` / `pip3` / `python -m pip`.

`uv` is not a project dependency, is not on `PATH`, and there is no
`uv.lock` or `[tool.uv]` in [`pyproject.toml`](pyproject.toml). Do not run
`uv pip install` or `uv sync` unless those files change.

This document does not download model weights and does not require secrets.

## Commands the tree actually runs

Cloud Agent ([`.cursor/install.sh`](.cursor/install.sh)):

```bash
pip install --break-system-packages \
  "numpy" torch==2.5.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple

pip install --break-system-packages -e ".[dev]"
```

GitHub Actions tests ([`.github/workflows/test.yml`](.github/workflows/test.yml)):

```bash
pip install --upgrade pre-commit
pip3 install .["dev"] <numpy-requirement> torch==<matrix>+cpu \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
```

Release ([`.github/workflows/python-publish.yml`](.github/workflows/python-publish.yml)):

```bash
python -m pip install --upgrade pip
pip install setuptools wheel twine
```

User-facing README Setup:

```bash
pip install -U openai-whisper
```

## Inventory

| Signal | Present? | Where this was checked |
| --- | --- | --- |
| `pip` / `pip3` install commands | Yes | [`.cursor/install.sh`](.cursor/install.sh), [`.github/workflows/test.yml`](.github/workflows/test.yml), [`.github/workflows/python-publish.yml`](.github/workflows/python-publish.yml), [`README.md`](README.md) |
| `uv.lock` | No | repository root and `git ls-files` |
| `[tool.uv]` | No | [`pyproject.toml`](pyproject.toml) |
| `uv` as a dependency | No | [`pyproject.toml`](pyproject.toml), [`requirements.txt`](requirements.txt) |
| `Pipfile` / `poetry.lock` | No | repository root |

## Confirm on a Cloud Agent VM

```bash
command -v pip
pip --version
command -v uv   # expected: not found
```

On this environment those resolve to system pip 24.0
(`/usr/lib/python3/dist-packages/pip`, Python 3.12). `uv` is absent.

## Out of scope

- Do not download or commit checkpoints (`.pt`).
- Do not add API keys, tokens, or credentials.

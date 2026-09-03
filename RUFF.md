# Ruff

Status: **not present**.

This repository does not ship Ruff. There is no `ruff.toml`, `.ruff.toml`,
or `[tool.ruff]` table. The Cloud Agent environment therefore does not
install or run `ruff`; that was implicit.

Do not run `ruff check` / `ruff format` or add a Ruff config unless you
intend to migrate the lint stack.

## Inventory

| Item | Present? | Where this was checked |
|------|----------|------------------------|
| `ruff.toml` | No | repository root and `git ls-files` |
| `.ruff.toml` | No | repository root and `git ls-files` |
| `[tool.ruff]` | No | [`pyproject.toml`](pyproject.toml) |
| `ruff` in `.[dev]` | No | [`pyproject.toml`](pyproject.toml) optional-dependencies |
| `ruff` pre-commit hook | No | [`.pre-commit-config.yaml`](.pre-commit-config.yaml) |
| `ruff` on Cloud Agent `PATH` | No | `command -v ruff` after `.cursor/install.sh` |

## Implicit environment (actual linter)

Dev extras from [`pyproject.toml`](pyproject.toml): `black`, `flake8`,
`isort`, `pytest`, `scipy`. Lint is **flake8**, configured in
[`.flake8`](.flake8) and invoked from
[`.pre-commit-config.yaml`](.pre-commit-config.yaml).

[`.flake8`](.flake8):

```ini
[flake8]
per-file-ignores =
    */__init__.py: F401
```

Pre-commit / CI args (line length 88, same as Black / isort):

```bash
flake8 --max-line-length 88 --ignore E203,E501,W503,W504
```

GitHub Actions runs the same stack through pre-commit:

```bash
pip install --upgrade pre-commit
pre-commit install --install-hooks
pre-commit run --all-files
```

[`.cursor/install.sh`](.cursor/install.sh) installs flake8 via
`pip install -e ".[dev]"` and does not install Ruff.

## Weights and secrets

This document does not download model weights and does not require
secrets. Do not commit `.pt` files. No credentials are needed to run
flake8 or pre-commit.

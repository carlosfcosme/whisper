# Mypy

**If any:** none. This checkout does not install or configure [mypy](https://mypy-lang.org/).

The developer / Cloud Agent environment treated a type checker as implicit.
Do not run `mypy` here, and do not add a mypy config just to document this.

## Inventory

| Item | Present? |
|------|----------|
| `mypy` on `PATH` | No |
| `import mypy` | No (`ModuleNotFoundError`) |
| `mypy` in `[project.dependencies]` | No |
| `mypy` in `optional-dependencies.dev` | No (that extra is `black`, `flake8`, `isort`, `pytest`, `scipy`) |
| `[tool.mypy]` in [`pyproject.toml`](pyproject.toml) | No |
| `mypy.ini` / `.mypy.ini` / `setup.cfg` | No |
| pre-commit mypy hook | No ([`.pre-commit-config.yaml`](.pre-commit-config.yaml)) |
| CI mypy job | No ([`.github/workflows/test.yml`](.github/workflows/test.yml)) |

`.cursor/install.sh` installs `.[dev]`. That extra is lint + test tooling, not a type checker.
Static checks that *are* present: Black, isort, flake8 (locally and via the CI `pre-commit` job).

`whisper/` uses `typing` annotations for readers and IDEs. Nothing in this tree type-checks them.

`tests/test_mypy_none.py` locks the repo-side rows of this table (dependencies, `[tool.mypy]`, config files, pre-commit, CI). It does not import mypy and does not download models.

## Weights and secrets

Confirming this inventory does not need Whisper checkpoints, API keys, or credentials.
Do not commit `.pt` / `.pth` files or `.env` secrets.

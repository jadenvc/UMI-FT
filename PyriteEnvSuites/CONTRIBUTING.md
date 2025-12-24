# Contributing to PyriteEnvSuites

## Code Formatting and Linting

This project uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting. Local hooks and CI are pinned to the same Ruff version defined in `.pre-commit-config.yaml` (currently `v0.8.4`) to keep results consistent.

### First-Time Setup

```bash
pip install pre-commit
pre-commit install
```

Before your first commit, run the full suite once so fixes get recorded:

```bash
pre-commit run --all-files
```

### What Happens on Commit

`pre-commit` runs `ruff check --fix` followed by `ruff format`. The commit is aborted if Ruff changes files or reports errors; re-stage the changes and rerun `git commit` after the hooks complete cleanly.

### Manual Checks

```bash
ruff format .            # apply formatting
ruff format --check .    # verify formatting only
ruff check . --fix       # lint and autofix where possible
ruff check .             # lint without autofix
```

### Continuous Integration

GitHub Actions runs the same Ruff version used by `pre-commit` (`0.8.4`) with `ruff check` and `ruff format --check`. Keep your local hooks up to date before pushing to avoid CI failures.

### Configuration

Ruff is configured in `pyproject.toml`:
- Line length: 88 characters (Black default)
- Target Python: 3.8+
- Lint rules: pycodestyle (E, W), Pyflakes (F), isort (I); ignores `E501`
- Formatter: double quotes, spaces for indentation, auto-detected line endings

### Troubleshooting

- Hooks not running? Reinstall with `pre-commit install`.
- Need to skip hooks temporarily? Use `git commit --no-verify` (not recommended).
- Bumping Ruff? Run `pre-commit autoupdate` and commit the updated hook revision.

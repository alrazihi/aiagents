# ADR-007: pyproject.toml without BOM

## Status

Implemented

## Context

The `pyproject.toml` file had a UTF-8 BOM (`\xef\xbb\xbf`) at the
beginning. Python's `tomllib` module (used by pytest to read
`[tool.pytest.ini_options]`) does not handle BOMs and raised
`TOMLDecodeError: Invalid statement (at line 1, column 1)`. This caused
pytest to exit with code 4 (usage error) whenever run from the project
directory, completely breaking the CI pipeline.

## Decision

1. Rewrite `pyproject.toml` without a BOM, using `utf-8` encoding
   without `utf-8-sig`.
2. Fix the build backend from `setuptools.backends.legacy:build` to the
   standard `setuptools.build_meta`.
3. Add `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and
   `asyncio_mode = "auto"` to configure pytest declaratively.
4. Add `[tool.ruff]` and `[tool.mypy]` sections for linting and type
   checking configuration.
5. Add `[tool.setuptools.packages.find]` to declare the package
   explicitly.

## Consequences

- `pytest` runs correctly from the project root.
- `pip install -e .` works with the standard build backend.
- Linting and type checking are configured declaratively in
  `pyproject.toml`, not in separate config files.

## Evidence

- `pyproject.toml` — no BOM, standard build backend
- `.github/workflows/ci.yml` — CI runs ruff, mypy, pytest
- `dev-requirements.txt` — includes `google-generativeai` for test imports

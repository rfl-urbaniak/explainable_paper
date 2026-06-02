#!/bin/bash
# Static checks for Python sources and notebooks. Non-mutating: fails if
# anything is unformatted or has lint/type errors. Run `make format` to fix.
set -euxo pipefail

PY_SRC="pci tests"
NB_SRC="docs/source"

uv run --group dev mypy $PY_SRC
uv run --group dev ruff check $PY_SRC
uv run --group dev ruff format --diff $PY_SRC

uv run --group dev nbqa mypy $NB_SRC
uv run --group dev nbqa 'ruff check' $NB_SRC
uv run --group dev nbqa 'ruff format --diff' $NB_SRC

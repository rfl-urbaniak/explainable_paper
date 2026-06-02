#!/bin/bash
# Auto-fix lint issues and reformat Python sources and notebooks in place.
# Unused imports (F401) are left alone here; run `make remove-imports` for those.
set -euxo pipefail

PY_SRC="pci tests"
NB_SRC="docs/source"

uv run --group dev ruff check --fix-only $PY_SRC
uv run --group dev ruff format $PY_SRC

uv run --group dev nbqa 'ruff check --fix-only' $NB_SRC
uv run --group dev nbqa 'ruff format' $NB_SRC

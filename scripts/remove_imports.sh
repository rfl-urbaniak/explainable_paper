#!/bin/bash
# One-off pass to strip unused imports (F401), which `make format` deliberately
# leaves in place (notebooks often import for display side effects).
set -euxo pipefail

PY_SRC="pci tests"

uv run --group dev ruff check --select F401 --fix --extend-fixable F401 $PY_SRC

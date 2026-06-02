#!/bin/bash
# Run the test suite with coverage. Extra args are forwarded to pytest,
# e.g. `./scripts/test.sh tests/test_scores.py -k rsample`.
set -euo pipefail

uv run --group dev pytest -s --cov --cov-report=term-missing --cov-report=html "${@-}"

TOTAL=$(uv run --group dev coverage report | tail -1 | awk '{print $NF}')
HTML="file://$(realpath tests/coverage/index.html)"

echo
echo "================================ Coverage ================================"
echo "Overall coverage: ${TOTAL}"
echo "HTML report:      ${HTML}"
echo "=========================================================================="

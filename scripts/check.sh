#!/usr/bin/env bash
# The gates. Every sprint must pass all of these before it is committed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"

echo "== lint";        "$PY" -m ruff check src tests scripts
echo "== format";      "$PY" -m ruff format --check src tests scripts
echo "== tests";       "$PY" -m pytest
echo "== traceability"; "$PY" scripts/trace.py
echo "All gates passed."

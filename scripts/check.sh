#!/usr/bin/env bash
# The gates. Every sprint must pass all of these before it is committed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
RUST=(--manifest-path rust/Cargo.toml)
# the Python the Rust bindings are built against (an absolute path: cargo changes directory)
PYO3_PYTHON="$(command -v "$PY")"
[[ "$PYO3_PYTHON" == /* ]] || PYO3_PYTHON="$PWD/$PYO3_PYTHON"
export PYO3_PYTHON

echo "== rust format"; cargo fmt "${RUST[@]}" --all --check
echo "== rust lint";   cargo clippy "${RUST[@]}" --workspace --all-targets --quiet -- -D warnings
echo "== rust tests";  cargo test "${RUST[@]}" -p lastfirstdate-core --quiet
echo "== rust build";  "$PY" -m maturin develop --release --skip-install --quiet  # lastfirstdate._core
echo "== lint";        "$PY" -m ruff check src tests scripts
echo "== format";      "$PY" -m ruff format --check src tests scripts
echo "== tests";       "$PY" -m pytest
echo "== traceability"; "$PY" scripts/trace.py
echo "All gates passed."

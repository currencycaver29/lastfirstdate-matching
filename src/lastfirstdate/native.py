"""The Rust core: when the engine uses it (D16, FR-LM-35).

The engine's hot loops have two versions with the same answers: the readable NumPy one in
each module (the reference, and what the sprint notes explain), and a Rust one in
`rust/lastfirstdate-core`, compiled into `lastfirstdate._core` when the package is installed.
The engine uses Rust whenever the compiled module is present.

- Set the environment variable `LASTFIRSTDATE_BACKEND=python` to run everything in Python
  (for studying, debugging or comparing).
- `with python_only():` does the same inside a block; the tests use it to check that both
  versions agree.

Rust is used for: the logistic fit behind the looks model, the forecast and learning
(`looks.fit_logistic`); the uncertainty of many candidates and the pair search in
calibration (`PersonalTaste.uncertainty`, `looks.select_pair`); the fair daily lists
(`fairness.fair_day_lists`); and the ratio gate (`forecast.gate_decision`).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

try:
    from lastfirstdate import _core as core
except ImportError:  # not compiled: everything runs in Python
    core: ModuleType | None = None

BACKENDS = ("rust", "python")
_requested = os.environ.get("LASTFIRSTDATE_BACKEND", "rust").strip().lower()
if _requested not in BACKENDS:
    raise ValueError(f"LASTFIRSTDATE_BACKEND must be one of {BACKENDS}, not {_requested!r}")
_forced_python = _requested == "python"


def available() -> bool:
    """Is the compiled Rust core installed?"""
    return core is not None


def enabled() -> bool:
    """Should the engine call Rust right now?"""
    return core is not None and not _forced_python


def backend() -> str:
    """'rust' or 'python': which version the engine is using right now."""
    return "rust" if enabled() else "python"


@contextmanager
def python_only() -> Iterator[None]:
    """Run the block with the Python versions only."""
    global _forced_python
    before = _forced_python
    _forced_python = True
    try:
        yield
    finally:
        _forced_python = before

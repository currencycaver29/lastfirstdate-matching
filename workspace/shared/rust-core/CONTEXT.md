# Shared: Rust core

The engine's hot loops in Rust (D16, FR-LM-35), called from Python. Each Rust function has a
Python twin with the same answers.

## Layout

- `rust/lastfirstdate-core/`: pure Rust, with no Python and no `unsafe` code; 19 tests of its
  own.
  - `logistic`: the Newton fit with step halving, Cholesky, and the inverse from the triangle;
  - `calibration`: symmetric quadratic forms and the pair search;
  - `lists`: round-robin lists under the exposure cap;
  - `gate`: an incremental ratio gate;
  - `windows`: free date windows, the first shared window, and blocked days;
  - `linalg`: small dense linear algebra.
- `rust/lastfirstdate-py/`: PyO3 bindings, imported as `lastfirstdate._core`.
- `src/lastfirstdate/native.py`: decides when Python calls Rust. `LASTFIRSTDATE_BACKEND=python`
  forces Python; `native.python_only()` does it inside a block.

## Which functions are twins

| Python | Rust |
|---|---|
| `looks.fit_logistic` (`_fit_logistic_python`) | `_core.fit_logistic` |
| `PersonalTaste.uncertainty` | `_core.quad_forms` |
| `looks.select_pair` | `_core.best_pair` |
| `fairness.fair_day_lists` | `_core.fair_day_lists` |
| `forecast.gate_decision` | `_core.gate_scan` |
| `booking.window_bits` | `_core.window_bits` |
| `booking.earliest_common` (one against many, or pair by pair) | `_core.earliest_common` |
| `app._blocked_days` | `_core.blocked_days` (via `app._blocked_days_rust`) |

## Commands

```bash
.venv/bin/maturin develop --release --skip-install   # rebuild after editing rust/
cargo test --manifest-path rust/Cargo.toml -p lastfirstdate-core
```

`./scripts/check.sh` runs cargo fmt, Clippy (warnings are errors), the Rust tests and a fresh
build.

## What we know

- Looks tastes run 3.6× faster, daily lists 10×, and the gate 28× to 81×. Scores and bookings
  are identical to the pure-Python engine.
- AVX2 instructions gained too little to justify CPU-specific code.
- Writing the twins found a shared weakness: plain Newton could diverge. Both versions now
  halve bad steps.
- Round 2 (v0.13.1), the date windows: per call 2× to 75× faster; the app's daily lists 1.6×
  to 2× faster end to end, test runs about 1.3×. Timings on this shared machine vary by about
  20%, so compare medians of several runs, alternating the two versions.

## If you change a twin

Change both versions. `tests/test_native.py` must still show they agree on random problems,
and a synthetic month must book the same dates.

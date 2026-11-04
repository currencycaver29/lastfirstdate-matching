# Stage 9: guarantee

Protects "3 guaranteed dates": forecasts each man's chance, warns mid-cycle, decides how many
men to admit, and counts what was delivered and what is owed.

## Takes in / passes on

- **In:** eligibility, women and paying men in the zone, the waitlist; bookings and their
  outcomes; each man's signup day.
- **Out:** each man's chance of reaching 3 dates; at-risk flags; gate decisions (admit, defer,
  stop and why); delivered dates, missing dates, credits and fee refunds per cycle.

## Forecast and gate (Sprint 6)

    P_j = sigmoid(b0 + b1·log r + b2·log(1 + pool_j) + b3·log(1 + share_j) + b4·log(min(r, 1)))

- **The inputs:** r is women per man; pool_j is the women he could meet; share_j is his
  expected share of their attention.
- **Fitting:** `forecast_fit.py` fits it on simulated markets and writes
  `config/forecast_model.toml`, which records the range it was fitted on.
- **The gate:** admits waitlisted men in order while women per man ≥ the floor (3), inside the
  fitted range, and the forecast ≥ the target (90%) (`forecast.gate_decision`, with a Rust
  twin).

## Billing (Sprint 7)

- **Cycles:** a 30-day free trial, then 28-day cycles per man (`billing.cycle_on`).
- **Delivered:** confirmed, paid, inside his cycle, and it happened. His own late
  cancellation or no-show counts (D17); hers is refilled.
- **Credit:** $200 × missing ÷ 3. The $10 fee is refunded by timing.

## Code

`src/lastfirstdate/forecast.py`, `forecast_fit.py`, `billing.py`; `phase1_report.py` runs the
gate trial.

## Settings

`[gate]`, `[billing]`, `config/forecast_model.toml`.

## Tests

FR-LM-33, FR-LM-34 (`tests/test_phase1.py`), FR-LM-38, FR-LM-42 (`tests/test_sprint7.py`).

## What we know

- The forecast's held-out error is 3.3 points. A first version, fitted only on 1 to 5 women
  per man, said 83% where 7% actually got 3 dates. Hence the shortfall term and the fitted
  range.
- The gate keeps admitted men at 92% to 96% of *booked* dates; admitting everyone gives 12.5%.
- **The real flow delivers only about 68%** (Sprint 7), but the forecast and gate were fitted on
  booked dates. Refitting them on delivered dates is the proposed Sprint 8.
- Credits cost about 17% of subscription revenue in the real flow.

## If you change it

Refit with `python -m lastfirstdate.forecast_fit` (about 10 minutes, in the background) and
rerun `python -m lastfirstdate.phase1_report --skip-calibration`.

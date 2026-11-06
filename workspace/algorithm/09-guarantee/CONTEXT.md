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
- **Fitting:** `forecast_fit.py` fits it on synthetic markets and writes
  `config/forecast_model.toml`, which records the range it was fitted on.
- **The gate:** admits waitlisted men in order while women per man ≥ the floor (3), inside the
  fitted range, and the forecast ≥ the target (90%) (`forecast.gate_decision`, with a Rust
  twin).

## On delivered dates (Sprint 8)

- **Forecast:** `forecast_fit.py --delivered` fits the same five start features on the real
  booking flow (`cycle_runs.py`). Each paying man's market at launch is labelled with the share
  of his paid cycles that reached 3 delivered dates. It is saved to
  `config/forecast_delivered.toml` with its fitted range and the frictions and rate of new
  women it assumes.
- **Mid-cycle alerts:** the same start features plus delivered so far ÷ 3, log(1 + confirmed
  dates to come) and days left ÷ cycle days (`delivered_progress_matrix`,
  `at_risk_delivered`).
- **Which forecast the gate uses:** `[gate] forecast = "booked" | "delivered"` picks the file
  (`forecast.gate_model`). Phase 1 uses `delivered`; the gate's code and its Rust twin are
  unchanged.

## On real cycles (Sprint 11)

`refit.cycle_rows` rebuilds every completed paid cycle and its weekly progress from the app's
records. `refit.propose_refit` recalibrates the forecast (logit' = a + b · logit, folded into
the coefficients) and recommends switching only after testing on the latest cycles. Settings
are in `[refit]`. Fulfilment drifts as a zone ages, so it should run monthly.

## Billing (Sprint 7)

- **Cycles:** a 30-day free trial, then 28-day cycles per man (`billing.cycle_on`).
- **Delivered:** confirmed, paid, inside his cycle, and it happened. His own late
  cancellation or no-show counts (D17); hers is refilled.
- **Credit:** $200 × missing ÷ 3. The $10 fee is refunded by timing.

## Code

`src/lastfirstdate/forecast.py`, `forecast_fit.py`, `billing.py`. The gate trial in the batch
flow is in `phase1_report.py`; in the real flow, in `gate_report.py`.

## Settings

`[gate]` (including `forecast`), `[billing]`, `config/forecast_model.toml` (booked),
`config/forecast_delivered.toml` (delivered).

## Tests

FR-LM-33, FR-LM-34 (`tests/test_phase1.py`), FR-LM-38, FR-LM-42 (`tests/test_sprint7.py`),
FR-LM-43 to FR-LM-45 (`tests/test_sprint8.py`).

## What we know

- The forecast's held-out error is 3.3 points. A first version, fitted only on 1 to 5 women
  per man, said 83% where 7% actually got 3 dates. Hence the shortfall term and the fitted
  range.
- The gate keeps admitted men at 92% to 96% of *booked* dates; admitting everyone gives 12.5%.
- **The real flow delivers about 82%** at 3:1 in a 300-woman zone (Sprint 7, corrected in
  Sprint 8). The delivered forecast's held-out error is 3.8 points; its alerts reach AUC 0.97
  to 1.00.
- **The zone's size matters far more than the ratio.** With 500 women, about 90% even at 3:1;
  with 300, the delivered gate admits only a handful of men at a 90% target.
- **In large zones the gate is sensitive:** fulfilment is flat in the number of men, so 2 to 3
  points of forecast error move the stopping point by about 100 men.
- Credits run about 6% of revenue in a 500-woman zone and 15% in a 300-woman zone at 3:1
  (NYC demo, Sprint 14).
- **Fake women (Sprint 14, D22, FR-LM-66):** 50 fakes in a 300-woman zone made the gate sell 6.7
  places instead of 3, and those men's 3-date share fell from 94% to 83%. `app.counted_women`
  now counts only verified women who aren't silent (`gate.silent_after_days` list days and no
  pick ever): after four weeks it drops 83% of fakes and 0.6% of honest women. At launch only
  identity checks help.

## If you change it

Refit with `python -m lastfirstdate.forecast_fit` (about 10 minutes, in the background) and
rerun `python -m lastfirstdate.phase1_report --skip-calibration`.

# Stage 05: handoff to the app

The owner builds the algorithm; someone else builds the app. This stage writes down what the
app developer needs to plug the engine in. Run it when an interface the app uses changes, and
before any app integration.

## Read

- `db/schema.sql`: what the app stores.
- The public functions of the engine modules: `booking.py`, `billing.py`, `trust.py`,
  `geo.py`, `forecast.py` (the gate), `fairness.py` and `lists.py` (daily lists),
  `looks.py` (calibration), `mutual.py` (the 50/50 score).
- `config/phase1.toml`: the settings the app's operators may change.

## Produce

`docs/handoff/` (create it the first time) with:

- **when to call what:**
  - at sign-up, calibration;
  - every night, scores and daily lists;
  - when a woman opens the app, date windows;
  - when she picks, the booking and payment hold;
  - after a date, ratings and trust;
  - at cycle end, credits;
  - before every men's sale, the gate;
- **the data each call needs and returns**, mapped to the tables in `db/schema.sql`;
- **settings** the app may change and which ones need a fresh simulation first;
- **assumptions** the app must keep (rules R1 to R7, the 15-minute hold, D17), and the open
  legal questions in `docs/risk-plan.md`.

## Done when

An app developer could wire the engine in from `docs/handoff/` alone, without reading the
engine's code.

## Status

Done in Sprint 9: `docs/handoff/README.md` and `src/lastfirstdate/app.py`. When an interface
changes, update both. `tests/test_sprint9.py` checks that every call the handoff names exists
and that its settings table matches `config.settings_table()`.

# Stage 6: daily lists

Builds each chooser's daily list of up to 30 profiles, sharing attention fairly.

## Takes in / passes on

- **In:** `M` (stage 5), today's availability (eligible, not shown in the last 7 days, not
  full, not closed), and each candidate's guarantee bonus and trust factor (stages 9 and 10).
- **Out:** today's list for each chooser.

## How lists are built

- **Who chooses:** women pick and men accept (D1, `lists.pick_rights`).
- **The "fair" policy (default):** round-robin under an exposure cap. Everyone's first slot,
  then everyone's second, and so on; nobody appears on more than

      cap = ceil(exposure_multiple × list slots today / candidates today)

  lists a day (`fairness.fair_day_lists`, with a Rust twin).
- **The guarantee bonus:** priority × missing dates ÷ weeks left in his own cycle
  (`billing.refill_bonus`).
- **Alternatives:** "nsw", a greedy Nash social welfare fill, and "basic", each chooser's
  own top 30. A hold-back rule exists but is off.
- **Exploration:** when the learning loop supplies uncertainty (stage 8), 10% of slots go to
  profiles the model is unsure about.

## Code

`src/lastfirstdate/lists.py`, `src/lastfirstdate/fairness.py`.

## Settings

`[lists]`: `daily_size`, `policy`, `reshow_after_days`, `exposure_multiple`,
`guarantee_priority`, `exploration_share`, `holdback_yeses`, `hide_booked`, `nsw_epsilon`;
`[fairness]`.

## Tests

FR-LM-09, FR-LM-21 to FR-LM-24 (`tests/test_lists.py`, `tests/test_fairness.py`,
`tests/test_month.py`), FR-LM-35 (Rust twin).

## Papers

- 12, Singh and Joachims: fairness of exposure.
- 11, Tomita and Yokoyama: Nash social welfare.
- 13, Su et al.: rankings in matching markets.
- 14, Rios et al.: holding back likely matches.
- 15 and 16: short-side choosing and congestion.

## Decisions

D1, D3 (30 a day; a passed profile can return after 7 days), D7 (exposure cap 1×).

## What we know

- The exposure Gini falls from 0.39 to 0.31; the guarantee rose from about 78% to 82% in the
  research city.
- Hiding members whose month is full, and capping free members, both made things worse, so
  both are off.
- Repeated showings wear a closed market out: each pass makes a later pick less likely.
  Showing men less often did not help in the cycle simulation.

## If you change it

Keep the Rust and Python lists identical (`tests/test_native.py`). Rerun
`python -m lastfirstdate.fairness_report` and `python -m lastfirstdate.cycle_sim`.

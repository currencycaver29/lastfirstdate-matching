# Sprint 9 results: the engine on the app's database

A Phase 1 zone (seed 7: 100 men, 300 women; 60 men subscribed, the rest on the waitlist) is
written into SQLite by `app_sim.fill_database`, then loaded back by `app.py` as the app would.
Commands: `python -m pytest tests/test_sprint9.py -q`, and the timing snippet in
`docs/sprints/sprint-9.md`.

## Equivalence (FR-LM-47)

| Check | Result |
|---|---|
| Profiles from rows against the originals | identical, except locations are neighbourhood centres and only subscribers count as paying (by design) |
| Eligibility and personality scores from rows | identical to computing them directly on the same profiles at their neighbourhood centres |
| Taste models refitted from stored answers | identical to the photo exercise's (largest difference 7 × 10⁻¹⁶) |
| Gate decision from rows | identical admissions and reason to `gate_decision` on arrays |
| Lists from rows | every profile eligible, admitted and with a window both can make |
| A booking written back | passes the schema's checks; next time her week is blocked and the pair is closed |

## Speed of each call (Phase 1 zone, one laptop core)

| Call | Time |
|---|---:|
| `load_market` (400 members) | 26 ms |
| `photo_feature_matrix` | 14 ms |
| `calibration_answers` | 36 ms |
| `fit_tastes` (383 taste models, starting taste included) | 482 ms |
| `nightly_scores` | 236 ms |
| `day_lists`, all 300 women in one call | 158 ms |
| `weekly_alerts` | 2 ms |
| `gate_for_sale` | 2 ms |

The whole nightly job takes under a second. All 286 women with anyone to see got a list,
21.8 profiles on average.

## Found while measuring

- **Fair sharing needs every list built together.** The exposure cap divides attention
  across all of a day's lists, so `day_lists` must be called once a day for every chooser, not
  once per woman as she opens the app. The handoff says so.
- **Signup days can be before day 0.** Using −1 to mean "no subscription" hid every member
  who signed up before the calendar's epoch; a proper sentinel fixed it, caught by the
  smoke test.

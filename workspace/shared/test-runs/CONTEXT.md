# Shared: test run

Fake members and fake behaviour, used to test and measure the engine. Test runs only: the
engine never imports these modules or reads `GroundTruth` (FR-LM-03, FR-LM-11, enforced by
`tests/test_engine_boundary.py`).

## Pieces

- **Synthetic population:** `synthetic.generate(cfg, seed)` makes profiles, photo embeddings
  and hidden truth (`GroundTruth`: true taste, looks weight, pickiness), from the
  `[population]` settings.
- **Synthetic people:** `member_behavior.Behavior` makes their choices (choose, accept, show up,
  rate) from the hidden truth only.
- **A month, batch flow:** `month_runs.run_month` runs Sprints 1 to 6's flow (daily lists, picks,
  weekly batch scheduling) and returns `MonthReport`.
- **Billing cycles, real flow:** `cycle_runs.CycleRun` runs Sprint 7's flow day by day
  (windows, payment holds, cancellations, no-shows, events, ratings, trust, new women
  arriving). `cycle_runs.compare` runs every scenario over several markets.
- **Reports:** `looks_report`, `mutual_report` (`build_scores`), `fairness_report`,
  `learning_report`, `phase1_report`, `forecast_fit`, `cycle_runs`, `nyc_report`, `attacks`.
  Each is a `python -m lastfirstdate.<name>`.
- **Attackers (Sprint 14):** `attacks.py` adds stuffed prompts, answer floods, fake accounts,
  fake women on the gate, harsh raters, rating rings and spies. `attacks.AttackedRun` plugs them
  into `CycleRun` through its hooks (`opens`, `choose`, `accepts`, `rate`).

## How to measure honestly

- **Seeds:** the test city is seed 7; settings are tuned on seed 101 (D11).
- **Common random numbers:** every version sees the same people and the same randomness.
- **Replicates:** several seeds, reported with a standard error. Single runs of the cycle
  test run vary by about ±3 points.
- **Unmet targets** become strict expected failures, never weaker tests.
- **Earlier results must not move:** `test_default_population_is_unchanged_by_sprint_6`
  guards the default fake population.

## Settings

`[test_runs]` (behaviour: open rate, thresholds, show-up rate, ratings), `[cycle_runs]`
(holds, cancellations, events, new women, bad actors, ratings), `[population]`.

## What we know (limits)

- The fake people's yes rates and show-ups decide every number. They compare designs; they
  are not predictions.
- Women's commitment and the $10 fee don't change the fake people's behaviour.
- Fake women pick on true appeal with no position bias, so a better place on a list is worth
  less here than to real people (Sprint 14).

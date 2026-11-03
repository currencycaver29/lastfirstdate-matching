# lastfirstdate-matching

Matching engine for lastfirstdate. After a member swipes on about 50 profiles, it learns
their taste and ranks matches 50% on looks (to that member) and 50% on personality and
interests. Scores are mutual, daily lists are spread fairly, and dates are scheduled into
time both people have free.

Everything here runs on **synthetic members only**. No real profiles, photos or embeddings
belong in this repo.

## Status

| Sprint | What | Status |
|---|---|---|
| 0 | Foundation: settings, member schema, fake NYC member generator, gates | Done |
| 1 | Working matcher on personality only: filter, personality score, daily list, weekly scheduler, simulated month | Done (about 80% of paying members reach 3 dates) |
| 2 | Looks model: the 50 calibration profiles and personal photo taste | Next |
| 3 | 50/50 mutual score | Planned |
| 4 | Fair daily lists | Planned |
| 5 | Learning from dates | Planned |

Plan and requirements: [`docs/SPEC.md`](docs/SPEC.md). Papers:
[`docs/research.md`](docs/research.md). Study notes per sprint:
[`docs/sprints/`](docs/sprints/).

## Run it

Needs Python 3.11 or newer.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"      # on Windows: .venv\Scripts\pip install -e ".[dev]"
.venv/bin/python -m lastfirstdate.synthetic
.venv/bin/python -m lastfirstdate.simulate   # one simulated month and its report
./scripts/check.sh                     # lint, format, tests, traceability
```

## Layout

```
config/lastfirstdate.toml   every weight, cap and threshold
src/lastfirstdate/
  config.py                 loads and validates the settings
  models.py                 the member profile
  synthetic.py              fake NYC members plus hidden ground truth for testing
  arrays.py                 profiles as numpy arrays for fast pairwise work
  filter.py                 Stage 1: hard filter
  personality.py            Stage 4: interests, values and Big Five score
  lists.py                  Stage 6: who chooses and the daily list
  scheduler.py              Stage 7: weekly date booking (min-cost flow or integer program)
  simulator.py              fake people's decisions, from hidden truth only
  simulate.py               a simulated month and its report
tests/                      one or more tests per requirement id
scripts/check.sh            the gates
scripts/trace.py            fails if a requirement has no test
docs/                       spec, research map, sprint study notes
db/                         SQL queries (the original pre-filter goes here)
```

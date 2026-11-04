# Stage 8: learning

Improves the engine from what actually happens: picks, dates, ratings.

## Takes in / passes on

- **In:** recorded list choices and `DateRecord`s (who met, who showed up, "would meet
  again", ratings).
- **Out:** updated looks models, each member's personal looks weight, uncertainty for
  exploration.

## What it does

- **Personal looks weight:** each member's balance between looks and personality, learned
  from their picks (`learning.looks_weights`, `personal_directional_score`).
- **Date outcomes into the looks model:** with a weight (`update_tastes`). This did not help,
  and a heavy weight made it worse.
- **Exploration:** a share of list slots goes to profiles the model is unsure about
  (`uncertainty_matrix`, `exploration_slots`, `explore_picks`).
- **Never trust:** conversation ratings and "would meet again" belong here, never in trust
  (stage 10).

## Code

`src/lastfirstdate/learning.py`; reports in `learning_report.py`.

## Settings

`[learning]`; `[lists] exploration_share`.

## Tests

FR-LM-25 to FR-LM-28 (`tests/test_learning.py`).

## Papers

17, Liu et al. (competing bandits: explore a little); 9, Joel et al.

## What we know

- Personal looks weights are learned well (correlation +0.3 to +0.4 with the hidden truth).
- But second dates didn't measurably improve in a 1,000-member city over 3 months, and three
  claims are recorded as strict expected failures.
- Second dates fall about 10 points from month 1 to month 3 as the best pairs are used up.

## If you change it

Run `python -m lastfirstdate.learning_report` with replicates; single runs are too noisy here.

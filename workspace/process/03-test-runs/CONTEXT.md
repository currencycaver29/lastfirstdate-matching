# Stage 03: test runs

Measure what the sprint changed, on fake members, honestly.

## Read

- The new sprint section in `docs/SPEC.md` (what each requirement claims).
- The report module the sprint touched (for example `src/lastfirstdate/cycle_runs.py`).
- The previous results file in this folder, to compare against.

## How to measure

- **Several markets:** use seeds 7, 8 and 9 and report the mean ± one standard error. One run
  is too noisy to judge a difference of a few points.
- **Paired comparisons:** run every version on the same markets (common random numbers).
- **Tune on seed 101, report on seed 7** when a setting is being chosen, so tuning can't
  flatter the result.
- **Long commands** (over a minute) run in the background; save their output.

## Produce

`workspace/process/03-test-runs/results-sprint-N.md` containing:

- each command that produced a number, so anyone can rerun it;
- the tables (versions as rows, mean ± standard error);
- surprises and anything that contradicts the spec. A missed acceptance target goes back to
  stage 02 as a strict expected failure, never as a weaker test.

## Done when

Every number in the results file can be reproduced by its command.

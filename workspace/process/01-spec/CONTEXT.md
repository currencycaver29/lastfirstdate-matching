# Stage 01: spec

Turn the sprint's goal into numbered requirements the owner agrees to, before any code.

## Read

- `workspace/CONTEXT.md` (current state, open questions)
- `docs/SPEC.md`: the Rules, the Decisions table, the previous sprint's section and the last
  entry under "Changes to this plan". Not the whole file.
- The "Questions for you" section at the end of the previous sprint's notes
  (`docs/sprints/sprint-7.md` for Sprint 8).
- For business-facing sprints: `docs/fit-report.md` section 8 and `docs/risk-plan.md` section 1.

## Produce

- A new `### Sprint N: ...` section in `docs/SPEC.md`, before "## Changes to this plan", with
  one `#### FR-LM-NN · Title` block per requirement: a `Sprint: N` line, what it does in plain
  words, and a `Test:` line saying what will prove it.
- `Current sprint: N` at the top of `docs/SPEC.md`.
- New decisions, if any, as rows in the Decisions table (D18 onward).

Numbering continues from FR-LM-43.

## Done when

The owner has agreed to the scope in chat. (`scripts/trace.py` will fail until stage 02 adds a
test for each new requirement; that is expected.)

## Sprint 8 (proposed, confirm with the owner)

**Goal:** the forecast and the ratio gate (Sprint 6) assume about 90% fulfilment from the old
batch simulation, but the real booking flow (Sprint 7) delivers about 68%. Refit them on the
real flow so the gate stops selling subscriptions the market can't deliver.

Candidate requirements:

- **FR-LM-43 · Forecast on delivered dates:** fit the start-of-cycle forecast on rows from
  `cycle_sim.py` (target: 3 delivered dates in his cycle), across market sizes, women per man
  and new-women rates.
- **FR-LM-44 · Mid-cycle alerts on delivered dates:** the at-risk model with delivered dates
  so far, confirmed dates still to come, and days left in his own cycle.
- **FR-LM-45 · Gate on the real flow:** the gate uses the delivered-date forecast. In the
  cycle simulation, admitted men reach 3 delivered dates near `gate.target`, or the shortfall
  is recorded as a finding.

Files the sprint will touch: `src/lastfirstdate/forecast.py`, `forecast_fit.py`,
`cycle_sim.py`, `config/forecast_model.toml`, `config/phase1.toml` (`[gate]`), and the FR-LM-33
and FR-LM-34 tests in `tests/test_phase1.py`.

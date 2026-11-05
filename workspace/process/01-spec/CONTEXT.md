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
- New decisions, if any, as rows in the Decisions table (D20 onward).

Numbering continues from FR-LM-59.

## Done when

The owner has agreed to the scope in chat. (`scripts/trace.py` will fail until stage 02 adds a
test for each new requirement; that is expected.)

## Sprint 13 (not chosen yet: ask the owner)

Sprint 12 fixed the bugs a real app would hit on day one. Options:

- **(a) The real photo and text models** (issue #13), once the owner picks them: a
  `PhotoEncoder` for a CLIP-style model and a Sentence-BERT `TextEncoder`, with offline
  stand-ins kept for the tests (R4), and the settings re-checked.
- **(b) The gate every cycle,** with the zone's age and recruitment as inputs.
- **(c) Readiness for the personality weights:** refit them on real "would meet again"
  answers (paper 9).
- **(d) Stop until the app and real data exist,** then run `refit.propose_refit` monthly.

Recommended: (a) as soon as the models are chosen; until then, (d).

Numbering continues from FR-LM-59.

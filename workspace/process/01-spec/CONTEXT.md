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

Numbering continues from FR-LM-51.

## Done when

The owner has agreed to the scope in chat. (`scripts/trace.py` will fail until stage 02 adds a
test for each new requirement; that is expected.)

## Sprint 11 (not chosen yet: ask the owner)

Options:

- **(a) The gate every cycle:** re-run it with real data as men leave and women join, with the
  recruitment rate as an input.
- **(b) Wait for real data:** once Phase 1 runs, refit the forecast, the personality weights and
  the looks prior on real cycles.
- **(c) Whatever the app developer asks for** after reading `docs/handoff/README.md`.

Numbering continues from FR-LM-51.

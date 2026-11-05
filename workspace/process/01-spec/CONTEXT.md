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
- New decisions, if any, as rows in the Decisions table (D21 onward).

Numbering continues from FR-LM-63.

## Done when

The owner has agreed to the scope in chat. (`scripts/trace.py` will fail until stage 02 adds a
test for each new requirement; that is expected.)

## Sprint 14 (not chosen yet: ask the owner)

Sprint 13 plugged in the real models. Options:

- **(a) A consented pilot:** 30 to 100 volunteers do the photo exercise; recheck
  `calibration.pca_dim` and the penalties on SigLIP's real vectors (no data enters the repo).
- **(b) The gate every cycle,** with the zone's age and recruitment as inputs.
- **(c) Readiness for the personality weights:** refit them on real "would meet again"
  answers (paper 9).
- **(d) Stop until the app and real data exist,** then run `refit.propose_refit` monthly.
- **(e) Abuse testing (AI red team):** add synthetic attackers to the test runs and measure what
  each gains: keyword-stuffed prompts, fake accounts inflating the ratio gate, coordinated
  rating rings against the trust score, fake accounts poisoning the starting taste, attempts
  to read a hidden "desirability" score from lists (R2), and malformed rows (code review
  finding 2). Every attack that gains anything gets a rule, a fix and a test. Reading list:
  `docs/research.md`, "Reading list: abuse testing". Complements, not replaces, the outside
  pentest of the app before the beta.

Numbering continues from FR-LM-63.

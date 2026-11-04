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

Numbering continues from FR-LM-46.

## Done when

The owner has agreed to the scope in chat. (`scripts/trace.py` will fail until stage 02 adds a
test for each new requirement; that is expected.)

## Sprint 9 (proposed, confirm with the owner)

**Goal:** the owner builds the algorithm; someone else builds the app. Write down exactly how the
app plugs the engine in, so app development can start. This is process stage 05, made a sprint.

Candidate requirements (numbering continues from FR-LM-46):

- **FR-LM-46 · Call sequence:**
  - at sign-up, calibration;
  - every night, scores and lists;
  - when a woman opens the app, windows;
  - when she picks, the booking and hold;
  - after a date, ratings and trust;
  - every week, alerts;
  - at cycle end, credits;
  - before each men's sale, the gate.
- **FR-LM-47 · Data contract:** each call's inputs and outputs, mapped to `db/schema.sql`, with
  a test that a full simulated day can run from database rows alone.
- **FR-LM-48 · Operator settings:** which settings the app's operators may change, and which
  need a refit or a fresh simulation first.

Alternatives the owner may prefer: re-running the gate every cycle with real data, or a service
the app can call.

Files the sprint will touch: `docs/handoff/` (new), `db/schema.sql`, the public functions listed
in `workspace/process/05-handoff/CONTEXT.md`, and a new test file.

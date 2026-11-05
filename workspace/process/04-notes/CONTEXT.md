# Stage 04: notes and review

Explain the sprint so the owner can study it, then hand over for review.

## Read

- The sprint section in `docs/SPEC.md`.
- `workspace/process/03-simulate/results-sprint-N.md`.
- `docs/sprints/sprint-7.md` as the model for structure and tone (skim; don't copy).

## Produce

1. `docs/sprints/sprint-N.md`, in plain words:
   - a one-paragraph summary;
   - the papers used and how;
   - what was built;
   - the maths, explained step by step;
   - the results, as tables;
   - what changed in the plan;
   - honest limits;
   - maths to study;
   - commands to try;
   - questions for the owner.
2. A "Changes to this plan" entry in `docs/SPEC.md`.
3. The sprint's row in the Status table of `README.md`.
4. `docs/fit-report.md` and `docs/risk-plan.md`, if business numbers or risks changed.
5. **The version,** in `pyproject.toml`, `src/lastfirstdate/__init__.py` and `rust/Cargo.toml`
   (all three the same): the minor number is the sprint (Sprint 12 is 0.12.0); the major
   number changes when the app must change how it calls the engine; the fix number is for bug
   fixes. Update the current release in the README's Releases section.
6. **This workspace, ready for the next sprint:**
   - the "What we know" sections of the algorithm stages the sprint changed
     (`workspace/algorithm/`);
   - `workspace/CONTEXT.md` (current sprint, state, open questions);
   - the "Sprint N+1" section of `workspace/process/01-spec/CONTEXT.md`.

   Stale context files mislead the next session.

## Then

Run `./scripts/check.sh`, commit locally, and stop. Explain the sprint in depth in chat, send
the notes file, and ask whether to push. Push only on the owner's word. After the push, tag the
commit `vX.Y.Z` and publish a GitHub release whose notes summarise the sprint's notes.

## Done when

The owner has the notes and the explanation.

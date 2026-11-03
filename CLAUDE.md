# lastfirstdate-matching: instructions for Claude

Matching engine for lastfirstdate. The plan, rules and requirements are in `docs/SPEC.md`;
the papers behind each requirement are in `docs/research.md`; study notes for each finished
sprint are in `docs/sprints/`.

## Working rules

- Build one sprint at a time, in the order in `docs/SPEC.md`. After a sprint passes the
  gates, stop and write `docs/sprints/sprint-N.md`: what was built, the math in plain words,
  which papers it uses and how, what the simulation showed, and what to study. The owner
  studies each sprint before approving the next.
- Push to GitHub only when the owner says so.
- Every test docstring names the requirement id it proves (for example `FR-LM-04`).
- Gates: `./scripts/check.sh` (ruff lint, ruff format, pytest, traceability). Never commit
  with a red gate, and never weaken a test to make it pass.
- Bump `Current sprint:` in `docs/SPEC.md` when a sprint starts.

## Hard rules (from docs/SPEC.md)

- R1 mutual orientation lock. R2 no global attractiveness score, anywhere. R3 synthetic data
  only: no real photos, profiles or embeddings in this repo. R4 tests run offline, no model
  downloads. R5 every tunable number in `config/lastfirstdate.toml`. R6 dealbreakers are hard
  filters. R7 all randomness seeded.
- The simulator may read `GroundTruth`; the engine must never read it.

## Commands

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"
./scripts/check.sh
.venv/bin/python -m lastfirstdate.synthetic
```

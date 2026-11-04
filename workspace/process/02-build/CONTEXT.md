# Stage 02: build

Write the code and tests for the requirements agreed in stage 01.

## Read

- The new sprint section in `docs/SPEC.md`.
- The `CONTEXT.md` of each algorithm stage the sprint changes (`workspace/algorithm/`).
- Only the source and test files those name.
- `src/lastfirstdate/config.py` if you add a setting.

## Rules while building

- **Settings:** every new tunable number goes in both `config/lastfirstdate.toml` and
  `config/phase1.toml`, with a dataclass field and a range check in `config.py` (R5).
- **Engine and simulation stay apart:** engine modules never import simulation modules or
  read `GroundTruth`. Add new modules to the right list in `tests/test_engine_boundary.py`.
- **Rust:** if you change a function that has a Rust twin (`native.py` lists them), change
  both versions; `tests/test_native.py` checks that they agree. Rebuild with
  `.venv/bin/maturin develop --release --skip-install`.
- **Tests:** every test docstring starts with its `FR-LM-NN`. An acceptance target the
  simulation doesn't meet becomes `@pytest.mark.xfail(strict=True, reason=...)`. Never weaken
  a test.
- **Earlier results must not move:** with the default settings, the default fake population
  and the earlier reports stay the same.

## Commands

```bash
.venv/bin/python -m pytest -q tests/test_<module>.py      # quick, while working
./scripts/check.sh                                          # all gates, about 3 minutes
```

`check.sh` runs Rust format, Clippy, the Rust tests, a fresh build of the Rust module, ruff,
pytest and the traceability check.

## Produce

Code and tests, committed locally with the attribution lines from `CLAUDE.md`. Don't push.

## Done when

`./scripts/check.sh` ends with "All gates passed."

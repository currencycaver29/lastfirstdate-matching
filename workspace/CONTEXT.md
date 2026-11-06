# Workspace: the lastfirstdate matching algorithm

This folder organises the AI-assisted work on the algorithm with Interpretable Context
Methodology (ICM). Each part of the algorithm has a folder whose `CONTEXT.md` says what that
part does, what it takes in and passes on, where its code, settings and tests are, and what we
already know about it. The code lives in `src/` and `rust/` as usual; these files are a map,
so a session can load just the part it needs.

## Start here (new session)

1. Read this file.
2. Working on part of the algorithm? Read that stage's `CONTEXT.md` (below), then only the
   files it names.
3. Running a sprint? Read `process/` in order: spec, build, test runs, notes, handoff.
4. `CLAUDE.md` holds the rules and is loaded automatically. Don't read the whole repo, old
   sprint notes or long reports unless a context file asks for them: that is where tokens go.

## The algorithm, stage by stage

Each stage's output feeds the next.

```
profiles ─▶ 01 filter ─▶ 02 calibration ─▶ 03 looks taste ─┐
                                        04 personality ────┴▶ 05 mutual score ─▶ 06 daily lists
                                                                                       │
      10 trust ◀── ratings ◀── 07 booking ◀── her pick and his yes ◀───────────────────┘
          │                      │
          └──▶ lists (stage 6)   └──▶ 08 learning ──▶ back to stages 3 and 5
                                 └──▶ 09 guarantee: delivered dates, credits, the ratio gate
```

| Stage | In one line |
|---|---|
| [01 filter](algorithm/01-filter/CONTEXT.md) | Only pairs who could really meet: orientation, age, distance, dealbreakers |
| [02 calibration](algorithm/02-calibration/CONTEXT.md) | The ~30 photos (singles and pairs) that teach the looks model the most |
| [03 looks taste](algorithm/03-looks-taste/CONTEXT.md) | Each member's own photo taste: a starting taste plus a personal correction |
| [04 personality](algorithm/04-personality/CONTEXT.md) | Prompts, intent answers, interests |
| [05 mutual score](algorithm/05-mutual-score/CONTEXT.md) | 50% looks, 50% personality, both directions combined |
| [06 daily lists](algorithm/06-daily-lists/CONTEXT.md) | Up to 30 a day, shared fairly, with a boost for men owed dates |
| [07 booking](algorithm/07-booking/CONTEXT.md) | Date window, 15-minute payment hold, fair venue |
| [08 learning](algorithm/08-learning/CONTEXT.md) | Learning from picks, dates and ratings; a little exploration |
| [09 guarantee](algorithm/09-guarantee/CONTEXT.md) | Forecast, mid-cycle alerts, the ratio gate, billing cycles, credits |
| [10 trust](algorithm/10-trust/CONTEXT.md) | Behaviour-only trust: lowers ranking, flags for review |

Used by every stage:

| Shared | What's in it |
|---|---|
| [settings and data](shared/settings-and-data/CONTEXT.md) | The settings files, the member model, the app's database design |
| [test runs](shared/test-runs/CONTEXT.md) | Fake members and behaviour, the reports, how to measure honestly |
| [Rust core](shared/rust-core/CONTEXT.md) | The hot loops in Rust and their Python twins |

## How a sprint is run

| Stage | Produces | Done when |
|---|---|---|
| [01 spec](process/01-spec/CONTEXT.md) | The sprint's section in `docs/SPEC.md` | The owner agrees on the scope |
| [02 build](process/02-build/CONTEXT.md) | Code and tests | `./scripts/check.sh` passes; committed locally |
| [03 test runs](process/03-test-runs/CONTEXT.md) | `process/03-test-runs/results-sprint-N.md` | Every number has the command that made it |
| [04 notes](process/04-notes/CONTEXT.md) | `docs/sprints/sprint-N.md`; spec, README and workspace updated | The owner has the notes and the explanation |
| [05 handoff](process/05-handoff/CONTEXT.md) | `docs/handoff/` for whoever builds the app | Only when an interface the app uses changes |

After stage 04, stop: the owner studies the sprint. Push only when the owner says so.

## Current state

- **Current sprint:** 15, not chosen yet. Options include a consented pilot to recheck the
  photo settings, the gate every cycle, or waiting for the app and real Phase 1 data. Ask the
  owner in process stage 01.
- **Last finished:** Sprint 14 (v0.14.0), abuse testing: synthetic attackers in the test runs
  (`attacks.py`) and a defence for every attack that gained. Prompts are compared without the
  zone's average (D21); the gate counts only verified women who aren't silent (D22); one
  account, one photo exercise, and only verified members teach the starting taste (D23);
  ratings count only for completed dates; malformed booking rows are refused by name.
- **Where the engine stands:**
  - all ten stages are built and tested;
  - the real booking flow delivers about 77% to 82% at 3:1 in a 300-woman zone (NYC demo 77%);
  - the zone's size matters more than the ratio (about 500 women for a 90% promise);
  - the Phase 1 gate uses the delivered forecast;
  - the app calls the engine through `src/lastfirstdate/app.py` (see `docs/handoff/`);
  - open problems are GitHub issues; #1, #2, #14 and #17 block the launch.

## Rules that matter everywhere

- R1 to R7 in `docs/SPEC.md`: mutual orientation lock, no global attractiveness score,
  synthetic data only, offline tests, every tunable number in the settings files,
  dealbreakers as hard filters, seeded randomness.
- The engine never reads `GroundTruth`.
- Every test names its requirement (`FR-LM-NN`); never weaken a test.
- Commits are authored by the owner, with the Claude co-author line.
- Push only on the owner's word.

## Decisions already made (don't reopen without the owner)

D1 to D23 in `docs/SPEC.md`. The most relevant now:
- women choose (D1);
- 50/50 looks and personality (D2);
- the Phase 1 settings profile (D14);
- calibration counted in answers (D15);
- Rust twins for the hot loops (D16);
- his own absence counts as delivered (D17, pending the lawyer);
- forgetting a member keeps an anonymous record (D18, pending the lawyer);
- timestamps in UTC, dates in the zone's own time (D19);
- SigLIP and MiniLM as the production models (D20);
- prompts compared without the zone's average (D21);
- the gate counts verified, non-silent women (D22);
- one account, one photo exercise; verified members teach the starting taste (D23).

## Questions the owner hasn't answered yet (ask, don't assume)

1. D17: should his own late cancellation or no-show count as delivered?
2. Should a pick he declines count toward his guarantee?
3. Flag bad actors after about 6 poor ratings (now), or sooner?
4. Can marketing bring in about 15% new women per cycle?
5. Which show-up tools are acceptable (reminders, fees, deposits)?
6. Can Phase 1 recruit about 500 committed women before selling subscriptions?
7. Keep the 90% gate target, or use 85% to 88% in large zones, where the gate is sensitive?
8. Should a trust flag pause a man at once, or wait for a person? (Rings of 3 colluding women
   got a man paused in 5 of 24 tries; waiting protects against that, pausing against real bad
   actors.)
9. Code review findings 1 and 3 to 9 are still open (finding 2 was fixed in Sprint 14).

<h1 align="center">lastfirstdate-matching</h1>

<p align="center">
  The matching engine behind <a href="https://lastfirstdate.vercel.app/">lastfirstdate</a>:
  mutual, fair, and built around real dates.
</p>

<p align="center">
  <a href="https://github.com/currencycaver29/lastfirstdate-matching/actions/workflows/checks.yml"><img alt="checks" src="https://github.com/currencycaver29/lastfirstdate-matching/actions/workflows/checks.yml/badge.svg"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB">
  <img alt="Rust" src="https://img.shields.io/badge/rust-core-B7410E">
  <img alt="synthetic data only" src="https://img.shields.io/badge/data-synthetic%20only-555555">
</p>

---

## About

lastfirstdate is a verified dating service for New York: women choose who they meet, and men
subscribe for three guaranteed dates a month. This repository is the algorithm behind it,
the part that decides who sees whom, when they meet and where. The app itself is built
separately.

After a short photo exercise, the engine learns what each member finds attractive. It then
ranks every possible match **50% on looks, as that member sees them, and 50% on personality
and interests**, and only puts forward matches that work for both people. Daily lists are
shared out fairly, so a few popular profiles don't take all the attention. Each profile comes
with a date window both people can make. Behind it all, the engine protects the guarantee: it
forecasts whether each man will get his dates, and pauses men's sign-ups when the market
can't serve more of them.

Everything here runs on **synthetic members**. No real profile, photo or embedding is ever
stored in this repository.

## How it works

```mermaid
flowchart TD
    P(["Member profiles"]) --> F

    subgraph meet ["Who could meet"]
        F["<b>1 · Filter</b><br/>orientation · age · distance<br/>dealbreakers · verification"]
    end

    subgraph like ["What each member likes"]
        C["<b>2 · Calibration</b><br/>about 30 photos<br/>singles and pairs"] --> L["<b>3 · Looks taste</b><br/>a personal model<br/>for every member"]
        T["<b>4 · Personality</b><br/>prompts · intent answers<br/>interests"]
    end

    subgraph see ["Who sees whom"]
        M["<b>5 · Mutual score</b><br/>50% looks<br/>50% personality<br/>both must want it"] --> D["<b>6 · Daily lists</b><br/>up to 30 a day<br/>attention shared fairly"]
    end

    subgraph date ["The date"]
        B["<b>7 · Booking</b><br/>a window both can make<br/>15-minute payment hold<br/>a venue fair to both"]
    end

    subgraph after ["After every date"]
        LE["<b>8 · Learning</b><br/>sharpens each<br/>member's taste"]
        TR["<b>Trust</b><br/>behaviour only, never looks<br/>lowers low-trust profiles"]
        G["<b>Guarantee</b><br/>boosts men owed dates<br/>admits men only when<br/>they can be served"]
    end

    F --> C
    F --> T
    L --> M
    T --> M
    D -- "she picks, he accepts" --> B
    B -- "how it went" --> LE
    B -- "private ratings" --> TR
    B -- "delivered dates" --> G

    classDef input fill:#4a5568,stroke:#2d3748,color:#ffffff
    classDef stage fill:#2b6cb0,stroke:#1a4971,color:#ffffff
    classDef loop fill:#2f855a,stroke:#1c5a3c,color:#ffffff
    classDef business fill:#c05621,stroke:#8a3c14,color:#ffffff
    class P input
    class F,C,L,T,M,D,B stage
    class LE,TR loop
    class G business
    style meet fill:transparent,stroke:#8b949e,stroke-dasharray:5 3
    style like fill:transparent,stroke:#8b949e,stroke-dasharray:5 3
    style see fill:transparent,stroke:#8b949e,stroke-dasharray:5 3
    style date fill:transparent,stroke:#8b949e,stroke-dasharray:5 3
    style after fill:transparent,stroke:#8b949e,stroke-dasharray:5 3
```

Blue boxes are the matching stages a profile passes through every day. After each date, the
green boxes learn from what happened and the orange box keeps the promise of three dates a
month; all three shape the next day's lists.

<details>
<summary><b>What each stage does</b></summary>

<br/>

| # | Stage | What it does |
|:-:|---|---|
| 1 | **Filter** | Keeps only pairs who could actually meet: orientation both ways, age, distance, dealbreakers, verification |
| 2 | **Calibration** | Picks about 30 photos, singles and "who are you more drawn to?" pairs, that teach the model the most about a new member |
| 3 | **Looks taste** | Learns each member's personal photo taste as a small Bayesian model on top of a shared starting point |
| 4 | **Personality** | Scores free-text prompts, intent answers (timeline, exclusivity, kids) and interests |
| 5 | **Mutual score** | Mixes looks and personality 50/50 in each direction, then combines both directions so both people must want it |
| 6 | **Daily lists** | Builds each woman's daily list under an exposure cap, with a boost for men still owed dates |
| 7 | **Booking** | Proposes the earliest window both are free, holds it for 15 minutes while both $10 payments arrive, picks a venue fair to both |
| 8 | **Learning** | Learns from picks, dates and private ratings, and keeps exploring a little |
| – | **Trust** | Builds a private score from respectfulness, profile accuracy and showing up; lowers low-trust profiles and flags them for review |
| – | **Guarantee** | Forecasts each man's chance of three dates, admits men from the waitlist only while that holds, counts delivered dates and credits shortfalls |

</details>

## Results in simulation

All numbers come from synthetic members. They compare designs; they are not predictions.

| Question | Answer |
|---|---|
| How well does the looks model learn a member's taste? | AUC 0.844, against a ceiling of 0.875 |
| Do fair lists spread attention? | Exposure Gini falls from 0.39 to 0.31 |
| Does the ratio gate protect the guarantee? | Admitted men reach 3 dates 92% to 96% of the time; admitting everyone gives 12.5% |
| How many men get 3 *delivered* dates in the real booking flow? | About 68%; 90% with no cancellations, no-shows or missed payments |
| How much faster is the Rust core? | Looks taste 3.6×, daily lists 10×, the gate 28× to 81×, with identical results |

Each sprint's notes in [`docs/sprints/`](docs/sprints/) explain these results and their limits.

## Getting started

### Requirements

- Python 3.11 or newer
- Rust, from [rustup.rs](https://rustup.rs). Installing the package compiles the Rust core.

### Install

```bash
git clone https://github.com/currencycaver29/lastfirstdate-matching.git
cd lastfirstdate-matching
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

On Windows, use `.venv\Scripts\pip` instead of `.venv/bin/pip`.

### Try it

```python
from lastfirstdate.config import load
from lastfirstdate.synthetic import generate
from lastfirstdate.simulate import run_month

cfg = load()                   # research defaults; load("config/phase1.toml") for Phase 1 rules
pop = generate(cfg, seed=7)    # 1,000 fake New Yorkers
report = run_month(pop, cfg)
print(report.text())
```

### Reports

Each report runs a simulation and prints what it found:

```bash
.venv/bin/python -m lastfirstdate.simulate          # one simulated month
.venv/bin/python -m lastfirstdate.looks_report      # how well the looks model learns taste
.venv/bin/python -m lastfirstdate.mutual_report     # list rankings compared
.venv/bin/python -m lastfirstdate.fairness_report   # exposure, fairness audit, hold-back test
.venv/bin/python -m lastfirstdate.learning_report   # three months with and without learning
.venv/bin/python -m lastfirstdate.phase1_report     # Phase 1 rules: pair questions, the ratio gate
.venv/bin/python -m lastfirstdate.cycle_sim         # the real booking flow over three billing cycles
.venv/bin/python -m lastfirstdate.forecast_fit      # refit the guarantee forecast (about 10 minutes)
```

### Checks

```bash
./scripts/check.sh
```

This runs Rust formatting, Clippy and tests, rebuilds the Rust module, then runs ruff, the
Python tests and a check that every requirement has a test. The same checks run on GitHub for
every push.

## Configuration

Every weight, cap and threshold lives in a settings file, never in code:

| File | Use |
|---|---|
| [`config/lastfirstdate.toml`](config/lastfirstdate.toml) | Research defaults: a 1,000-member city, every orientation |
| [`config/phase1.toml`](config/phase1.toml) | lastfirstdate's Phase 1: one NYC zone, 100 paying men and 300 women, a three-date cap, 30 calibration photos |

`LASTFIRSTDATE_BACKEND=python` runs everything in plain Python instead of Rust. The answers
are the same, only slower.

## Project structure

```
lastfirstdate-matching/
├── config/              settings files and the fitted forecast
├── db/schema.sql        the database the app should keep
├── docs/
│   ├── SPEC.md          requirements, rules and decisions, sprint by sprint
│   ├── research.md      the 17 papers behind the design
│   ├── sprints/         study notes for every sprint
│   ├── fit-report.md    how the engine fits lastfirstdate's real rules
│   └── risk-plan.md     launch risks and questions for a lawyer
├── rust/
│   ├── lastfirstdate-core/   the hot loops in pure Rust
│   └── lastfirstdate-py/     Python bindings (lastfirstdate._core)
├── src/lastfirstdate/   the engine (Python) and the simulator
├── tests/               one or more tests per requirement
├── workspace/           a map of every algorithm stage, for working with an AI assistant
└── scripts/check.sh     all the checks
```

## Principles

These rules are never broken, and tests enforce them:

1. **Mutual orientation lock.** Someone is shown to you only if each of you is a gender the
   other is looking for.
2. **No global attractiveness score.** Looks are only ever scored from one member's own point
   of view. Nothing ranks people by overall attractiveness.
3. **Synthetic data only** in this repository. In production, photo embeddings are treated as
   biometric data: encrypted, kept apart, deleted when a member leaves.
4. **Tests run offline,** with no model downloads.
5. **Every tunable number lives in the settings files.**
6. **Dealbreakers are hard filters,** never soft penalties.
7. **Same seed, same result.** All randomness is seeded.

## Documentation

| Document | What's in it |
|---|---|
| [Specification](docs/SPEC.md) | Every requirement, rule and decision, and how the plan changed |
| [Research map](docs/research.md) | The 17 papers and where each one is used |
| [Sprint notes](docs/sprints/) | In-depth study notes for every sprint, with results and limits |
| [Fit report](docs/fit-report.md) | The engine checked against lastfirstdate's website |
| [Risk plan](docs/risk-plan.md) | What to do before launch, including a ready-to-send lawyer briefing |
| [Workspace](workspace/CONTEXT.md) | A map of the whole algorithm, stage by stage, and how each sprint is run |

## Working with an AI assistant

The [`workspace/`](workspace/CONTEXT.md) folder follows Interpretable Context Methodology
(ICM). Every stage of the algorithm has a short `CONTEXT.md`: what it does, what it takes in
and passes on, its code, settings, tests and known findings. An assistant working on, say,
the daily lists reads only that folder instead of the whole repository. A `process/` folder
describes how each sprint runs: spec, build, simulate, notes, handoff.

## Roadmap

| Sprint | Focus | Status |
|:-:|---|---|
| 0 | Foundation: settings, member model, synthetic city, checks | Done |
| 1 | A working matcher on personality alone | Done |
| 2 | The looks model and the photo exercise | Done |
| 3 | The 50/50 mutual score | Done |
| 4 | Fair daily lists | Done |
| 5 | Learning from dates | Done |
| 6 | lastfirstdate's real rules, the guarantee forecast and the ratio gate | Done |
| – | The Rust core for the hot loops | Done |
| 7 | Booking, billing, trust, geography and the app's database | Done |
| 8 | The forecast and the gate on delivered dates | Next |

## License

No open-source license has been chosen yet. All rights are reserved by the owner.

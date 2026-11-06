# Sprint 15 results: the code review findings

Tests: `python -m pytest tests/test_sprint15.py -q` (8 tests, one per finding). Run against the
code before the fixes, all eight fail; with the fixes, all pass. The full suite and the Rust
tests pass.

## Each finding, before and after

| Finding | Before | After |
|---|---|---|
| 1. A date from before a subscription | one cancelled booking dated before a man's signup stopped every woman's list (`ValueError: no cycle before signup`) | the lists are built; that booking uses none of his cycles' dates; `close_cycle` for a day before signup is refused by name |
| 3. Photo orientation | a portrait photo saved sideways with an EXIF tag reached SigLIP rotated 90 degrees | turned upright first; the pixels equal the upright photo's exactly |
| 4. The cache | changed answers went unnoticed when their number stayed the same; a forgotten member's taste, and the shared fits made with their photo and answers, stayed until the weekly refit | any change refits that member; a forgotten member's taste goes at the next refresh, and the shared fits are refitted that night (D24) |
| 5. Times written | a time without a zone was stored as if it were UTC (4 to 5 hours wrong from a New York server) | refused; aware times stored in UTC with `+00:00` (D25) |
| 6. Pending picks | the morning lists offered a man for an evening another woman's pick was waiting on (61 of 286 such picks refused in the Sprint 12 stress test) | the pending pick holds his day in the lists too |
| 7. Checking a pick | read every booking and built a members-by-members block grid inside the pick's transaction | reads only the two members' rows (below) |
| 8. The model check | read every encrypted embedding to compare a name | asks for the distinct names only |
| 9. The text model's memory | kept every prompt it ever read, forgotten members' included | keeps only the prompts of its last call |

## Time to check a pick (finding 7)

The script below (run from the repo root); SQLite in memory, Phase 1 settings, the
first entry of 50 women's lists, mean time per check:

| Zone | Other bookings in the database | Before | After |
|---|---:|---:|---:|
| 400 members | none | 0.1 ms | 0.1 ms |
| 400 members | 8,000 | 64.8 ms | 1.4 ms |
| 2,000 members | none | 0.8 ms | 0.1 ms |
| 2,000 members | 40,000 | 422.6 ms | 2.0 ms |

Before, the time grew with every booking in the zone and the square of its members; now it
depends only on the two members' own rows. The answers are identical (the FR-LM-76 test
compares every one of a day's picks with the old version).

```python
import dataclasses, sqlite3, sys, time
from datetime import date
sys.path.insert(0, "tests")
from test_sprint15 import reference_recheck  # the old version, kept in the test
from lastfirstdate import app
from lastfirstdate.app_fill import fill_database
from lastfirstdate.config import load
from lastfirstdate.synthetic import generate

TODAY, EPOCH = date(2026, 11, 4), date(2026, 11, 2)
cfg = load("config/phase1.toml")
for n in (400, 2000):
    c = dataclasses.replace(cfg, population=dataclasses.replace(cfg.population, n_members=n))
    pop = generate(c, seed=7)
    con = sqlite3.connect(":memory:")
    filled = fill_database(con, pop, c, TODAY, 7, subscribers=n // 8)
    market = app.load_market(con, c, app.Calendar(EPOCH), TODAY)
    feats = app.photo_feature_matrix(pop.photo_embeddings, c)
    answers = app.calibration_answers(con, market, c)
    tastes = app.fit_tastes(market, feats, answers, c, start=filled.start)
    scores = app.nightly_scores(market, tastes, feats, c)
    women = [int(market.ids[k]) for k, m in enumerate(market.members) if m.gender == "woman"]
    picks = [(w, es[0]) for w, es in app.day_lists(con, market, scores, women, c).items()][:50]
    for extra in (0, 20 * n):
        if extra:
            subs = filled.subscribers
            rows = [(women[k % len(women)], subs[(k * 7) % len(subs)],
                     f"2026-{1 + k % 9:02d}-{1 + k % 28:02d}") for k in range(extra)]
            con.executemany(
                "INSERT INTO bookings (member_a, member_b, source, date_on, part, state, "
                "proposed_at, venue_id, showed_a, showed_b) VALUES (?, ?, 'list', ?, 'evening', "
                "'completed', '2026-01-01T00:00:00Z', 1, 1, 1)", rows)
        for name, f in (("before", reference_recheck), ("after", app.recheck_pick)):
            t = time.perf_counter()
            for w, e in picks:
                f(con, market, w, e.member_id, e, c)
            print(n, extra, name, (time.perf_counter() - t) / len(picks) * 1000, "ms")
```

## The test runs don't move

The test runs keep their own booking state (`cycle_runs.py`) and don't call the functions these
fixes changed, so the NYC demo, the attack report and every earlier report give the same
numbers. The Sprint 12 fixes report (`python -m lastfirstdate.fixes_report`) does call them.
With prompt centring off it gives exactly Sprint 12's numbers (160 picks on a taken window;
90 kept, 135 moved, 61 refused). With centring on, as the Phase 1 settings have it since
Sprint 14 (D21), the women's top profiles differ a little: 152 picks on a taken window; 89
kept, 137 moved, 60 refused; still nobody with two dates on one day. Centring off, for the
comparison:

```python
from lastfirstdate import attacks, fixes_report
from lastfirstdate.config import load
fixes_report.load = lambda *a, **k: attacks.centered(load(*a, **k), False)
fixes_report.main()
```

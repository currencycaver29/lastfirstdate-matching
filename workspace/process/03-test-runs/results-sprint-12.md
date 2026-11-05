# Sprint 12 results: bug fixes before Phase 1

Command for every number below: `python -m lastfirstdate.fixes_report`. Tests:
`python -m pytest tests/test_sprint12.py -q` (11 tests). Zone: Phase 1 settings, seed 7, 60
subscribers, today Wednesday 4 November 2026.

## Times (FR-LM-54)

A Friday 6 November, 7 pm date, cancelled at different times, stored in UTC without an offset
as the schema asked:

| True hours ahead | The old code read | Read now | Old verdict | Verdict now |
|---:|---:|---:|---|---|
| 23 | 18 | 23 | late | late |
| 24 | 19 | 24 | late | early |
| 26 | 21 | 26 | late | early |
| 28 | 23 | 28 | late | early |
| 29 | 24 | 29 | early | early |

- Every cancellation made 24 to 28 hours ahead was judged late (4 to 5 hours wrong, the gap
  between UTC and New York). That decides D17 and the fee refund.
- With an offset (`Z` or `+00:00`), the old code crashed with a `TypeError`.
- The Sprint 11 history now stores UTC with offsets and still rebuilds all 291 cycles and 838
  weekly snapshots identically. Its cancellations are only 6 or 48 hours ahead, so none fell in
  the wrong band.
- Across the 1 November clock change, 7:30 pm the evening before to 7 pm is 24.5 hours, not
  23.5 (tested).

## Picks (FR-LM-55)

One morning, all 286 women with a list pick the top profile on it at once (a stress test):

| | Count |
|---|---:|
| Picks that would land on a window already taken | 160 |
| Picks that would land on a day he already has a date | 184 |
| With `recheck_pick`: kept the window shown | 90 |
| moved to the next window both can make | 135 |
| refused (no window left in 14 days, or at his cap) | 61 |
| Members with two dates on one day afterwards | 0 |

`recheck_pick` takes about 1 ms; `day_lists` about 0.06 s.

## Blocks and forgetting (FR-LM-56, FR-LM-57)

- A block counts at once: off the lists (even with last night's scores), out of both photo
  pools, refused by `recheck_pick`, ineligible at the next refresh (tested).
- Forgetting a member: about 1 ms for the statements; the zone reloads and the nightly refresh
  takes about 0.5 s; the forgotten member has no eligible pairs.

## Earlier results

Unchanged: with no blocks and no forgotten members, eligibility, lists, tastes and the gate are
exactly as before, and every earlier test passes.

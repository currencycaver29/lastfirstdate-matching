# Sprint 10 results: step-by-step calibration and the nightly cache

Commands: `python -m pytest tests/test_sprint10.py -q`; the timing snippet in
`docs/sprints/sprint-10.md` (a Phase 1 zone and a 2,000-member zone, seed 7).

## Equivalence (FR-LM-49)

| Check | Result |
|---|---|
| `CalibrationSession` against a frozen copy of the pre-Sprint 10 algorithm (25 members; Phase 1 with pairs, research setup without) | identical questions and taste models |
| Default 1,000-member city, `build_scores` | mutual scores byte-identical to the saved pre-Sprint 6 reference; a month books the same dates |
| Interrupted after 3, 14, or all but one answer, then resumed | the same remaining questions |
| Resumed from the database after 15 answers | asks exactly the 16th question originally asked |

## The cache (FR-LM-50)

| Zone | Full rebuild | One member's answers changed | Nothing changed | Scores in memory |
|---|---:|---:|---:|---:|
| 400 members | 0.50 s | 0.23 s (refitted 1) | 0.07 s | 5 MB |
| 2,000 members | 4.87 s | 3.07 s (refitted 1) | 0.52 s | 132 MB |

A cached refresh equals a full rebuild with the same photo compression and starting taste
(every taste model and every score). A profile edit with no new answers recomputes the scores.

## Found while measuring

- The first timing picked a member who never did the exercise ("refitted 0"); corrected.
- An unchanged refresh still recomputed every score (2.2 s at 2,000 members); now it returns
  the cached scores.
- Memory grows with the square of the zone size: run a city as zones of a few thousand.

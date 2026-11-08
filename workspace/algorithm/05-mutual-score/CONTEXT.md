# Stage 5: mutual score

Combines looks and personality, in both directions, so a match has to work for both people.

## Takes in / passes on

- **In:** `V` (stage 3) and `P` (stage 4).
- **Out:** `S` (directional, i about j) and `M` (mutual), each an (n, n) matrix. M ranks the
  daily lists.

## The maths

1. **Percentiles per viewer:** V and P are turned into each viewer's own percentiles, so the
   two are on the same scale (`percentile_rows`).
2. **50/50:** S_ij = looks_weight · V_ij + (1 − looks_weight) · P_ij (`directional_score`).
3. **Mutual:** M_ij is a weighted harmonic mean of S_ij and S_ji, with the chooser's side
   weighted by `chooser_weight` (`mutual_score`). A harmonic mean is high only if both
   directions are.

## Code

`src/lastfirstdate/mutual.py`. The full pipeline for a population, including calibration, is
`mutual_report.build_scores`; it returns `Scores(eligible, personality, looks, directional,
mutual)`.

## Settings

`[mix]`: `looks_weight` (0.5), `chooser_weight` (0.5).

## Tests

FR-LM-18 to FR-LM-20 (`tests/test_mutual.py`).

## Research

- Pizzato et al., *RECON* (2010): score both directions, then combine them with the harmonic
  mean (`docs/research.md`).

## Decisions

D2 (50/50), D13 (chooser weight 0.5).

## What we know

- Higher M means the other person is likelier to accept.
- Choosers are the pickier side: a chooser weight near 0.8 gives about 1 point more mutual
  yeses, but it is kept at 0.5 until real data says otherwise.
- In a 1,000-member city members see almost their whole pool each month, so the ranking
  method barely changes monthly outcomes.

## If you change it

Every list and booking downstream changes. Compare with
`python -m lastfirstdate.mutual_report` before and after.

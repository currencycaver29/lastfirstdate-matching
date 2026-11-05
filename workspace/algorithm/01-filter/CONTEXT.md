# Stage 1: filter

Keeps only the pairs who could really meet, before anything is scored. Everything after this
stage works on its output, so a mistake here silently removes or adds matches everywhere.

## Takes in / passes on

- **In:** member profiles (`models.Member`), as arrays (`arrays.MemberArrays`).
- **Out:** `eligible`, an (n, n) boolean matrix: True where i and j may be matched. Also
  `member_ok`, the members who are verified and active.

## Code

`src/lastfirstdate/filter.py`: `member_ok`, `eligibility_matrix` (all pairs at once),
`pair_passes` (one pair, the readable version), `dealbreakers_pass`.

## Rules applied, all both ways

- **Orientation (R1):** each is a gender the other seeks.
- **Age:** each is inside the other's range.
- **Distance:** within both people's maximum (straight-line km, D9).
- **Free time:** at least `min_shared_slots` free weekly slots in common.
- **Dealbreakers:** hard filters, never penalties (R6), including incompatible intent answers
  such as "Want kids" with "Not for me".
- **Verification and activity.**
- **Blocks (Sprint 12):** a pair where either blocked the other is never eligible
  (`market.blocked`, passed as `excluded`); forgotten members are never eligible.

## Settings

`[filter]`; dealbreakers and incompatible pairs under `[[values]]`.

## Tests

FR-LM-04 (`tests/test_filter.py`), FR-LM-30 (`tests/test_phase1.py`), FR-LM-56 and FR-LM-57
(`tests/test_sprint12.py`).

## Decisions

D9 (straight-line distance until the original SQL pre-filter is added to `db/`).

## What we know

- In the 1,000-member research city the median member has about 62 possible matches; about
  65 have fewer than 10.
- In Phase 1, a woman's pool holds about 46 men and a man's about 70 to 130 women, depending
  on the ratio.

## If you change it

The fast matrix version and the one-pair version must agree (tested). Pool sizes feed the
guarantee forecast (stage 9), so rerun `python -m lastfirstdate.forecast_fit` if the filter
changes what counts as a possible match.

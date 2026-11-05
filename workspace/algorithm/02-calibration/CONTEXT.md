# Stage 2: calibration

Chooses the photos a new member is asked about, so the looks model (stage 3) learns their
taste from as few answers as possible.

## Takes in / passes on

- **In:** the member's calibration pool (matchable members of a gender they seek, inside their
  age range: `looks.calibration_pool`, D12), and everyone's photo features.
- **Out:** the photos shown, in order, and the member's answers, which feed `PersonalTaste`
  (stage 3).

## How it chooses

1. **Variety first:** `n_variety` photos picked by a greedy determinantal point process
   (`select_varied`). Each pick is the photo least like those already chosen.
2. **Pair questions** ("who are you more drawn to?"): `n_pairs` pairs, each the most
   informative pair among the 30 most uncertain profiles (`select_pair`).
3. **Informative singles** for the rest: the largest expected information,
   log(1 + p(1 − p)·s²) (`PersonalTaste.information`).

## Code

`src/lastfirstdate/looks.py`: `CalibrationSession` (the exercise one question at a time:
`next_question`, `answer`, `resume`), with `calibrate` running a session to the end, and
`calibration_pool`, `select_varied`,
`select_pair`, `PersonalTaste.information`. The quadratic forms and the pair search have Rust
twins (`native.py`).

## Settings

`[calibration]`: `n_profiles`, `n_variety`, `n_pairs`, and the model penalties.

## Tests

FR-LM-16, FR-LM-17 (`tests/test_looks.py`), FR-LM-32 (`tests/test_phase1.py`), FR-LM-35
(`tests/test_native.py`).

## Papers

- 1, Brochu et al.: active learning from choices.
- 2, Kulesza and Taskar: determinantal point processes.
- 3, Chu and Ghahramani: preference learning.

## Decisions

D5 (50 profiles in the research setup), D12 (the calibration pool), D15 (count answers, not
photos).

## What we know

- Choosing well beats choosing at random by only 0.004 to 0.009 AUC; most of the gain comes
  from having a personal model at all.
- Per answer, pairs teach as much as singles. At a fixed 30 photos, 9 pairs cost 0.019 AUC
  (0.818 against 0.838), because a pair uses two photos for one answer.
- Pairs' real advantage, being immune to a member's mood or threshold, isn't modelled.
- In the app, `app.photo_pool` is the pool: never someone either person blocked (Sprint 12).
  When a member is forgotten, everyone's answers about their photo are deleted, so those
  members' models refit with fewer answers; the photo compression is fitted only on members
  who still have a photo.

## If you change it

Run `python -m lastfirstdate.looks_report` and `python -m lastfirstdate.phase1_report`, and
keep the Rust and Python versions agreeing.

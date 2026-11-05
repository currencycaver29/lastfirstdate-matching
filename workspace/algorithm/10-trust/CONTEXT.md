# Stage 10: trust

A private trust score that lowers a member's ranking and flags them for review, built from
behaviour only.

## Takes in / passes on

- **In:** private ratings after each date (respectfulness, profile accuracy, conversation,
  "would meet again") and attendance (kept, no-show, late cancellation).
- **Out:** trust per member; the ranking factor applied to lists (stage 6); review flags that
  pause a profile.

## The maths

    conduct     = (k × prior + sum of scaled respect and accuracy ratings) / (k + ratings)
    reliability = (k × prior + dates kept) / (k + dates scheduled)
    trust       = prior + (1 − w)(conduct − prior) + w(reliability − prior)
    factor      = 1 − ranking_weight × max(0, prior − trust) / prior

- A newcomer gets exactly the prior: no penalty.
- High trust gives no boost.
- A member is flagged when conduct < `flag_below` with at least `flag_min_ratings` ratings.
- Conversation and "again" never enter trust (R2: trust must not become popularity).

## Code

`src/lastfirstdate/trust.py`: `Rating`, `TrustBook`, `with_trust`.

## Settings

`[trust]`.

## Tests

FR-LM-39, FR-LM-42 (`tests/test_sprint7.py`).

## What we know

- In test runs, every bad actor with 6 or more ratings was flagged, and no one else. A
  typical bad actor crosses the line at about the 6th poor rating, roughly two cycles.
- With a small synthetic halo (raters kinder to people they find attractive), trust
  correlated +0.15 with looks. Audit real ratings by group.

## Open question

Flag sooner, at the cost of reviewing more ordinary members? That is the owner's call.

## If you change it

Rerun `python -m lastfirstdate.cycle_runs` and check the flag numbers and the trust-versus-looks
correlation.

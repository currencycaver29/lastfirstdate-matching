# Stage 4: personality

Scores how well two members fit on what they wrote and answered, apart from looks.

## Takes in / passes on

- **In:** prompts or interests, intent answers, Big Five (if collected).
- **Out:** `P`, an (n, n) personality score.

## The score

    P = w_text · t + w_values · A + w_big5 · p

- **t:** text similarity. TF-IDF cosine over prompt words (`WordEncoder`) or interest tags
  (`TagEncoder`); MiniLM replaces these in production through the `TextEncoder`
  interface.
- **A:** values compatibility. "Accepts" questions, or closeness questions:
  1 − |rank_i − rank_j| / (options − 1).
- **p:** Big Five similarity. The weight is 0 in Phase 1, because the site has no personality
  test.

## Code

`src/lastfirstdate/personality.py`: `personality_matrix`, `interest_similarity`,
`values_compatibility`, `big5_similarity`, the encoders.

## Settings

`[personality]`: `w_text`, `w_values`, `w_big5`, `text_encoder`, `center_text`; `[[values]]`
questions with `mode`.

## Tests

FR-LM-05 to FR-LM-08 (`tests/test_personality.py`), FR-LM-30, FR-LM-31
(`tests/test_phase1.py`), FR-LM-61 (`tests/test_sprint13.py`).

## Methods

- Sentence embeddings (MiniLM in production).
- Traits measured before meeting barely predict desire for one specific person, so treat the
  weights as guesses and retune them from outcomes.

## Decisions

D6 (the research split 0.4 / 0.3 / 0.3); Phase 1 uses 0.55 text and 0.45 values.

## What we know

- TF-IDF matches words, not meaning: "rock climbing" and "bouldering" score 0.
- **Production (Sprint 13, D20):** `text_encoder = "sentence"` reads prompts with MiniLM:
  "rock climbing" and "bouldering" score 0.67, against 0.13 for opera. On the synthetic
  prompts, which reuse a small vocabulary, it separates shared lifestyles only a little better
  (AUC 0.888 against 0.869); real prompts vary far more. It changes about 2 of each member's
  top 10. Tests and test runs keep "words".
- The weights are guesses until real dates exist.
- **Stuffed prompts (Sprint 14, D21, FR-LM-64):** a prompt full of the words everyone writes
  climbed from the middle to the 67th percentile of women's personality rankings with word
  counts; popular interests written as a prompt reached the 59th with MiniLM. With
  `center_text` (on in Phase 1) the zone's average vector is taken out before comparing, and
  no stuffing gains with either encoder. Honest matching is unchanged.
- **Memory (Sprint 15, FR-LM-78):** `SentenceEncoder` keeps only the prompts of its last call,
  so a long-running server doesn't keep old or forgotten members' prompts.

## If you change it

P feeds the mutual score (stage 5). Rerun `python -m lastfirstdate.month_runs` and check that
the default population's results are unchanged unless the change is intended.

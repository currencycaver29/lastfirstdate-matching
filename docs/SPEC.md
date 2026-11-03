# lastfirstdate matching engine: spec and sprint plan

Current sprint: 0

## Goal

After a member swipes on about 50 profiles, the engine searches the member database and
presents matches scored 50% on how they look **to that member** and 50% on personality and
interests. Scores are made mutual, daily lists are spread fairly so every paying member gets
dates, and dates are scheduled into time both people have free.

## The eight stages

| Stage | What it does | Papers (see `docs/research.md`) | Sprint |
|---|---|---|---|
| 1. Filter | Removes anyone who fails a hard rule | none, plain engineering | 1 |
| 2. Calibration profiles | Chooses the ~50 profiles a new member swipes on | 1, 2, 3 | 2 |
| 3. Photo taste | Learns how much member i likes how member j looks: V_i(j) | 3, 4, 5, 6, 7 | 2 |
| 4. Personality score | How well i and j fit on interests, values and personality: P_i(j) | 8, 9 | 1 |
| 5. Mix and make mutual | 50/50 mix S(i to j), then mutual score M(i, j) | 6, 10 | 3 |
| 6. Daily list | Up to 30 profiles a day, spread fairly | 11, 12, 13, 14, 16 | 1 (basic), 4 (fair) |
| 7. Scheduling | Weekly assignment of dates into shared free slots | 15, 16 | 1 |
| 8. Learning | Retunes everything from how real dates go | 17 | 5 |

## Rules (never broken)

- **R1 Mutual orientation lock.** j is shown to i only if j's gender is one i seeks and i's
  gender is one j seeks.
- **R2 No global attractiveness score.** Looks are only ever scored from one member's point
  of view, V_i(j). Nothing stores, ranks, logs or shows anyone by overall attractiveness.
  The starting taste w_0 (Stage 3) is a starting point under a member's own swipes, never a
  ranking on its own.
- **R3 Synthetic data only in this repo.** No real photos, profiles or embeddings. Photo
  embeddings count as biometric data: in production they are encrypted, stored apart from
  the profile and deleted when a member leaves.
- **R4 Tests run offline.** No model downloads or network calls in tests. Real models
  (CLIP, Sentence-BERT) sit behind interfaces with offline stand-ins.
- **R5 Every tunable number lives in `config/lastfirstdate.toml`.**
- **R6 Dealbreakers are hard filters**, never soft score penalties.
- **R7 Same seed, same result.** All randomness goes through a seeded generator.

## Decisions (defaults in use; change any of them)

| # | Decision | Default | Why |
|---|---|---|---|
| D1 | Who chooses | Women pick from their daily list; men accept or decline. Same-gender pairs: both get lists, whoever picks first, the other accepts. | Paper 15: letting the short side initiate can raise welfare |
| D2 | Looks vs personality | 50 / 50 (`mix.looks_weight`) | Your brief. Sprint 5 can learn a personal split. |
| D3 | Daily list size | Up to 30. If fewer new eligible people exist, show fewer; never pad with people who failed the filter. | Your brief; paper 16 on caps |
| D4 | Dates per month | 3 guaranteed for paying members; nobody gets more than 4 | Your brief |
| D5 | Calibration set | 50 profiles: 20 picked for variety, 30 for uncertainty | Papers 1 and 2 |
| D6 | Personality split | 0.4 interests text, 0.3 values, 0.3 Big Five | Starting guess; paper 9 says retune from outcomes |
| D7 | Exposure cap | Automatic: no one on more than twice a fair share of lists per day | Paper 12 |
| D8 | Guarantee target | 95% of paying members with a workable pool reach 3 dates in the simulated month | Measurable version of the promise |
| D9 | Distance | Straight-line km within both people's max distance. Replaced by the original SQL query's route rule once that query is added to `db/`. | The SQL query is not in this repo yet |
| D10 | Exploration | 10% of list slots go to profiles the model is unsure about | Paper 17 |
| D11 | Test population | 1,000 synthetic NYC members, seed 7 | Fast and big enough to see effects |

## Requirements by sprint

Each requirement has an id. Every test names the id it proves, and `scripts/trace.py` fails
the checks if a requirement in a finished or current sprint has no test.

### Sprint 0: foundation

#### FR-LM-01 · Config
Sprint: 0
All weights, caps and thresholds load from `config/lastfirstdate.toml` into typed settings.
Out-of-range or misspelled values fail loudly.
Test: default config loads with the agreed defaults; bad values, missing sections and
unknown keys are rejected.

#### FR-LM-02 · Member schema
Sprint: 0
A member holds only real profile fields: gender, who they seek, age and age range, location
and max distance, verified, last active, paying, free weekly slots, interests, bio, values
answers and accepted answers, dealbreakers, Big Five, and an audit group used only for
fairness audits. No attractiveness field (R2); no photo or embedding field (R3).
Test: field names contain no score or rank words; slots round-trip.

#### FR-LM-03 · Synthetic population
Sprint: 0
Generates fake NYC members from the config, deterministic by seed (R7). Also generates
hidden ground truth (true visual taste, true looks weight, pickiness) kept apart from the
profiles, so tests can check whether the engine learns what is really there. Photo
embeddings cluster by look type; fake demographic groups lean toward look types so the
fairness audit has a bias to catch.
Test: determinism; shares match config; every field valid; clusters exist; truth is not on
profiles.

### Sprint 1: a working matcher on personality only

#### FR-LM-04 · Hard filter (Stage 1)
Sprint: 1
A pair (i, j) passes only if: both verified (when required); both active within
`filter.active_within_days`; mutual orientation lock (R1); each age inside the other's
range; distance within both people's max distance (D9); each passes the other's
dealbreakers (R6); at least `filter.min_shared_slots` shared free slots; and they have not
already dated or blocked each other. The filter is symmetric.
Test: symmetric; each rule removes its counterexample; matches a brute-force reference.

#### FR-LM-05 · Interest similarity t (Stage 4)
Sprint: 1
Cosine similarity of interest tags, weighted so rare shared interests count more than
common ones. Behind a `TextEncoder` interface so Sentence-BERT (paper 8) can replace it
later without changing callers; tests use the tag encoder only (R4).
Test: identical interests give 1, no overlap gives 0, always between 0 and 1.

#### FR-LM-06 · Values compatibility A (Stage 4)
Sprint: 1
A_ij is the share of values questions where each person's answer is one the other accepts.
Test: hand-built pairs give the expected share; symmetric.

#### FR-LM-07 · Big Five similarity p (Stage 4)
Sprint: 1
p_ij = 1 - distance between the two Big Five vectors / sqrt(5), so it runs from 0 to 1.
Test: identical gives 1, opposite corners give 0.

#### FR-LM-08 · Personality score P (Stage 4)
Sprint: 1
P_i(j) = w_text * t + w_values * A + w_big5 * p, weights from config (D6).
Test: weights applied; result between 0 and 1; symmetric.

#### FR-LM-09 · Basic daily list (Stage 6)
Sprint: 1
Each chooser (D1) gets up to `lists.daily_size` filtered candidates not shown before this
month, ranked by score, with fully booked members removed (D3, D4).
Test: size limit; no repeats; no filtered-out or booked profiles.

#### FR-LM-10 · Weekly date scheduler (Stage 7)
Sprint: 1
Each week, picks which picked-and-accepted pairs get a date and in which shared slot, as a
b-matching solved by min-cost flow with OR-Tools. Each person's capacity is their remaining
dates this month; members still below their guarantee get `dates.late_month_bonus` added
late in the month.
Test: capacities respected; only allowed pairs and shared slots used; result matches brute
force on small cases.

#### FR-LM-11 · Behavior simulator
Sprint: 1
Fake members swipe, pick, accept and rate dates using only the hidden ground truth, never
the engine's scores.
Test: deterministic; members like profiles they truly prefer more often.

#### FR-LM-12 · Month simulation report
Sprint: 1
Runs a simulated month (daily lists, picks, weekly scheduling) and reports dates per paying
member, the share reaching the guarantee, candidate pool sizes, and everyone who missed with
the reason.
Test (acceptance): at least `simulation.guarantee_target` of paying members with a workable
pool reach 3 dates (D8).

### Sprint 2: the looks model

#### FR-LM-13 · Photo embeddings (Stage 3)
Sprint: 2
A `PhotoEncoder` interface: production uses a pretrained image model such as CLIP; tests use
the synthetic embeddings (R4). Embeddings are shrunk to `calibration.pca_dim` numbers with
PCA fitted on the population. Papers 4 to 7 all start from embeddings like these.
Test: output shape; PCA keeps most of the variance on synthetic data.

#### FR-LM-14 · Starting taste w_0 (Stage 3)
Sprint: 2
One starting taste per orientation group, fitted by logistic regression on pooled swipes.
Used only as the starting point for a member's own taste (R2). Papers 4 and 5.
Test: never exposed as a ranking; improves on a coin flip for a brand-new member.

#### FR-LM-15 · Personal taste V_i(j) (Stage 3)
Sprint: 2
V_i(j) = sigmoid((w_0 + delta_i) . x_j + b_i), where delta_i is member i's personal
correction, learned from their swipes with an L2 penalty that keeps it small until they
have swiped enough. Also reports how sure it is, for Stage 2. Papers 3, 4, 5.
Test: recovers a synthetic member's hidden taste better than w_0 alone.

#### FR-LM-16 · Calibration profiles (Stage 2)
Sprint: 2
Profiles 1 to 20: the most varied set by farthest-point selection (a simple stand-in for a
determinantal point process, paper 2). Profiles 21 to 50: each next one is the profile the
model is least sure about (papers 1 and 3).
Test (acceptance): on synthetic members, taste learned from these 50 predicts held-out
likes better than taste learned from 50 random profiles.

#### FR-LM-17 · Looks report
Sprint: 2
Reports how well the looks model predicts held-out swipes, by number of swipes, and checks
it by audit group so a photo taste turning into a group filter shows up.
Test: report runs and includes the per-group breakdown.

### Sprint 3: the 50/50 mutual score

#### FR-LM-18 · Percentile scaling (Stage 5)
Sprint: 3
V and P are converted to percentiles within member i's filtered pool, so neither half
dominates just because its numbers spread wider.
Test: percentiles are uniform within a pool.

#### FR-LM-19 · 50/50 directional score S (Stage 5)
Sprint: 3
S(i to j) = looks_weight * pct(V_i(j)) + (1 - looks_weight) * pct(P_i(j)).
Test: weight applied from config.

#### FR-LM-20 · Mutual score M (Stage 5)
Sprint: 3
M(i, j) = harmonic mean of S(i to j) and S(j to i): high only when both are high. Papers 6
and 10.
Test: symmetric; zero if either side is zero; in simulation, ranking by M gives more mutual
likes than ranking by one side only.

### Sprint 4: fair daily lists

#### FR-LM-21 · Fair greedy lists (Stage 6)
Sprint: 4
Lists filled by M with an exposure cap per profile per day (D7), priority for paying members
still below their guarantee, booked members removed. Papers 12 and 16.
Test: no one exceeds the cap; guarantee share holds in simulation.

#### FR-LM-22 · Nash social welfare allocation (Stage 6)
Sprint: 4
Alternative allocation that maximizes the sum of log expected matches (paper 11), compared
in simulation against the greedy lists and against paper 13's welfare view.
Test: exposure is spread more evenly than greedy at a small cost in total matches.

#### FR-LM-23 · Exposure and fairness report
Sprint: 4
Reports how often each profile is shown, spread measures (Gini, minimum, median), and
shown and dated rates by audit group and age band.
Test: report flags a deliberately biased allocation.

#### FR-LM-24 · Hold-back experiment (Stage 6)
Sprint: 4
Optional experiment from paper 14: hold back likely matches from members who just matched
a lot. Report only; switched off by default.
Test: experiment runs and reports the match-rate difference.

### Sprint 5: learning from dates

#### FR-LM-25 · Learn from date outcomes (Stage 8)
Sprint: 5
Records show-up, "would meet again" and a private 1 to 5 rating; retrains delta_i counting
"would meet again" more than a swipe.
Test: taste accuracy improves after outcomes in simulation.

#### FR-LM-26 · Personal looks weight (Stage 8)
Sprint: 5
lambda_i = beta_V / (beta_V + beta_P) from a two-feature logistic regression on member i's
date outcomes, starting at 0.5.
Test: recovers synthetic members' hidden looks weight direction.

#### FR-LM-27 · Exploration (Stage 8)
Sprint: 5
`lists.exploration_share` of list slots go to profiles the model is least sure about.
Paper 17.
Test: share respected; learning improves versus no exploration.

#### FR-LM-28 · Three-month simulation
Sprint: 5
Three simulated months with learning against the same months without it: second-date rate,
guarantee share and exposure spread.
Test (acceptance): learning raises the second-date rate; every requirement has a test.

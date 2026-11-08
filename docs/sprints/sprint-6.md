# Sprint 6: lastfirstdate's real rules, the forecast and the ratio gate

**In one paragraph.** Sprints 0 to 5 built a general matching engine for a city-wide app.
Your website describes something more specific: every man pays $200 a month for 3 guaranteed
dates, women join free, nobody has more than 3 dates a cycle, Phase 1 is one NYC zone with
about 300 committed women before 100 men are sold subscriptions, and "if the ratio isn't there,
male sales pause". Sprint 6 makes the engine match those rules and turns the last sentence into
software. It adds a second settings file with the Phase 1 rules, the website's single-answer
intent questions, free-text prompts, and "Who are you more drawn to?" pair questions. Most
importantly, it adds a **forecast** of each man's chance of reaching 3 dates and a **ratio
gate** that admits men from the waitlist only while the forecast stays at 90% or above. In
test runs the gate works: with more men waiting than there are women, the men it admits reach
3 dates **92% to 96%** of the time. Admitting everyone would give **12.5%**. Along the way the
first version of the forecast failed badly in a way worth understanding (section 9).

**Ideas used**

| Idea | What we took from it | Where in the code |
|---|---|---|
| Learning from choices | Learning taste from "A or B?" choices, and asking the most informative pair next | `looks.PersonalTaste.add_pair`, `looks.select_pair` |
| Pairwise preferences | A pair answer is evidence about the *difference* between two items, learned in the same model as single answers | `looks.PersonalTaste.add_pair` |
| Sentence embeddings | Free text becomes a vector so similar text scores as similar. Sprint 6 uses an offline word-count stand-in behind the same interface | `personality.WordEncoder` |
| Market balance | The short side's capacity limits everyone | Why the gate watches women per man |
| Caps cut congestion | Limiting how many people compete for the same partners raises the share who succeed | The gate limits men per zone, the way caps limit applications |

The forecast and the gate themselves are ordinary logistic regression applied to your business
rule. Research on matching markets explains *why* the ratio matters so much; the test run
measures *how much*.

---

## 1. What got built

| File | What it does |
|---|---|
| `config/phase1.toml` | Second settings profile: your Phase 1 rules (section 2) |
| `config/lastfirstdate.toml` | Research defaults, unchanged in effect. It gained the settings that moved out of the code (city areas, prompt style) and the new `[gate]` section |
| `config/forecast_model.toml` | The fitted forecast: coefficients and the range of women per man they were fitted on |
| `forecast.py` | Engine: market features, the forecast, at-risk alerts, the ratio gate |
| `forecast_fit.py` | Test side: builds 42 synthetic markets, fits the forecast, tests it on 8 others, saves it |
| `phase1_report.py` | Test side: pair questions versus singles, and the gate in action |
| `synthetic.py` | Fake members can now follow the Phase 1 rules: exact head counts, one zone, single-answer intent questions, prompt-style bios |
| `filter.py`, `personality.py`, `member_behavior.py` | Single-answer questions (closeness), the word encoder, and synthetic answers to pair questions |
| `looks.py` | Pair questions in the taste model and in calibration |
| `month_runs.py` | Optionally records each man's dates and waiting yeses at the end of every week (for at-risk alerts) |

With the default settings file, every earlier result is identical. The fake population and the
month report were checked line by line before and after (`test_default_population_is_unchanged_by_sprint_6`).

---

## 2. Two settings profiles (D14, FR-LM-29)

The research defaults model a whole city: all orientations, 55% men, some free members of every
gender, 50 calibration photos. Changing those numbers in place would have broken the
comparisons in Sprints 1 to 5. So there are now two files:

| | `lastfirstdate.toml` (research) | `phase1.toml` (your Phase 1) |
|---|---|---|
| Members | 1,000 across 9 NYC areas | 400 in one zone: exactly 100 men and 300 women |
| Orientation | all | man-woman only |
| Who pays | some of everyone | every man; no woman |
| Date cap | 4 a month | 3 for everyone |
| Calibration | 50 single photos | 30 photos: 12 singles and 9 pairs |
| Values questions | "which answers do you accept" | single answer, scored by closeness (section 3) |
| Free text | interest tags | prompts ("A perfect Sunday looks like…"), read word by word (section 4) |
| Personality mix | text 0.40, values 0.30, Big Five 0.30 | text 0.55, values 0.45 (the site has no personality test) |

Any report can run on either file: `load("config/phase1.toml")`.

---

## 3. Single-answer intent questions (FR-LM-30)

Your onboarding asks things like "When do you want commitment?" with one answer:
Now / Within a year / No timeline. The old engine asked "which answers would you accept?",
which your site doesn't. A question can now be in **closeness** mode:

    compatibility = 1 − |rank_i − rank_j| / (number of options − 1)

With three ordered options, the same answer scores 1, neighbours score 0.5, and opposite ends
score 0. Some pairs should never meet at all. "Want kids" with "Not for me" is listed as
**incompatible**, so the Stage 1 filter removes that pair for everyone (rule R6: dealbreakers are
hard filters). The values score A is the average closeness over the questions, as before.

---

## 4. Free-text prompts (FR-LM-31)

Your profiles use written prompts, not interest tags. The production plan is a sentence model,
but tests must run offline with no model downloads (rule R4). So Sprint 6 adds a
stand-in, `WordEncoder`, using **TF-IDF**:

1. Split each bio into words and drop common words ("the", "and", "looks").
2. For each word, count how often it appears in this bio (TF) and how rare it is across all
   bios (IDF = log of members / members using the word). Rare shared words like "climbing"
   count much more than common ones.
3. Each bio becomes a vector of TF × IDF, scaled to length 1. The similarity of two bios is
   the dot product of their vectors (cosine similarity).

The setting `personality.text_encoder = "words"` selects it. Swapping in a sentence model later
changes one class, not the engine.

---

## 5. Pair questions in calibration (FR-LM-32)

### The math

A single swipe teaches the model "x_j was liked or not":
P(like j) = sigmoid(w · x_j), where x_j is photo j's features and w is the member's taste
(w_0 + delta from Sprint 2).

A pair question "Who are you more drawn to, A or B?" is a **Bradley–Terry** comparison:

    P(A over B) = sigmoid(w · x_A − w · x_B) = sigmoid(w · (x_A − x_B))

So a pair is simply one more example for the same logistic model, with the feature vector
x_A − x_B and the label "chose A". That is why it fits into the existing taste model with one
line (`add_pair` calls `add_label` on the difference). Preference-learning methods learn from
exactly this kind of data.

**Which pair to ask?** Among the 30 profiles the model is least sure about, take the pair whose
answer is most informative: the largest log(1 + p(1 − p) · dᵀΣd), where d = x_A − x_B, p is the
predicted chance of choosing A, and Σ is the model's uncertainty. The idea matches Sprint 2's
singles: ask where the model is unsure *and* the answer would move it most.

### What the test run showed

Same members, same answers to the same photos (common random numbers), and accuracy measured on
photos no version showed:

| Version | Answers | AUC | Match with true taste |
|---|---:|---:|---:|
| 50 single photos (research default) | 50 | 0.850 | 0.909 |
| 30 single photos | 30 | 0.838 | 0.888 |
| 30 photos: 12 singles + 9 pairs (site demo) | 21 | 0.818 | 0.857 |

At a fixed 30 photos, pairs lose 0.019 AUC (standard error 0.003). But 9 pairs use 18 photos for
only 9 answers. Counted **per answer**, pairs and singles teach the same amount:

| Comparison | Pairs | Singles | Difference |
|---|---:|---:|---:|
| 21 answers | 0.817 | 0.818 | −0.000 (se 0.003) |
| 30 answers | 0.833 | 0.838 | −0.004 (se 0.003) |

**Decision D15:** count calibration length in answers, not photos. If you want pairs, ask about
as many questions as you would have asked singles.

**What the test run can't show.** People's yes/no threshold drifts (tired, picky today); a
choice between two photos is immune to that drift. Our fake people have a fixed threshold, so
pairs' main real-world advantage isn't modelled. An A/B test at launch should settle it
(`docs/risk-plan.md`, section 4).

---

## 6. The guarantee forecast (FR-LM-33)

### What it predicts

For each paying man j, at the start of a cycle: the chance he reaches 3 dates.

    P_j = sigmoid(b0 + b1·log r + b2·log(1 + pool_j) + b3·log(1 + share_j) + b4·log(min(r, 1)))

| Feature | Plain words | Why |
|---|---|---|
| r, women per man | committed women ÷ paying men in the zone | the market balance |
| pool_j | how many women he *could* be matched with (passes the filter both ways) | more options, more chances of mutual yeses |
| share_j | his expected share of women's attention: for each woman in his pool, 1 ÷ (number of men she could be matched with), summed | a woman with 3 possible men gives him more of her 3 dates than one with 60 |
| log(min(r, 1)) | 0 while there's at least one woman per man; negative below that | below 1:1 the women's date caps run out and success collapses (section 9) |

Pool size depends only on the member's own settings and location, never on how often he is
picked, so rule R2 (no attractiveness score) holds.

**At-risk alerts** use a second model mid-cycle, adding three features: dates so far ÷ 3,
log(1 + yeses waiting), and weeks left ÷ 4. Men below a 50% chance (`gate.at_risk_threshold`)
are flagged, so staff can step in (extra list priority, a concierge message, an event invite)
before the cycle ends.

### How it was fitted

Before launch there are no real cycles, so the starting coefficients come from **42 synthetic
markets** with Phase 1 rules. Each combines 60, 100 or 150 men with 0.5, 0.75, 1, 1.5, 2, 3 or 5
women per man, on two seeds, run for 2 cycles each. Every paying man in every cycle is one
row: features, plus whether he reached 3 dates. That gives about 8,200 start rows and 24,700
mid-cycle rows (one per man per week). Logistic regression with a small penalty (`gate.forecast_l2`) fits them, the same
`fit_logistic` as the looks model. Once real cycles exist, the same function is rerun on real
rows.

The fitted start coefficients:

| | intercept | log r | log(1+pool) | log(1+share) | log(min(r,1)) |
|---|---:|---:|---:|---:|---:|
| b | −5.944 | 0.281 | 1.438 | 1.891 | 2.810 |

**A worked example** from a synthetic 3:1 market. A man with 126 possible matches and attention
share 2.58:

    −5.944 + 0.281·log 3 + 1.438·log 127 + 1.891·log 3.58 + 2.81·0
    = −5.944 + 0.309 + 6.966 + 2.411 = 3.74   →   sigmoid(3.74) = 97.7%

A man in the same market with only 36 possible matches (share 1.82) gets a logit of 1.52, which
is **82%**. Same market, different chances, because of his own filters. Averaged over all men,
the forecast for that market was 92.3%; 90.6% actually reached 3 dates.

### Tested on markets it never saw

Eight held-out markets (different seed and sizes):

| Men | Women per man | Forecast | Actual, cycle 1 | Actual, cycle 2 |
|---:|---:|---:|---:|---:|
| 100 | 0.58 | 9.9% | 10.3% | 17.5% |
| 100 | 1.23 | 71.4% | 76.8% | 81.1% |
| 100 | 2.44 | 93.2% | 92.9% | 89.8% |
| 100 | 3.98 | 98.3% | 100.0% | 97.9% |
| 150 | 0.59 | 16.4% | 13.9% | 9.0% |
| 150 | 1.24 | 80.8% | 85.5% | 85.5% |
| 150 | 2.58 | 96.6% | 97.9% | 97.9% |
| 150 | 4.11 | 99.2% | 100.0% | 100.0% |

The market forecast is off by **3.3 points on average** (largest 9.7). Around 1.2 women per man it
leans cautious, the safe direction for a gate. Ranking individual men (AUC: 1.0 is perfect,
0.5 is a coin flip):

| When | AUC |
|---|---:|
| start of cycle | 0.959 |
| after week 1 | 0.983 |
| after week 2 | 0.991 |
| after week 3 | 0.998 |

The at-risk model gets sharper every week, as expected: by week 3 it mostly knows.

---

## 7. The ratio gate (FR-LM-34)

### The rule

Applicants on the men's waitlist are considered **in order** (first come, first served). For each:

1. If his pool is below `gate.min_pool` (10), he is **deferred** rather than sold a guarantee he
   would likely miss. The next applicant is considered.
2. Add him to a trial market. If women per man would fall below `gate.min_women_per_man` (3),
   **stop**.
3. If women per man would fall below the range the forecast was fitted on, **stop**. The gate
   never trusts the forecast where it has no data (section 9).
4. If the forecast average fulfilment would fall below `gate.target` (90%), **stop**.
5. Otherwise admit him and continue.

The result says how many to admit, who was deferred, the resulting ratio and forecast, and why
it stopped. All three thresholds live in `config/phase1.toml`.

### What the test run showed

Each market was run on synthetic members for two cycles with only the admitted men in it, on three seeds. Three
versions: the gate as set, the forecast alone (no 3:1 floor), and admitting everyone.

**Waitlist smaller than the women** (about 285 women, 187 men waiting):

| Version | Men admitted | Women per man | Forecast | Actual |
|---|---:|---:|---:|---:|
| Gate as set | 95 | 3.0 | 94.8% | 93.9% |
| Forecast only | 155 | 1.8 | 90.2% | 91.8% |
| Everyone | 187 | 1.5 | 87.9% | 89.9% |

**Waitlist larger than the women** (about 285 women, 429 men waiting):

| Version | Men admitted | Women per man | Forecast | Actual |
|---|---:|---:|---:|---:|
| Gate as set | 94 | 3.0 | 95.0% | 96.3% |
| Forecast only | 174 | 1.6 | 90.1% | 92.3% |
| Everyone | 429 | 0.66 | 57.2% | **12.5%** |

What this means:

- **The gate does its job.** When men outnumber women, admitting everyone would have sold 429
  guarantees that almost nobody gets. The gate sold 94 and kept them above target.
- **The forecast is well calibrated where the gate decides** (near 90%): forecast and actual
  are within about 2 points, and the forecast errs on the cautious side.
- **The 3:1 floor is cautious in a zone this size.** The forecast alone admits about 1.8 times
  as many men and still reaches 92%. With about 290 women, men have larger pools, so they need
  fewer women each. The floor is a safety margin against the test run being wrong about real
  people. Whether to keep it at launch is your call (questions below).
- **For women,** at 3:1 the average woman can have at most about 1 date a cycle (each man needs
  3, three women share him). That is why "commit to three dates" should read "up to three".

---

## 8. What changed in the plan

- **D14:** two settings profiles, research and Phase 1.
- **D15:** calibration length is counted in answers, not photos.
- **The forecast gained a fifth feature** (the shortfall below one woman per man) and records
  its fitted range; the gate gained the matching stop (step 3).
- **Deferred to Sprint 7**, because they need the app's database: the proposed date window and
  15-minute payment hold, per-member billing cycles and refills after cancellations, the three
  rating dimensions and a behaviour-only trust score, event dates, and neighbourhood geography
  with venue midpoints.

---

## 9. A lesson: a model is only right where it has seen data

The first forecast was fitted on markets with 1 to 5 women per man. On held-out markets in that
range it looked excellent (79.7% versus 76.8% actual at 1.5:1; 93.5% versus 93.8% at 3:1). Then
the gate trial gave it a waitlist of 432 men for 278 women (0.64 per man). It forecast **83%**.
In the test run, **7.4%** of men reached 3 dates.

The reason is arithmetic. 278 women × 3 dates is 834 dates. 432 men × 3 is 1,296. Most men
*cannot* reach 3, however good the matching. And the scheduler spreads the dates out, so it's
far worse than the 64% that perfect packing would allow. A logistic curve fitted only above 1:1
had no way to know about this cliff. It just extended the gentle slope it had seen.

The fix had three parts:

1. **Show it the cliff:** add markets at 0.5 and 0.75 women per man, and 150-man markets, to
   training.
2. **Give it a way to bend:** the term log(min(r, 1)) is zero above 1:1 and lets the curve drop
   steeply below. Five versions were compared on the same held-out markets:

   | Version | Mean error | Largest error |
   |---|---:|---:|
   | old features, trained on 1 to 5 | 8.4 pts | 37.1 pts |
   | old features, trained on 0.5 to 5 | 5.2 pts | 14.8 pts |
   | + shortfall in attention share | 4.6 pts | 13.3 pts |
   | + both shortfall terms | 3.2 pts | 9.4 pts |
   | **+ shortfall in women per man (chosen)** | **3.3 pts** | **9.7 pts** |

   The chosen version is as good as the two-term one and simpler.
3. **Never extrapolate:** the model file records the range it was fitted on (0.47 to 5.25 women
   per man), and the gate stops before leaving it.

One miss remains, kept as a strict expected failure: far below 1:1 in markets *larger* than any
it was fitted on, it is still too optimistic (57% forecast versus 12.5% actual for 429 men). It
doesn't affect the gate, which stops at 90% long before. But refit on larger markets, or on real
cycles, before using it to answer "what if we admitted everyone?"

The general lesson for the real app: **every model should carry the range of data it was fitted
on, and anything that acts on it should refuse to act outside that range.**

---

## 10. Honest limits

- **Everything is run on synthetic members.** Fake people's yes rates, show-up rates and taste decide how many
  women per man are enough. The real number could be higher or lower. Refit the forecast after
  the first real cycles; until then the 3:1 floor is the safety margin.
- **The forecast doesn't know about seasons, events, or marketing pushes**, only the market's
  shape. Real data will add features.
- **Pair questions' real advantage** (immune to threshold drift) isn't modelled.
- **TF-IDF reads words, not meaning:** "rock climbing" and "bouldering" don't match. A sentence model
  will fix that in production.
- **One zone.** Phase 2's multiple zones would need the gate per zone, with members' travel
  limits deciding who belongs where.

---

## 11. Math to study

1. **Bradley–Terry model:** why P(A over B) depends only on the difference of scores, and why
   that makes a pair one logistic example on x_A − x_B.
2. **TF-IDF and cosine similarity:** why dividing by how common a word is helps, and why vectors
   are scaled to length 1.
3. **Logistic regression as a forecast:** reading coefficients (each +1 in a feature multiplies
   the odds by e^b), and why log features suit "more helps, but less and less".
4. **Calibration versus ranking:** AUC says whether the model orders men correctly; the market
   table says whether its percentages are right. A gate needs both.
5. **Hinge features:** log(min(r, 1)) lets one straight-line model have two slopes.
6. **Extrapolation:** why a model fitted on one range can be confidently wrong outside it.

---

## 12. Try it yourself

```bash
.venv/bin/python -m lastfirstdate.phase1_report            # calibration and both gate scenarios (~5 min)
.venv/bin/python -m lastfirstdate.phase1_report --skip-calibration   # gate only (~4 min)
.venv/bin/python -m lastfirstdate.forecast_fit --no-save   # refit and test without saving (~10 min)
.venv/bin/python -m pytest tests/test_phase1.py -q         # Sprint 6's tests
```

Things to try: change `[gate] min_women_per_man` or `target` in `config/phase1.toml` and rerun
the gate report; change `n_pairs` under `[calibration]` and watch the calibration table.

---

## Questions for you

1. **The 3:1 floor.** Keep it at launch as a safety margin (my recommendation until real cycles
   exist), or let the forecast alone decide? In test runs the forecast alone would sell about
   1.8 times as many subscriptions at 92% fulfilment.
2. **The target.** 90% matches your north star. A higher target sells fewer subscriptions but
   costs fewer credits.
3. **Calibration.** Keep the demo's 30 photos with 9 pairs (21 answers, AUC 0.818), or ask about
   30 *answers* to keep accuracy: 30 singles (AUC 0.838) or 12 singles and 18 pairs (48 photos,
   AUC 0.833)?
4. **The women's wording.** "Up to three dates a month" instead of "commit to three".
5. **Sprint 7** needs the app's database for booking, billing, trust and geography. Is there an
   app repo or schema yet, or should Sprint 7 define one for the app to follow?

# Sprint 0: the foundation

**In one paragraph.** Sprint 0 builds no matching yet. It builds the world the matcher will be
tested in: a settings file holding every number we might want to tune, a member profile with
exactly the fields a real lastfirstdate profile has, a generator that creates 1,000 fake New
York members, and the checks ("gates") that every later sprint must pass. The fake members
also carry hidden ground truth, such as each person's true visual taste, so later sprints
can measure whether the algorithm actually learns what is really there.

**Papers used:** none of the 17 methods is built yet. One idea from paper 9 shaped the fake
data: part of attraction is shared by most people and part is personal (explained in
Concept 5).

---

## 1. What got built

| File | What it does |
|---|---|
| `config/lastfirstdate.toml` | Every weight, cap and threshold: 50/50 split, 30 a day, 3 dates a month, the 50 calibration profiles, and the fake population's shape |
| `src/lastfirstdate/config.py` | Reads that file and refuses to run if a value is missing, misspelled or out of range |
| `src/lastfirstdate/models.py` | The `Member` profile and the 21 weekly time slots |
| `src/lastfirstdate/synthetic.py` | Generates fake members, fake photo embeddings and hidden ground truth |
| `tests/` | 24 tests, each naming the requirement it proves |
| `scripts/check.sh` | The gates: lint, formatting, tests, traceability |
| `scripts/trace.py` | Fails if any requirement due by now has no test |
| `docs/SPEC.md` | The sprint plan: 8 stages, 7 rules, 11 decisions, 28 requirements |
| `docs/research.md` | The 17 papers, mapped to stages, requirements and sprints |

## 2. Why start with fake members?

Three reasons, and the third matters most.

1. **Privacy.** Real photos and photo embeddings are biometric data (rule R3). They never go in
   a code repo.
2. **No real data exists yet.** Learning from dates needs hundreds of completed dates.
3. **Fake members come with an answer key.** With real people you never know someone's true
   taste; you only see what they swipe. With fake members we *choose* their true taste, hide
   it from the engine, and then check whether the engine recovers it. That is the only way to
   prove the looks model works before launch.

This is standard practice in recommender research: build a simulator with known ground truth,
then measure how close the algorithm gets.

## 3. Concept: one settings file for every number

Every tunable number sits in `config/lastfirstdate.toml` (rule R5). For example:

```toml
[mix]
looks_weight = 0.5        # the 50/50 split

[lists]
daily_size = 30

[dates]
guaranteed_per_month_paying = 3
```

Why it matters: you will want to try 60/40, or 20 profiles a day, or a different personality
split. That should be a one-line change, not a code change. `config.py` also checks the file:
personality weights must add up to 1, shares must sit between 0 and 1, the 20 "variety"
profiles cannot exceed the 50 calibration profiles, and so on. A typo such as `daily_sise`
stops the program instead of being silently ignored.

## 4. Concept: the member profile

`Member` in `models.py` holds:

| Field | Example | Used in |
|---|---|---|
| `gender`, `seeking` | woman, {man} | Stage 1: mutual orientation lock (R1) |
| `age`, `age_min_pref`, `age_max_pref` | 29, 26 to 36 | Stage 1: each must fit the other's range |
| `lat`, `lon`, `max_km` | Astoria, 12 km | Stage 1: distance |
| `verified`, `last_active_days`, `paying` | yes, 2 days, yes | Stage 1, and the 3-date guarantee |
| `free_slots` | {fri evening, sat lunch, ...} | Stage 1 and Stage 7: dates need a shared free slot |
| `interests`, `bio` | hiking, jazz, coffee | Stage 4: interest similarity |
| `values`, `accepts`, `dealbreakers` | kids: want; accepts want or open | Stage 4 score, and Stage 1 dealbreakers |
| `big5` | 5 numbers from 0 to 1 | Stage 4: personality similarity |
| `audit_group` | G3 | Fairness audits only, never a matching input |

**What is deliberately missing:**

- **No attractiveness, desirability, Elo or rating field (rule R2).** Your homepage promises
  no global attractiveness score. If a field like that existed, someone would eventually sort
  by it. A test fails if any field name contains words like "score", "rank" or "attract".
- **No photo or embedding on the profile (rule R3).** Embeddings live in a separate matrix,
  so in production they can be encrypted on their own and deleted when a member leaves.

**Time slots.** A week is split into 21 slots: 7 days by lunch, evening and late. Slot 13 is
"fri evening". Two people can only be scheduled into a slot both have free.

## 5. Concept: how the fake members are generated

Each field is drawn from a probability distribution. Here is what each one is and why it was
chosen:

| Field | How it is drawn | In plain words |
|---|---|---|
| Gender | 55% men | Dating apps usually have more men; this makes women the short side, which is why "she chooses" (D1) |
| Orientation | 90% straight, 7% gay or lesbian, 3% bi | Set in config |
| Age | 21 + Beta(2, 3) x 24 | The Beta distribution produces a number between 0 and 1. Beta(2, 3) leans toward the lower end, so most members are in their late 20s and early 30s (median 30) |
| Age range | Own age minus 2 to 8 years, plus 2 to 8 years | Men skew a little younger, women a little older: a starting assumption you can change |
| Location | One of 9 areas (Midtown, North Brooklyn, Astoria, Jersey City, ...) plus about 1.5 km of random spread | Normal distribution around each area's center |
| Max distance | Anywhere from 5 to 25 km | Uniform: every value equally likely |
| Last active | 85% active within 2 weeks; the rest 15 to 90 days ago | Exponential distribution for the active ones: most were active in the last few days |
| Free slots | 4 to 10 of the 21, with evenings and weekends more likely | Weighted random choice |
| Interests | 5 to 10 tags, 70% from one lifestyle group (outdoors, arts, food, nightlife, fitness, quiet, tech, travel) | People's interests cluster, so similarity is meaningful |
| Values answers | Each question's answer shares come from a Dirichlet draw | A Dirichlet draw produces a set of shares that add up to 1, so each question gets a realistic, uneven split |
| Big Five | Beta(2.5, 2.5) per trait | Bell-shaped between 0 and 1, centered at 0.5 |

**Same seed, same members (rule R7).** All randomness goes through one random number
generator started from a seed (default 7). Run it twice with seed 7 and you get the same
1,000 people, so every test and simulation can be repeated exactly.

The population with seed 7:

```
1000 members: 554 men, 446 women
paying: 283 men, 146 women
straight 903, gay or lesbian 66, bi 31
verified 922, active in last 14 days 849
age median 30 (range 21-44), free weekly slots median 7
```

## 6. Concept: photos as embeddings

The engine never looks at a photo directly. A pretrained image model such as CLIP turns each
photo into an **embedding**: a list of numbers (512 for CLIP; 128 in our fake data) where
similar-looking photos get similar lists. Papers 4 to 7 all work this way.

**Unit vectors.** Every embedding is scaled to length 1:

```
length(e) = sqrt(e_1^2 + e_2^2 + ... + e_d^2) = 1
```

**Cosine similarity** measures how close two embeddings point:

```
cos(a, b) = a . b = a_1*b_1 + a_2*b_2 + ... + a_d*b_d     (for unit vectors)
```

It is 1 when they point the same way, 0 when they are unrelated, and -1 when opposite.
Tiny example in 2 numbers: a = (0.6, 0.8), b = (0.8, 0.6). Then a . b = 0.48 + 0.48 = 0.96:
very similar.

**How the fake embeddings are made.** There are 8 "look types". Each fake photo is:

```
photo = (center of its look type) + (random personal variation) + (shared appeal) + (age shift)
```

then scaled back to length 1. Measured result: photos of the same look type have cosine
**0.51** with each other; photos of different types have **0.00**. That is roughly how a real
image model behaves: similar faces are close, unrelated faces are not.

## 7. Concept: hidden ground truth

Each fake member gets a true visual taste, a unit vector u_i in the same space as the photos.
How much member i is drawn to member j's looks is then:

```
looks appeal of j to i  =  u_i . e_j
```

The taste is a mix of two parts:

```
u_i = normalize( rho_i * s  +  (1 - rho_i) * personal_i ),   rho_i between 0.2 and 0.6
```

- **s** is a direction most people find appealing (shared taste).
- **personal_i** points toward the look type this member personally prefers.

Measured on seed 7: **39% of the variation in looks appeal is shared** (everyone agrees) and
**60% is personal**.

**Link to paper 9 (Joel, Eastwick and Finkel).** Their speed-dating study found that general
desirability (who most people like) was predictable, but desire for one specific person was
not, even with 100+ traits measured. Our fake world has both parts on purpose. It matters for
Sprint 2: the starting taste w_0 will learn the shared part, and only a member's own swipes can
learn the personal part. The 39 / 60 split is our assumption, not a number from the paper,
and it can be changed.

Other hidden traits: `looks_weight` (how much each fake member truly cares about looks
versus personality, around 0.5 on average), `pickiness`, `look_type` and `lifestyle`.

**The engine must never read any of this.** Only the simulator may read it. A test checks that
none of these fields exist on the `Member` profile.

## 8. Concept: a fairness trap, built in on purpose

The research doc warns that photo taste can quietly become an ethnicity or age filter, because
image embeddings encode both. To make sure our fairness audit (Sprint 4) can catch that, the
fake data contains the problem on purpose: each of the 5 fake demographic groups (G1 to G5)
leans toward two look types, and age shifts the embedding. If the looks model ends up
showing some groups much less often, the audit must flag it. A test confirms the groups really
do lean toward their look types.

## 9. Concept: values questions and dealbreakers

Six questions: kids, relationship goal, smoking, drinking, religion importance, timeline. As on
OkCupid, each member gives **their answer** and **the answers they accept** from a partner.

- In Stage 4, A_ij will be the share of questions where each person's answer is one the other
  accepts.
- A question marked as a **dealbreaker** becomes a hard filter in Stage 1 (rule R6): if the other
  person's answer isn't accepted, they are never shown.

Answer options are listed in order in the config, so neighboring answers count as "close". A
member who marks drinking as a dealbreaker and answers "socially" may still accept "never".

## 10. Concept: the gates

`./scripts/check.sh` runs four checks. A sprint is not done until all four pass:

1. **Lint (ruff):** catches bugs and sloppy code patterns.
2. **Format (ruff):** one consistent code style.
3. **Tests (pytest):** 24 tests now.
4. **Traceability:** every requirement due by the current sprint must be named by a test. It
   prints which test file proves each one:

```
Current sprint 0: 3 requirements due, 28 in spec
  FR-LM-01  test_config.py
  FR-LM-02  test_models.py
  FR-LM-03  test_synthetic.py
```

These are the parts of claude-company worth keeping (no merge while tests fail, every
requirement has a test), without its cost.

## 11. What went wrong, and how we fixed it

Two problems surfaced while building. Both were in the fake data, not the plan.

1. **Dealbreakers were far too strict.** First version: each member had about 1.5
   dealbreakers, each accepting only their own answer. Only **17%** of pairs survived, and the
   median member had just 13 possible matches. Real people are more flexible, so now a
   question is a dealbreaker 12% of the time (0.73 per member on average) and a dealbreaker
   still accepts neighboring answers half the time. Now **58%** of pairs pass.
2. **Fake photos were mostly noise.** A test checks that same-type photos are clearly closer
   than different-type photos. It failed: the random part was about 4 times longer than the
   look-type signal, so the cosine was 0.07 versus 0.00. After rescaling it is 0.51 versus
   0.00.

Lesson: the tests on the fake data itself caught both. If the fake world is wrong, every later
result is wrong.

## 12. What the numbers already tell us

After the hard filters (age, distance, dealbreakers, shared slots), among the 780 verified,
active members:

| Measure | Value |
|---|---|
| Pairs passing the age rule | 56% |
| Passing distance | 68% |
| Passing dealbreakers | 58% |
| Sharing at least one free slot | 95% |
| Median number of possible matches per member | 62 |
| Members with fewer than 30 possible matches | 27% |
| Members with none | 3 |

**Why this matters.** A daily list of 30 new people means about 900 new profiles a month, but
the typical member in a 1,000-member city only has about 62 possible matches. After 2 or 3
days they would run out of new people. So either:

- lists shrink after the first days (show only newcomers), or
- people who were shown but not picked can come back later, or
- the real member base is much larger than 1,000.

Sprint 1's simulation will measure this properly. The answer depends on how many members you
expect at launch in NYC (see the questions at the end).

## 13. Math to study for this sprint

From *Mathematics for Machine Learning* (Deisenroth, Faisal and Ong, free at
[mml-book.github.io](https://mml-book.github.io)):

- **Chapter 2 (linear algebra):** vectors, adding and scaling vectors.
- **Chapter 3 (analytic geometry):** dot product, length (norm), angles and cosine similarity.
  This is Concept 6.
- **Chapter 6 (probability):** what a distribution is, the normal distribution, sampling.
  This is Concept 5.

Try it yourself (after setting up as in the README):

```bash
.venv/bin/python -m lastfirstdate.synthetic --seed 7
.venv/bin/python -m lastfirstdate.synthetic --seed 8 --n 200
```

Then in Python:

```python
from lastfirstdate.synthetic import generate
pop = generate(seed=7)
m = pop.members[0]
print(m.gender, m.age, m.interests, m.values, m.dealbreakers)
e = pop.photo_embeddings
print(e[0] @ e[1])   # cosine similarity of member 0's and member 1's photos
```

## 14. Read before Sprint 1

| Paper | What to look for |
|---|---|
| 15. Kanoria and Saban, *Facilitating the Search for Partners* | Why letting only one side initiate (she chooses) can make everyone better off. This justifies decision D1. |
| 16. Arnosti, Johari and Kanoria, *Managing Congestion in Matching Markets* | Why capping how many people someone can contact cuts wasted effort. This justifies the 30-a-day and 3-a-month caps. |
| 9. Joel, Eastwick and Finkel, *Is Romantic Desire Predictable?* | Why the personality weights (0.4 / 0.3 / 0.3) are only a starting guess. |
| 8. Reimers and Gurevych, *Sentence-BERT* (abstract and section 3) | What will replace simple interest tags: turning text into vectors so "trail runs" and "hiking" count as similar. |

## 15. Questions for you

1. **How many members do you expect in NYC at launch, and after 6 months?** This decides
   whether 30 new profiles a day is realistic (section 12).
2. **Can you paste the original lastfirstdate pre-filter SQL query** from the earlier session?
   It isn't in this repo yet. Until then, Stage 1 uses straight-line distance (decision D9).
3. Any default in the decisions table in `docs/SPEC.md` you want to change.

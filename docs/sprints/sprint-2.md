# Sprint 2: the looks model

**In one paragraph.** Sprint 2 builds the looks half of the score. Each member's photo
taste is learned from about 50 swipes: a starting taste shared by their orientation group,
plus a personal correction learned from their own swipes. It also chooses *which* 50
profiles a new member sees: 20 picked to cover every kind of look, then 30 picked because
each teaches the model the most. On 377 fake members it predicts held-out likes with an
**AUC of 0.844**, against a ceiling of 0.875 (the best possible score given how noisy
swipes are). The starting taste alone scores 0.750, and a coin flip 0.5. Choosing the 50
cleverly beats choosing them at random, but only a little: +0.009 AUC, or about 10
swipes saved.

**Papers used**

| Paper | What we took from it | Where in the code |
|---|---|---|
| 4. Ren et al., *Personalized Image Aesthetics* | The core design: a general model plus a small personal correction, so a few ratings are enough | `looks.StartingTaste`, `looks.PersonalTaste` |
| 5. Rothe, Timofte and Van Gool, *Some Like It Hot* | Predicting one specific person's attraction from image features | `PersonalTaste.like_chance` |
| 3. Chu and Ghahramani, *Preference Learning with Gaussian Processes* | A Bayesian taste model that reports how sure it is. Ours is the simplest version: a linear model with a Laplace approximation | `looks.fit_logistic`, `PersonalTaste.uncertainty` |
| 1. Brochu, de Freitas and Ghosh, *Active Preference Learning* | The principle: show next whatever the model can learn most from. Their exact method (comparing two items at a time) is not used | `PersonalTaste.information` |
| 2. Kulesza and Taskar, *Determinantal Point Processes* | Greedy DPP selection for the 20 varied profiles | `looks.select_varied` |
| 6 and 7. Neve and McConville | Evidence that photo embeddings alone carry strong preference signal. Their mutual part is Sprint 3 | `looks.PhotoEncoder` |

Not used from these papers: paper 5's collaborative-filtering component, paper 7's
recurrent network, and paper 1's pairwise-comparison setup.

---

## 1. What got built

| File | What it does |
|---|---|
| `looks.py` | The engine side: photo encoder interface, PCA, logistic regression, starting taste, personal taste, calibration pool, varied and informative selection |
| `looks_report.py` | The test run side: trains, calibrates, scores and audits on fake members |
| `member_behavior.py` | New: a photo-only swipe decision |
| `tests/test_looks.py` | 20 tests for FR-LM-13 to FR-LM-17 |

## 2. The big picture

```
photo --image model--> 128 numbers --PCA--> 64 numbers + 1 = x_j

new member signs up
  1. show 20 profiles chosen for variety          (Stage 2, paper 2)
  2. after each swipe, refit their taste          (Stage 3, papers 3, 4, 5)
  3. show 30 more, each the most informative one  (Stage 2, paper 1)
  4. now V_i(j) = sigmoid((w_0 + delta_i) . x_j) can score anyone
```

## 3. Photo embeddings and PCA (FR-LM-13)

An image model turns each photo into a list of numbers (Sprint 0). Production would use a
model such as CLIP (512 numbers). Our fake photos have 128. Behind the `PhotoEncoder`
interface, a real model can replace the fake one without changing anything else.

**Why shrink to 64?** The taste model learns one weight per number. With 50 swipes,
learning 128 weights is asking too much of too little data. PCA (principal component
analysis) finds the directions in which photos differ most and keeps only the top 64:

```
x_j = (e_j - average photo) projected onto the top 64 directions
```

Those directions are the eigenvectors of the photos' covariance matrix, found with an SVD.
On our fake photos the top 64 keep **83.8%** of the variation, and **99.6%** of the hidden
taste signal survives. The thrown-away directions are mostly noise. A constant 1 is added to
each x_j so the model also has an intercept, giving 65 numbers in all.

## 4. The taste model (FR-LM-15)

### Logistic regression in one minute

The model gives each candidate a score called a **logit**, z = w . x_j, a weighted sum of
their photo numbers. The **sigmoid** turns any number into a probability:

```
sigmoid(z) = 1 / (1 + e^(-z))        z = -2 -> 0.12,   z = 0 -> 0.50,   z = 2 -> 0.88
```

So V_i(j) = sigmoid(w_i . x_j) is "the chance member i likes member j's looks."

### Starting point plus personal correction (paper 4)

```
w_i = w_0[g] + delta_i
```

- **w_0[g]** is what members of i's orientation group (for example, women looking at men)
  like on average, learned from everyone's swipes.
- **delta_i** is member i's personal correction, learned only from i's own swipes.

This is paper 4's idea. A general model is cheap to learn because there's lots of data. A
small personal correction needs only a few ratings, because it only has to describe how
someone differs from the average.

### How delta_i is learned

Choose delta_i to minimize

```
sum over i's swipes of  logloss(liked?, V_i(j))   +   (l2 / 2) * |delta_i|^2
```

- **Log loss** punishes confident wrong guesses: predicting 0.95 for someone she passed on
  costs much more than predicting 0.6.
- **The penalty** (l2 = 1.0) pulls delta_i toward zero. With 3 swipes the model stays near
  w_0; with 50 it can move a long way. A test checks both cases.

It's solved with **Newton's method**: repeatedly step to where a quadratic approximation of
the loss is lowest, using the gradient (slope) and the Hessian (curvature). For 65 numbers
and 50 swipes this takes a few milliseconds, so the model is refitted after every swipe.

### The Bayesian view, and uncertainty (paper 3)

The penalty is the same as saying "before seeing any swipes, I believe delta_i is near 0",
a Gaussian prior. Minimizing loss plus penalty then finds the most probable delta_i given
the swipes (the MAP estimate). This is Bayesian logistic regression, which is a Gaussian
process with a linear kernel: the simplest member of paper 3's family.

A **Laplace approximation** gives the uncertainty. The inverse of the Hessian at the optimum
is a covariance matrix, Sigma, saying how unsure each weight is. For a candidate:

```
s_j^2 = x_j' Sigma x_j        how unsure the model is about j's logit
```

Every swipe shrinks Sigma, and a test checks that 30 swipes lower the uncertainty about an
unseen profile.

## 5. The starting taste w_0 (FR-LM-14)

It's fitted on **19,468 swipes from 390 members** (half the city; the other half is used to
evaluate). First one pooled fit over everyone, then one per group (woman to man, man to
woman, man to man, woman to woman), each pulled toward the pooled fit with penalty 10.
Small groups, such as women looking at women, borrow strength from everyone instead of
overfitting their few swipes.

**Why w_0 is dangerous (rule R2).** w_0 is roughly "what most people in this group find
attractive". Ranked on its own, it would be the global attractiveness score your homepage
promises not to have. So:

- it's only ever used *under* a member's own swipes;
- `PersonalTaste.list_scores` refuses to score a daily list until the member has done all 50
  calibration swipes, and a test checks this;
- it's never stored per member.

**Why it matters in practice.** Member 301, a 27-year-old woman, has a taste that runs
against the shared taste. On her held-out profiles, w_0 alone scores an **AUC of 0.389**,
worse than a coin flip. After her own 50 swipes, her model scores **0.944**. Across members,
the further someone's true taste is from the shared taste, the more the personal model gains
(correlation -0.39).

## 6. Choosing the 50 profiles (FR-LM-16)

### Profiles 1 to 20: variety (paper 2)

A determinantal point process (DPP) is a way of picking a subset that favors items that are
different from each other. With photo directions as items, a set's "score" is the volume
its vectors span: similar photos span almost no volume, while different photos span a lot.
Greedily, each next pick is the profile whose photo vector has the **largest part not
already explained** by those chosen. That's the same as one step of Gram-Schmidt
orthogonalization:

```
residual_j = v_j minus its projection onto the span of the chosen profiles
next pick  = the j with the largest |residual_j|
```

The first pick is the profile closest to the average look. Result: the first 20 profiles
covered **all 8 look types for every member**, versus 7.33 on average for 20 random
profiles.

### Profiles 21 to 50: the most informative (papers 1 and 3)

For each remaining candidate:

```
information_j = log(1 + p_j (1 - p_j) s_j^2)
```

- **p(1 - p)** is largest when the predicted like chance is near 50%: a swipe whose answer
  you can already guess teaches little.
- **s^2** is largest for kinds of profiles the model hasn't learned about yet.

This is the expected information gain from one more swipe in Bayesian logistic regression.
I tested the alternatives on the tuning city:

| Method for profiles 21 to 50 | Gain over random (AUC) |
|---|---|
| Expected information (chosen) | +0.001 to +0.013 depending on the penalty |
| Closest to a 50% like chance (the research doc's rule) | +0.003 to +0.012 |
| Variance reduction over the pool | +0.003 to +0.012 |
| Uncertainty s^2 alone | -0.010 to -0.015, worse than random |
| All 50 for variety | -0.005 to -0.006, worse than random |

Uncertainty alone chases unusual-looking profiles that tell you little about the people
you'll actually rank, and so does pure variety. The chosen method is no worse than the best
alternatives, and simple.

## 7. How it was measured

- **Different people for training and testing.** Matchable members are split in half at
  random: one half fits w_0, the other half is evaluated.
- **Same answers for both methods.** Each evaluated member calibrates twice, once with our
  method and once with 50 random profiles. Every swipe answer is fixed in advance, so if
  both methods show the same profile, the member answers the same way. Any difference comes
  from which profiles each method chose, not from luck. This is called common random
  numbers.
- **Held-out profiles.** Both models then score profiles neither method showed.
- **AUC.** Pick a random profile she liked and a random one she passed on: AUC is the chance
  the model scores the liked one higher. 0.5 is a coin flip, 1.0 is perfect.
- **The ceiling.** The swipes are noisy (the logistic choice from Sprint 1), so even ranking
  by her true taste can't reach 1.0. Ranking by true taste gives the ceiling: the best any
  model could do.
- **Rank correlation with true taste.** This is only possible in test runs: how closely the
  model's order matches her hidden true order (1 = identical).

## 8. Results

On the main city (seed 7), across 377 members:

| Model | AUC | Correlation with true taste |
|---|---|---|
| w_0 alone (no personal swipes) | 0.750 | 0.738 |
| Personal, 50 random profiles | 0.835 | 0.879 |
| Personal, 50 calibration profiles | **0.844** | **0.889** |
| Ceiling: knows true taste | 0.875 | 1.000 |

Calibration minus random: **+0.009 AUC** (standard error 0.002). Calibration minus w_0:
**+0.094** (standard error 0.004).

**Learning curve (AUC by number of swipes):**

| Swipes | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|
| Random | 0.784 | 0.807 | 0.822 | 0.829 | 0.835 |
| Calibration | 0.788 | 0.811 | 0.828 | 0.838 | 0.844 |

Calibration reaches random's 50-swipe accuracy after about 40 swipes.

**Four different fake cities** (200 members each, except the main city):

| City | Calibration minus random | Personal minus w_0 | Look types in first 20 (calibration vs random) |
|---|---|---|---|
| seed 7 (main) | +0.009 (SE 0.002) | +0.094 | 8.00 vs 7.33 |
| seed 101 (tuning) | +0.009 (SE 0.003) | +0.095 | 8.00 vs 7.43 |
| seed 8 | +0.004 (SE 0.002) | +0.086 | 8.00 vs 7.24 |
| seed 9 | +0.009 (SE 0.002) | +0.094 | 8.00 vs 7.43 |

**Three individual members:**

| Member | w_0 alone | 50 random | 50 calibration | Ceiling |
|---|---|---|---|---|
| 981, typical | 0.754 | 0.827 | 0.821 | 0.846 |
| 301, biggest gain | 0.389 | 0.947 | 0.944 | 0.969 |
| 439, smallest gain | 0.735 | 0.709 | 0.681 | 0.760 |

For member 439 the personal model did *worse* than w_0. His swipes are noisy (his ceiling is
only 0.760), and 50 noisy swipes can mislead. Averages hide cases like this.

## 9. The fairness audit (FR-LM-17)

For each member, take the top 30 profiles by (a) the model, (b) w_0 alone and (c) their true
taste, and count which fake group each profile belongs to:

| Group | Share of pool | True taste | Model | w_0 alone |
|---|---|---|---|---|
| G1 | 23.0% | 22.5% | 22.7% | 23.0% |
| G2 | 20.4% | 19.1% | 18.4% | 16.6% |
| G3 | 20.4% | 22.1% | 22.3% | 24.0% |
| G4 | 18.7% | 19.6% | 20.8% | 22.7% |
| G5 | 17.4% | 16.7% | 15.8% | 13.7% |

How to read it:

- **True taste already differs from the pool.** In the fake data, groups lean toward certain
  looks (Sprint 0 built this in on purpose), so members' own tastes favor some groups
  slightly.
- **The personal model tracks true taste closely.** Every group is within 1.2 points, so
  nothing is flagged; the tolerance is 5 points.
- **w_0 alone exaggerates the skew.** G4 rises to 22.7% (true taste: 19.6%), and G5 falls to
  13.7% (true taste: 16.7%). An average taste amplifies the majority's lean. That's one more
  reason w_0 never ranks lists alone.

**What this audit does not decide.** It checks that the model doesn't exaggerate members'
own tastes. It doesn't decide whether matching on looks that correlate with ethnicity or age
is acceptable at all. That's a legal and ethical question, and the research doc recommends a
New York lawyer review it before launch.

## 10. How the settings were chosen

The two penalties were tuned on a **different fake city (seed 101)**, then evaluated on the
main city. Tuning and testing on the same data would flatter the result. Part of the grid,
showing calibration AUC on the tuning city:

| Personal penalty | w_0 penalty 0.1 | 1.0 | 10.0 |
|---|---|---|---|
| 0.3 | 0.838 | 0.837 | 0.837 |
| 1.0 | 0.836 | 0.838 | **0.840** |
| 3.0 | 0.814 | 0.816 | 0.818 |
| 10.0 | 0.759 | 0.764 | 0.765 |

Too much penalty (10) and the personal model can barely move away from w_0, so accuracy
drops. The chosen values are 1.0 and 10.0.

## 11. Honest limits

- **The fake world suits this model.** In the test harness, true taste is a straight-line
  function of the photo embedding, and the model is also a straight-line function. The model
  matches the truth's form exactly. Real taste in CLIP space is messier, so real accuracy
  will be lower. Papers 5 and 7 report that photos carry strong signal in real data (paper 7:
  F1 of 0.87 on mutual matches), but they use different measures, so the numbers aren't
  comparable.
- **Calibration's edge is small** in test runs (+0.004 to +0.009 AUC). It might be bigger or
  smaller with real people. A launch A/B test (half random, half calibrated) would settle
  it.
- **13 members were skipped** because their calibration pool had fewer than 80 profiles (50
  to show plus 30 to test).
- **Looks are not in the matcher yet.** Sprint 3 mixes V with P.

## 12. What changed in the plan

- **Profiles 1 to 20:** greedy DPP selection itself, rather than the simpler farthest-point
  stand-in.
- **Profiles 21 to 50:** expected information, rather than "closest to 50%". It accounts for
  the model's uncertainty, and tested no worse.
- **New decision D12:** calibration profiles come from everyone of a sought gender inside the
  member's age range, ignoring distance, because many members' match pools are under 50.
- **D11 extended:** settings are tuned on a separate fake city.

All written up in `docs/SPEC.md`.

## 13. Math to study

| Topic | Where it shows up | Where to learn it |
|---|---|---|
| Logistic regression, sigmoid, log loss | The taste model | Jurafsky and Martin, *Speech and Language Processing*, chapter 5 (free online) |
| L2 regularization | The penalty on delta_i | Same chapter, section on regularization |
| PCA, eigenvectors, SVD | Photo features | *Mathematics for Machine Learning*, chapter 10 (and chapter 4 for SVD) |
| Newton's method, gradient, Hessian | Fitting the model | *Mathematics for Machine Learning*, chapters 5 and 7 |
| Priors, MAP, Laplace approximation, Bayesian logistic regression | Uncertainty s^2 | Bishop, *Pattern Recognition and Machine Learning*, sections 4.4 and 4.5; Rasmussen and Williams, *Gaussian Processes for Machine Learning*, chapter 3 (free online) |
| Determinantal point processes | Profiles 1 to 20 | Kulesza and Taskar (paper 2), sections 1 and 2 |
| Active learning | Profiles 21 to 50 | Settles, *Active Learning Literature Survey* (free tech report), then paper 1 |
| AUC (Mann-Whitney) | Measuring | Any statistics text on ROC curves |

## 14. Try it yourself

```bash
.venv/bin/python -m lastfirstdate.looks_report --viewers 100          # about 15 seconds
.venv/bin/python -m lastfirstdate.looks_report --seed 9 --viewers 100 # another city
```

Then change `config/lastfirstdate.toml`, rerun, and change it back:

- `n_variety = 0`: no variety profiles, all 50 by information. What happens to the look-type
  coverage?
- `n_profiles = 20`: a shorter onboarding. How much accuracy do you lose? (The learning curve
  above suggests about 0.03 AUC.)
- `l2_penalty = 10.0`: the personal model can barely move, and AUC drops toward w_0's.
- `pca_dim = 8`: fewer photo numbers. Does the model get better or worse?

## 15. Read before Sprint 3 (the 50/50 mutual score)

| Paper | What to look for |
|---|---|
| 10. Pizzato et al., *Reciprocal Recommender System for Online Dating* | Why a dating match must be scored in both directions, and how they combine the two |
| 6. Neve and McConville, *ImRec* | Merging two people's photo preferences into one mutual score |
| 7. Neve and McConville, *Photos Are All You Need* | How far photos alone get you on mutual matches |

Math: percentiles (ranking within a group), and the harmonic mean versus the ordinary
average.

## 16. Questions for you

1. **Is 50 swipes acceptable for onboarding?** With 30 swipes the model scores 0.828 instead
   of 0.844. A shorter onboarding may lose fewer sign-ups than it costs in accuracy. It's a
   product call.
2. **Who should own the audit tolerance** (5 points today), and the wider question of whether
   looks-based matching is acceptable when looks correlate with ethnicity and age? I suggest
   the lawyer review from the research doc.
3. Still open: capping dates for members who don't pay, offering the guarantee only to members
   with 10 or more possible matches, expected NYC member numbers, and the original pre-filter
   SQL query.

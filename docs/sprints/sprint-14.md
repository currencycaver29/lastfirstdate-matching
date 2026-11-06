# Sprint 14: abuse testing

**In one paragraph.** You chose option (e). Synthetic attackers joined the test runs, each
trying to game the engine, and the sprint measured what each one gained. Four attacks gained
something and now have a defence and a test:
- **Stuffed prompts:** a prompt stuffed with the words everyone writes climbed from the middle
  to the 67th percentile of women's personality rankings. Prompts are now compared after taking
  out the zone's average text, and the gain is gone.
- **Flooded photo answers:** one account writing twenty exercises' worth of answers could
  steer everyone's starting taste. One account now counts as one exercise.
- **Fake women on the gate:** 50 fake women in a 300-woman zone made the gate sell 6.7 places
  instead of 3, and the 3-date share of the men who bought them fell from 94% to 83%. The gate
  now counts only verified women who aren't silent.
- **Bad rows:** forged ratings and malformed booking rows counted, or crashed the daily lists.
  Now they're ignored or refused by name.

Two attacks were already blocked and now have tests that keep them so:
- **Reading a ranking from lists:** spies' lists correlate with consensus looks at only 0.04,
  thanks to the exposure cap.
- **Buying rank with ratings:** ratings can't raise anyone above neutral.

One attack is your call: a ring of 3 colluding women got a man paused in 5 of 24 tries. This is
not a pentest. The app's logins, API and servers still need the outside pentest before the
beta.

**Papers used** (all in `docs/research.md`, "Methods added since"):
- **Lam and Riedl (2004), Mobasher et al. (2007):** the "average attack" on recommenders, a
  fake profile made to look like everyone. A stuffed prompt is the text version.
- **Mu, Bhat and Viswanath (2018), Ethayarajh (2019):** take the average vector out before
  comparing texts, so what everyone shares counts for nothing. This is the fix for the average
  attack.
- **Douceur (2002), the Sybil attack:** without an identity check, one attacker can be any
  number of accounts. That is why only verified accounts teach the starting taste or count for
  the gate.
- **Steinhardt, Koh and Liang (2017); Li et al. (2016):** how far coordinated fake data moves a
  fitted model, and what statistical defences cost.
- **Dellarocas (2000):** unfair ratings and how to bound any rater's influence.
- **Calandrino et al. (2011):** what a recommender shows can leak what it learned. This is the
  spies test.

---

## 1. What got built

| Change | Where |
|---|---|
| The attackers and the report: `python -m lastfirstdate.attacks` (FR-LM-63) | `attacks.py` (test runs only); three hooks in `cycle_runs.py` |
| Centred prompt similarity, `personality.center_text` (FR-LM-64, D21) | `personality.py`, both TOMLs (on in Phase 1) |
| One account, one exercise; verified-only starting taste (FR-LM-65, D23) | `app.calibration_answers`, `app.starting_taste` |
| The gate counts verified, non-silent women: `app.counted_women`, `gate.silent_after_days` (FR-LM-66, D22) | `app.py`, both TOMLs |
| Ratings count only for completed dates between the two (FR-LM-67) | `app.trust_from_db` |
| Booking rows must carry their facts; `app.check_rows` (FR-LM-68) | `db/schema.sql`, `app.load_bookings` |
| Tests that pin the trust score's bounds and the lists' privacy (FR-LM-69, FR-LM-70) | `tests/test_sprint14.py` |

26 new tests. The research profile (`config/lastfirstdate.toml`) keeps centring off, so every
earlier result there is unchanged. Version 0.14.0. The only change to how the engine is called:
`app.calibration_answers` now takes the settings, and the app calls it only through
`MarketCache.refresh` and `resume_calibration`.

## 2. Stuffed prompts (FR-LM-64)

**The attack.** The personality score compares two prompts by the angle between their vectors
(cosine similarity). Every prompt shares some words with every other one: in the test runs the
prompt sentences themselves ("A perfect Sunday looks like…"); in real life words like "love",
"travel", "coffee", "fun". A man who stuffs his prompt with those words points his vector at
the middle of everyone's. He then sits close to every woman, a little above the typical man
with each of them, and so higher in nearly every woman's ranking.

**The maths of the fix.** Let x_i be member i's text vector (length 1) and m the average over
everyone in the zone who wrote something. Compare

    t_ij = cos(x_i - m, x_j - m)

instead of cos(x_i, x_j), then clip below at 0 as before. Subtracting m removes exactly the
part every prompt shares. What's left is what makes a prompt different from the zone's
average, and two people score high only if they differ from the average in the same direction
(both hike, both love opera). A stuffed prompt is close to m, so x - m is small and points
nowhere in particular. Its similarity with everyone drops to around the zone's typical value,
or lower. An empty prompt stays a zero vector and scores 0.

This is the same move the photo side already makes: the photo compression (PCA) centres the
photo vectors first, because SigLIP's vectors are all fairly alike (Sprint 13). Sentence models
like MiniLM put one shared direction in every vector (Ethayarajh), and Mu et al. showed that
taking out the mean improves similarity.

**Results** (ten attackers in each of three zones; mean percentile in women's personality
rankings, 0.5 is the middle):

| Prompt | Word counts, off | Word counts, on | MiniLM, off | MiniLM, on |
|---|---:|---:|---:|---:|
| honest | 0.49 | 0.48 | 0.49 | 0.48 |
| a list of popular interests | 0.59 | 0.46 | 0.23 | 0.43 |
| popular interests written as a prompt | 0.52 | 0.48 | **0.59** | 0.48 |
| the words everyone writes, repeated | **0.67** | 0.45 | 0.18 | 0.42 |

The two encoders reward different stuffing. Word counts reward repeated shared words. MiniLM
ignores word counts and judges meaning, so a bare keyword list looks strange to it and scores
low, but popular interests written as a normal prompt score high. With centring, no version
gains with either encoder.

**What it buys.** Stuffed men appeared in 12% of eligible women's top 10s instead of 10%, but
they weren't shown or picked more often: the exposure cap limits how often anyone is shown, and
in the test runs women pick on true appeal. Real people pick from the top of a list more
often, which the test runs don't model, so the real harm of a distorted ranking is larger.

**Honest matching doesn't suffer.** On the tuning seeds, chance both say yes in the top 10 and
the 3-date share moved within noise. In the NYC demo (held-out seeds) men with 3 dates went
75.5% → 77.2% (300 women) and 90.2% → 90.6% (500 women), and credits fell from 16.1% to 14.8%
and from 6.6% to 6.0%. The top-10 quality is unchanged. The README charts now show these
numbers.

## 3. One account, one exercise (FR-LM-65)

**The attack.** The starting taste w_0 is one logistic fit on everyone's single-photo answers
(Sprint 2). The database accepted any number of answer rows per member, about any photo, even
the member's own. A client that writes 600 answers, liking photos that look like a target man,
pulls w_0 toward his look. Every woman's taste starts from w_0, so he climbs everyone's looks
ranking.

**The fix.** For each member the engine reads answers in position order and keeps them only
while three things hold: no more than `calibration.n_profiles` (30) photos in total, each photo
once, and never their own photo. Everything else is ignored and listed by `app.check_rows`.
Only verified members' answers fit w_0. A normal exercise reads exactly as before.

**Coordinated fakes.** The flood was cheap (one account). Many accounts each answering one
honest-looking exercise the same way are a harder problem:

| Attack | Target's looks percentile | Honest women's looks AUC |
|---|---:|---:|
| no attack | 0.473 | 0.936 |
| one account, 20 exercises, no defence | 0.517 | 0.934 |
| one account, 20 exercises, one-exercise rule | 0.473 | 0.936 |
| 15 verified fakes (5% of women) | 0.510 | 0.934 |
| 30 verified fakes (10%) | 0.537 | 0.932 |
| 30 fakes without identity checks, verified-only rule | 0.471 | 0.936 |
| 30 verified fakes, robust fit | 0.488 | 0.917 |

The "robust fit" fits each member alone and averages their weights after cutting the top and
bottom quarter of each weight (a trimmed mean). Fakes then count as members, not as answers, and
can only shift the middle of the distribution. It halves their gain, but every honest member
loses 0.016 AUC, because one member's 30 answers make a noisy fit. That cost falls on everyone,
every day, against an attack that needs thirty real, verified identities. So the engine doesn't
use it. Identity checks are the defence (Douceur), and the numbers above show what is left if
someone gets past them.

## 4. The gate counts women who can really date (FR-LM-66)

**The attack.** The gate sells men's places while the forecast says at least 90% will get
their 3 dates. The forecast's main input is how many women there are. Women join free, so fake
women make the zone look bigger than it is.

| 300 women, 450 men waiting | Women counted | Places sold | Forecast | Men with 3 dates |
|---|---:|---:|---:|---:|
| no fakes | 284 | 3.0 | 94.1% | 94.4% |
| 50 fakes | 331 | 6.7 | 95.1% | **82.6%** |

The gate at 300 women is tight (it sells only a handful of places), so 50 fakes more than
double what it sells and break the promise for those men. At 500 women the zone can carry far
more men than the gate sells (Sprint 8), so fakes there move the stopping point without
breaking the promise.

**The fix** (D22). A woman counts only if she is verified (whatever `filter.require_verified`
says), active, not paused, and not *silent*: lists on at least `gate.silent_after_days` (14)
days with no pick, ever. She counts again as soon as she picks. After four weeks of lists the
rule stopped counting 83% of the fakes and 0.6% of honest women. Those few honest women never
picked anyone either, so they weren't producing dates. A fake that stops opening the app goes
inactive after 14 days and drops out anyway.

Why "never picked" rather than "no pick lately": in the test runs, many honest women stop
picking in the later cycles, because they've seen and passed on their whole pool (Sprint 7 saw
the same). The forecast was fitted with them counted, so dropping them would make the gate more
cautious than its calibration. "Never picked" catches fakes and almost nobody else.

**At launch it can't help:** before the first lists, nobody has had a chance to pick.
Identity checks are the only defence then.

## 5. Only real dates count; bad rows are refused (FR-LM-67, FR-LM-68)

**Ratings.** The ratings table can't check, by itself, that a rating belongs to a date between
the two people: SQL checks can't look at another table. So the engine checks it. A rating counts
only if its booking is completed and the rater and the person rated are that booking's two
members. A forged rating changes nothing, and `check_rows` lists it.

**Bookings** (code review finding 2). The schema now refuses:
- a cancelled booking without who cancelled and when;
- a no-show without who came, or with both coming;
- a completed date with someone marked absent;
- "who cancelled" on any other state.

Before, one such row made the daily lists crash halfway with a confusing error (an empty time,
a missing person). On a database without these checks, `load_bookings` now stops with
"booking 17 is cancelled without who cancelled and when". It also ignores a payment from
anyone who isn't on the booking.

## 6. Rating attacks stay bounded (FR-LM-69)

The trust score was already built to resist unfair ratings (Sprint 7):
- only respect and accuracy count;
- a prior worth 5 ratings means nobody's score swings on one review;
- nobody can be lifted above neutral, so a ring can't buy rank.

The test runs checked two kinds of harm:
- **Harsh raters** (15 women who rate every man 1) wrongly flagged 1 honest man in three zones
  over three cycles. They can't flag anyone alone: a flag needs 3 low ratings, and each
  woman dates a man only once.
- **Rings** (women eligible for one man who pick him whenever they can, and he always
  accepts, an upper bound) got him paused in 5 of 24 tries with 3 women and 8 of 24 with 6. A
  ring needs 3 real, paid dates with him before his honest ratings build up, and he has at most
  3 dates a cycle.

The engine can't tell 3 colluding raters from 3 honest bad experiences: ratings are private
and personal by design. Two choices bound the harm, and both are yours (questions below).

## 7. Lists don't reveal a ranking (FR-LM-70)

**The attack** (Calandrino et al.). Lists are ordered partly by looks, and the starting taste
is everyone's average taste. So could someone read lists to rebuild a "hotness" ranking, which
rule R2 forbids? 20 accounts per zone that never pick read their lists daily for two weeks.
They ranked men by the day, then the position, each first appeared, pooled over all the
accounts.

**Result:** that ranking correlates with how much the zone's women like each man's looks at
Spearman 0.04 (0.14 with the exposure cap turned off). The cap is what protects: popular men
reach their fair share of lists early in the round-robin, so they show up on few lists, in no
order an observer can use. A test keeps the correlation below 0.2.

## 8. What changed in the plan

- Three new defaults, D21 to D23.
- The Phase 1 NYC numbers moved slightly: 77% and 91% of men with 3 dates.
- Code review finding 2 is fixed. The other eight review findings are still open.

## 9. Honest limits

- **The attackers are only as smart as we made them.** Each attack is one strategy. A real
  attacker adapts: for example, they stuff a prompt with words that are popular among the women
  he wants but rare overall. Centring blunts that too, since it removes only the zone's average,
  but a targeted prompt is closer to honestly describing yourself.
- **Women pick on true appeal in the test runs,** with no position bias, so the test runs
  understate what a higher ranking is worth.
- **Fakes at launch** get past everything except identity checks. The silent rule needs about
  two weeks of lists.
- **Verified fakes** shift the starting taste in proportion to their share. 10% of women moved a
  target man up 6 percentile points.
- **Rings** are bounded, not stopped.
- **Out of scope:** the app's logins, API, rate limits and storage (the pentest), and attacks
  on the photo and text models themselves (doctored photos, model inversion), which are still
  on the reading list.

## 10. What to study

1. **Cosine similarity after centring.** Why subtracting the mean removes what all vectors
   share; why the result can go negative (and so we clip at 0).
2. **The average attack.** Why the point that maximises the average cosine with a set of unit
   vectors is the direction of their mean (sum the cosines, and you get one dot product with
   the sum).
3. **Robust statistics.** The median and the trimmed mean versus the plain mean. The breakdown
   point: what share of bad data an estimator can take before it moves without limit. Why
   robustness costs efficiency on clean data.
4. **The Sybil attack.** Douceur's argument for why only an identity authority, not any
   statistic, stops one person posing as many.
5. **Shrinkage in the trust score.** Why a prior worth k ratings bounds what any n raters can
   do: conduct = (k * prior + sum) / (k + n).

## Try it yourself

```bash
python -m pytest tests/test_sprint14.py -q
```

```bash
python -m lastfirstdate.attacks
```

```bash
python -m lastfirstdate.attacks --minilm ~/.cache/lastfirstdate/models
```

## Questions for you

1. **Should a trust flag pause a man at once, or wait for a person to review?** Pausing protects
   women from a real bad actor right away; waiting protects men from a ring. A middle way: pause
   at once only when respect is rated low (a safety issue), wait when only accuracy is.
2. **How many low ratings before a flag?** Phase 1 settings say 3 (`trust.flag_min_ratings`);
   the earlier question (number 3) suggested about 6. With 6, a ring needs 6 real dates with the
   same man, two full cycles.
3. **Identity checks for women before launch:** the gate trusts the women's count, and only
   ID checks stop fakes before the first lists go out. Confirm the app verifies every woman before
   she counts (Phase 1 settings require it).
4. **Turn on centring for MiniLM in production?** It's on in the Phase 1 settings (D21), with
   either text encoder. If you'd rather keep the old behaviour, it's one setting.

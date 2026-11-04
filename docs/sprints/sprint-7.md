# Sprint 7: booking, billing, trust and geography

> **Correction (Sprint 8).** The simulation in this sprint never gave a man his slot back after a
> woman didn't show up, so a no-show could never be refilled, though the site says it is. Fixed
> in Sprint 8 (`CycleSimulation.release_slot`). Corrected results for the same three markets:
>
> | Version | Reported here | Corrected |
> |---|---:|---:|
> | Real flow, 15% new women per cycle | 67.7% | **81.9% ± 2.8** |
> | Real flow, no new women | 58.2% | 69.2% ± 3.2 |
> | Every cycle starts on launch day | 70.4% | 86.6% ± 1.6 |
> | Show-up 97% instead of 92% | 75.5% | 81.7% ± 2.3 |
> | Everyone pays within the 15 minutes | 68.9% | 84.7% ± 1.8 |
> | No friction | 89.5% | 89.5% ± 3.1 |
> | Credits, real flow | 17.2% | 12.2% |
>
> What changes in the conclusions below:
> - friction costs about 8 points, not 22;
> - show-ups are no longer the biggest lever (refills absorb most no-shows);
> - new women joining every cycle are (+13 points);
> - the gap to the 90% north star is about the ratio: it takes about 4.5 women per man
>   (`docs/sprints/sprint-8.md`).

**In one paragraph.** Your website describes how a date actually happens: his profile appears
on her list *with a proposed date window*, she chooses, both $10 payments must arrive within
15 minutes, the venue is revealed, and the date counts toward his 3 only if it really happened
before *his* billing cycle ends. Sprints 1 to 6 skipped all of that: they booked dates in a
weekly batch and counted dates booked per calendar month. Sprint 7 builds your real rules.
Since no app exists yet, it also designs the **database the app should keep**. Then it
simulates the real flow **day by day over three billing cycles**. The result changes the
business picture: in three simulated Phase 1 markets, about **68% of men get 3 delivered dates
a cycle**, not the 87% to 96% the batch simulations suggested, and credits cost about **17% of
subscription revenue**. The causes are clear, and so are the levers (section 8).

**Papers used.** Sprint 7 is mostly your business rules turned into code, so it uses fewer
papers than earlier sprints.

| Paper | What it says | Where it shows up |
|---|---|---|
| 16. Arnosti, Johari, Kanoria, *Managing Congestion in Matching Markets* | In matching markets, people seeing and passing on the same options again and again waste everyone's effort | Why the market wears out without new women (section 7) |
| 12. Singh, Joachims, *Fairness of Exposure in Rankings* | Give each person a fair share of being seen, and boost those the system owes something | The refill bonus for men behind on their dates |

---

## 1. What got built

| File | What it does |
|---|---|
| `db/schema.sql` | The app's database: members, free slots, intent answers, prompts, photos and (separately) embeddings, the photo exercise, waitlist and gate decisions, subscriptions and cycles, events, bookings, payments, ratings, trust reviews, neighbourhoods and venues. Runs on PostgreSQL and SQLite |
| `booking.py` | Engine: proposed date windows and the life of a booking (proposed, held, confirmed, then completed, cancelled, no-show or expired) |
| `billing.py` | Engine: billing cycles after the free trial, what counts as delivered, refill bonus, pro-rated credits, $10 fee refunds |
| `trust.py` | Engine: private ratings and the behaviour-only trust score: lowers ranking, flags for review |
| `geo.py` | Engine: neighbourhoods and the fair venue choice |
| `cycle_sim.py` | Simulation: the real flow, day by day, over billing cycles, with women joining over time |
| `config/*.toml` | New sections `[booking]`, `[billing]`, `[trust]`, `[geo]`, `[cycle_simulation]` (rule R5: every number is a setting) |

The default fake population and every earlier result are unchanged (checked byte for byte).

---

## 2. The database (FR-LM-36)

One table per thing the app must remember. The design choices that matter:

- **No looks column anywhere** (rule R2). A test scans every column name.
- **Photo embeddings in their own table**, encrypted, deleted with the member (R3: they count
  as biometric data). Identity and income checks keep the *result* only.
- **Checks in the database itself**, so a bug in the app can't store nonsense. Ratings must be
  1 to 5, nobody can rate themselves, a confirmed date must have a venue, a refund can't exceed
  the fee, and a booking's state must be one of the seven the engine knows. A test checks that
  the engine and the database list the same states.
- Money in **cents** and times as ISO-8601 text, so it runs on both PostgreSQL (production)
  and SQLite (the tests).

---

## 3. Proposed date windows and the payment hold (FR-LM-37)

### Finding the window

Each member's weekly free slots (lunch, evening, late, for each of the 7 days) repeat every
week. Over the next 14 days that gives 14 × 3 = 42 possible windows, so a member's free windows
fit in one 64-bit number. Bit 3k + part is set when they are free on day today + k at that
part of the day.

Days are **blocked** when the member already has a date that day, their week is full (1 date
a week), or their month or cycle is full (3 dates). The proposed window for a pair is then just:

    common  = her_bits AND his_bits
    window  = the lowest bit set in common      (the earliest time both can make)

The lowest set bit has a classic trick: `common & (~common + 1)` keeps only that bit. Because
these are whole-number operations on arrays, the engine finds the window for every (woman, man)
pair at once. Men with no common window are not shown that day.

### The life of a booking

    proposed -> held -> confirmed -> completed
        |         |          |-----> cancelled   (by one of them, some hours ahead)
        |         |          '-----> no_show     (one or both didn't come)
        '---------'-----> expired               (the 15-minute hold ran out)

Any other move raises an error. A payment that arrives after 15 minutes expires the booking,
and the pair can find each other again later.

---

## 4. Billing cycles, delivered dates and credits (FR-LM-38)

- **Cycles.** A 30-day free trial, then 28-day paid cycles back to back, each man on his own
  clock.
- **Delivered:** both confirmed and paid, inside his cycle, and it happened.
- **Her cancellation or no-show:** doesn't count, and his slot is refilled.
- **His own late cancellation or no-show:** counts as delivered (decision **D17**). Otherwise a
  man could skip dates and still collect credits. The site doesn't say; it is now a question
  for the lawyer (risk plan, question 13).
- **Refill bonus:** `priority × missing dates / weeks left in his cycle`. It grows as his
  cycle runs out, so a man behind on his dates rises on women's lists.
- **Credit:** `$200 × missing / 3` at the end of each paid cycle (none during the trial).
- **$10 fee:** the person who cancels gets it all back 24+ hours ahead and half later; a
  no-show gets nothing; the person not at fault always gets it all back.

---

## 5. Ratings and the trust score (FR-LM-39)

After each date, both people privately rate respectfulness, profile accuracy and conversation
(1 to 5) and say whether they'd meet again.

**What trust may read: behaviour only.** Respectfulness, accuracy and reliability (showing up,
not cancelling late) count. Conversation and "would meet again" are chemistry, which is
personal, so they feed learning, never trust. Otherwise trust becomes a popularity score, which
rule R2 forbids. A test checks that changing them doesn't move trust at all.

**The maths:**

    conduct     = (k × prior + sum of scaled ratings) / (k + number of ratings)
    reliability = (k × prior + dates kept) / (k + dates scheduled)
    trust       = prior + (1 − w)(conduct − prior) + w(reliability − prior)

A rating r is scaled to (r − 1)/4, so 1 to 5 becomes 0 to 1. The prior counts as k = 5
ratings, so one bad date can't sink anyone, and a newcomer starts exactly at the prior.

- **Ranking:** `factor = 1 − 0.5 × max(0, prior − trust) / prior`. It is 1 for anyone at or
  above the prior: low trust pushes a member down, but high trust gives no boost, so it can't
  become something to compete on.
- **Safety:** conduct below 0.55 with at least 3 ratings flags the member for review and pauses
  their profile.

**How fast is a bad actor flagged?** Solving the conduct formula: a member whose ratings average
m (scaled) drops below the line after n > k(prior − 0.55)/(0.55 − m) ratings. For a typical
simulated bad actor (m ≈ 0.3) that is the **6th poor rating**. A man has about 3 dates a cycle,
so that's about two cycles. Flagging faster (a lower k, or a higher line) catches them sooner
but flags more ordinary members. That's your call (questions below).

---

## 6. Events and geography (FR-LM-40, FR-LM-41)

- **Event dates** are bookings with source "event". They follow the same pay-and-confirm rules,
  count toward his 3 and toward caps, and the pair is never shown to each other again.
- **Neighbourhoods, not addresses:** the app never holds anyone's exact location.
- **The venue:** the one that makes the *longer* of the two trips shortest, ties broken by the
  total trip. That's fairer than "closest to the midpoint" when venues are sparse. In the
  simulation the longer trip averaged 0.98 km, and the two trips differed by 0.30 km.

---

## 7. The real flow, simulated (FR-LM-42)

Every day of three billing cycles per man:

1. Women who open the app get up to 30 men, each with a window.
2. She picks, he accepts, and both pay within 15 minutes.
3. Before the date someone may cancel; on the day each shows up or not; afterwards both rate.
4. Events add a few dates.
5. New women join at 15% of the launch number per cycle.

Hidden truths decide how respectful each fake member is; 5% are bad actors. The engine only
sees their ratings.

### Results: three simulated markets, averaged

| Version | Men with 3 delivered dates | By cycle | Credits |
|---|---:|---|---:|
| **Real flow, 15% new women per cycle** | **67.7% ± 0.9** | 74%, 69%, 61% | 17.2% |
| Real flow, no new women | 58.2% ± 2.4 | 66%, 60%, 49% | 25.8% |
| Real flow, every cycle starts on launch day | 70.4% ± 0.5 | 73%, 72%, 67% | 14.6% |
| Show-up 97% instead of 92% | 75.5% ± 2.0 | 86%, 74%, 67% | 14.1% |
| Everyone pays within the 15 minutes | 68.9% ± 3.3 | 73%, 70%, 64% | 16.1% |
| No friction (all pay, nobody cancels, all show up) | 89.5% ± 3.1 | 97%, 89%, 82% | 7.4% |
| Sprint 6 weekly batch, booked dates, no new women | 93.0% ± 1.3 | | |

One market in detail (seed 7, the real flow):

| What happened | Count |
|---|---:|
| Picks | 3,581 |
| He declined | 985 (28%) |
| No window left that day (she had already booked someone that week; the pair stays open) | 1,278 |
| Payment hold expired | 156 |
| Bookings: completed / no-show / cancelled / expired | 894 / 151 / 99 / 149 |
| Event dates | 32 |
| Launch women's completed dates per 28 days | 0.66 on average; 55% had none |
| Trust flags | 10, all simulated bad actors; all 10 bad actors with 6+ ratings caught; none of the others |
| Trust against true looks appeal | correlation +0.15 |

### Why 68% and not 90%

1. **About one booked date in three falls through:** 31% of bookings were cancelled, no-shows
   or expired holds. Each one needs a refill, and late in a man's cycle there often isn't time.
   With no friction at all it's 90%, about the same as the batch simulation.
2. **The batch simulation was optimistic.** It counted *booked* dates, had no payment holds or
   cancellations, and forgot every pass at the start of each month.
3. **A closed market wears out.** Each woman's pool holds about 46 men. With up to 30 profiles
   a day and a 7-day rest after each showing, she sees every man in it about once a week. Each
   pass makes her less likely to pick him later (the simulator's model of people). By the third cycle the median man has been
   passed over about 300 times. Without new women, fulfilment falls from 66% to 49%. Paper 16
   describes exactly this kind of wasted repetition.

### What helped, and what didn't

| Lever | Effect |
|---|---|
| New women every cycle (15% of launch numbers) | **+10 points**, and no more decline from cycle to cycle |
| Show-ups 97% instead of 92% | **+8 points**, the biggest single lever |
| Everyone paying within the 15 minutes | +1 point (within noise) |
| All cycles starting on launch day | +3 points (within noise) |
| Showing the same man to the same woman less often (14 or 21 days) | no help (first test, one market) |
| Up to 2 dates a week instead of 1 | no help (67% against 68%, three markets) |
| A stronger refill bonus (×3) | no help (first test, one market) |

---

## 8. What this means for the business

- **Expect about 70%, not 90%, until show-ups improve.** Budget credits at about 17% of
  subscription revenue for the first cycles.
- **Show-ups are worth real effort:** reminders, the $10 fee, reliability inside the trust
  score, maybe a deposit for repeat no-shows. Every point of show-up rate is worth more than
  any ranking change tested so far.
- **Keep recruiting women after launch.** A beta that recruits 300 women once and stops will see
  fulfilment fall every cycle.
- **Women's experience:** at these ratios more than half of women have no completed date in a
  28-day period. "Up to three dates" is the honest wording, and events matter.
- **Two policy questions** for you and the lawyer (risk plan, questions 13 and 14): his own
  absences (D17), and whether a pick *he declines* (28% of picks in the simulation) should still
  leave you owing him a credit.

---

## 9. Bugs and corrections found on the way

- **His late cancellation counted twice.** Writing the "books balance" test caught it. Under
  D17, his own late cancellation counted as delivered, but the simulation also gave his slot
  back, so one man reached 5 "delivered" dates in a cycle. Now the cancelled date keeps using
  his cap and frees only hers.
- **Picks lost by accident.** A first version closed a pair forever when she picked a man but
  no window was left that day (her first booking had filled her week), or when the hold
  expired. In the app he'd just drop off her list, so the pair now stays open.
- **The first batch comparison was unfair.** It forgot who had already dated each month; now it
  remembers.
- **The original acceptance check was wrong.** It expected the real flow to be "near the batch
  simulation". It isn't, so the check is kept as a strict expected failure that will pass on
  its own if the real flow ever reaches 90%.

---

## 10. Honest limits

- **Everything is simulated.** Show-up, cancellation and payment rates are assumptions (all in
  `[cycle_simulation]`). Real Phase 1 data replaces them.
- **People never change:** the fake women's commitment doesn't make them pick more widely, and
  the $10 fee doesn't make anyone show up more often. Both would help in reality.
- **Ratings carry some halo.** The simulation assumes raters are slightly kinder to people they
  find attractive, and trust picked up a +0.15 correlation with looks from that alone. Audit
  real ratings by group, as Sprint 4 does for exposure.
- **No new men join,** and the gate isn't re-run each cycle in this simulation.
- **One zone,** with made-up venues.

---

## 11. Maths to study

1. **Bit masks:** a set of windows as one number; AND for "free for both"; `x & (~x + 1)` for
   the lowest set bit.
2. **State machines:** listing the legal moves makes illegal ones impossible to store.
3. **Bayesian shrinkage:** `(k × prior + sum) / (k + n)`, and solving it for "how many ratings
   until flagged".
4. **Minimax choice:** minimising the worse of two outcomes (the longer trip).
5. **Simulating over time:** state that carries over (passes, closed pairs, cycles) produces
   effects a single-month model can't show.
6. **Replicates and paired comparisons:** the same markets for every version, and a standard
   error before believing a difference.

## Try it yourself

```bash
.venv/bin/python -m lastfirstdate.cycle_sim                     # all versions, 3 markets (~2 min)
.venv/bin/python -m lastfirstdate.cycle_sim --seeds 7           # one market (~30 s)
.venv/bin/python -m pytest tests/test_sprint7.py -q             # Sprint 7's tests
```

Things to try: in `config/phase1.toml`, change `show_up_rate` under `[simulation]`,
`new_women_per_cycle` or `pay_in_time_chosen` under `[cycle_simulation]`, or `own_absence_counts`
under `[billing]`, then rerun.

---

## Questions for you

1. **D17:** should his own late cancellation or no-show count as one of his 3 dates (my
   default), or be refilled like hers?
2. **Declined picks:** if a woman chooses him and he says no, should that count toward his
   guarantee? It would cut credits a lot, but it needs clear terms.
3. **Trust speed:** flag a bad actor after about 6 poor ratings (now), or sooner, at the cost of
   reviewing more ordinary members?
4. **Recruiting:** can marketing keep bringing in about 15% new women per cycle after launch?
5. **Show-ups:** which tools are you willing to use (reminders, fees, deposits for repeat
   no-shows)?
6. **Sprint 8:** the original eight-stage plan is now built. Next could be (a) refitting the
   forecast and gate on the real cycle flow, so the gate uses delivered dates; (b) a Rust
   service the app can call; or (c) wiring to the real app once it has a repo.

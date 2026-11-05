# Sprint 11: ready for real data

**In one paragraph.** Option (b) was "wait for real data, then refit on real cycles". There is
no real data yet, and this repo must never hold any (rule R3). So this sprint builds and tests
what turns Phase 1's first real cycles into a better forecast: when they arrive, refitting is
one tested call. It rebuilds every completed billing cycle from the app's database: who was in
the market, whether he got his 3 dates, and his progress each week. It **recalibrates** the
forecast on those cycles with two numbers, a level and a steepness, and decides whether to
switch by testing on the latest cycles. In a simulated test where people showed up less often
than the forecast assumed, the forecast said 83% while 76% of men got their dates; the
recalibrated forecast said 76%, and the switch rule recommended it. On the way it found that
**fulfilment drifts as a zone ages**, so recalibration has to be repeated, about monthly.

**Papers used.** None from the original 17. The method is Platt scaling (Platt, 1999), the
standard way to correct a classifier's probabilities on new data, and the Brier score (Brier,
1950) judges the forecasts. Both are listed in `docs/research.md` under "Methods added since".

---

## 1. What got built

| File | What it does |
|---|---|
| `src/lastfirstdate/refit.py` | Engine side: `cycle_rows` (outcomes from records), `recalibrate` and `fold`, `brier`, `propose_refit` (the switch rule) |
| `src/lastfirstdate/app_sim.py` | `write_history`: writes a simulated run into the database as the app's records (subscriptions, join dates, bookings with timestamps, payments, refunds) |
| `src/lastfirstdate/cycle_sim.py` | Remembers when each booking was made and cancelled, so its history has real timestamps |
| `config/*.toml` | `[refit]`: `min_man_cycles` (150), `holdout_share` (0.3), `min_improvement` (0.002) |
| `docs/handoff/README.md` | A monthly "refit on real cycles" step for the app |

---

## 2. Outcomes from the app's records (FR-LM-51)

For every paid cycle that has ended, `cycle_rows` works out:

- **The market at the cycle's start:** the women who had joined by then and the men who had
  subscribed by then. The forecast's features come from this.
- **The outcome:** did it reach 3 delivered dates? This uses the same rules as billing (D17
  included).
- **Each week's progress:**
  - dates delivered so far;
  - **confirmed dates still to come**: confirmed before that day, inside the cycle on or after
    it, and not cancelled before it. Rebuilding this from timestamps is the subtle part.

**The proof:** a simulated history written into the database and rebuilt from the records
alone:

| Check | Result |
|---|---|
| Cycle outcomes (291 cycles the simulation reported) | all identical |
| Weekly progress (838 snapshots) | all identical |

The records also contain 6 extra fourth cycles the simulation's summary doesn't report, from
men whose cycles started on launch day. They are genuine completed cycles, so they count.

---

## 3. Recalibration (FR-LM-52)

### Why not refit from scratch?

Phase 1 is **one zone**. Every man in it sees the same women per man, so real Phase 1 data
can't teach the forecast how much the ratio matters. A refit from scratch would get that
coefficient wrong. What one zone *can* tell us reliably is whether the forecast is too
optimistic or too pessimistic, and whether it's too sure of itself.

### The method (Platt scaling)

    logit(p_real) = a + b · logit(p_model)

- a < 0: the model was too optimistic; a > 0, too pessimistic.
- b < 1: the model was too sure of itself; b > 1, too timid.
- a = 0, b = 1: no change. The fit is pulled toward that (the same small penalty as
  elsewhere), so a handful of cycles can't swing the forecast far.

**Folding.** The model is linear inside the sigmoid (logit = w · x), so

    a + b · (w · x) = (b·w) · x + a

The correction becomes new coefficients: b times every coefficient, plus a on the intercept.
The gate, its Rust twin and every alert keep working unchanged. The model file records a, b and
how many cycles were used.

### The test world

The Phase 1 market was simulated over three cycles with **80% show-ups instead of the 92% the
forecast assumes**:

| | Share of cycles with 3 delivered dates |
|---|---:|
| What happened | 76.1% |
| The current forecast | 83.3% (too optimistic) |
| Recalibrated (a = −0.48, b = 0.96) | **76.3%** |

---

## 4. When to switch (FR-LM-53)

`propose_refit`:

1. **Too few cycles?** With fewer than 150 completed paid cycles, the answer is no. Phase 1
   with 100 men reaches that in about two months.
2. **Otherwise** recalibrate on the earlier 70% of cycles and compare on the latest 30%,
   using the Brier score: the mean squared gap between forecast and outcome.
3. **Switch only if the recalibration is better by at least 0.002;** then recalibrate on
   every cycle.

In the test world: current 0.2179, recalibrated 0.2112, so the switch is recommended.

---

## 5. The finding: fulfilment drifts as a zone ages

My first test split the cycles at the median and expected the recalibration to do better on
the second half. **It did worse:**

| Later half of the cycles | Off by |
|---|---:|
| Current forecast | 15.0 points |
| Recalibrated on the first half | 16.1 points |

The first half is mostly first cycles, when about 88% of men get their dates. By the third
cycle it's about 71%, because the market wears out (Sprint 7: the same people keep passing on
each other). A recalibration learned on young-zone cycles can't foresee the decline.

What this means:

- **Recalibrate about monthly on recent cycles,** so the forecast follows the zone's age.
- **Trust the switch rule, not a one-off fit:** it tests on the latest cycles first, which is
  what catches a recalibration that doesn't transfer.
- **When several zones exist,** the zone's age is a candidate forecast feature.

The test now uses the switch rule's own split (70/30), where recalibration does win. Its
docstring says why the median split fails.

---

## 6. Honest limits

- **Everything is still simulated:** the "real" history is a simulation written as records. The
  machinery is proven; the numbers it will produce aren't known until Phase 1 runs.
- **Market features use today's profiles,** not profiles as they were then; old preferences
  and pauses aren't stored.
- **Only the guarantee forecast is refitted.** The personality weights and the looks starting
  taste also deserve refitting on real outcomes (paper 9); the looks starting taste already
  refits weekly in the cache. The personality weights need real "would meet again" answers,
  which only the real app can collect.

---

## 7. What to study

1. **Platt scaling:** why two numbers fix most miscalibration, and why folding works for any
   model that is linear inside a sigmoid.
2. **Time splits versus random splits:** testing on the *latest* data is the honest test when
   the world drifts.
3. **The Brier score:** a squared error for probabilities. It rewards being both well
   calibrated and sharp.
4. **Identifiability:** why one zone can't teach the ratio effect (no variation, no signal).

## Try it yourself

```bash
.venv/bin/python -m pytest tests/test_sprint11.py -q
```

---

## Questions for you

1. **Monthly refits:** who will run `propose_refit` each month once Phase 1 is live, the app
   developer or you? It's one call, and the handoff shows it.
2. **The threshold:** 150 completed cycles (about two months for 100 men) before any switch. Is
   that patience acceptable, or would you rather switch sooner on less evidence?
3. **Sprint 12:**
   - **(a)** re-run the gate every cycle with the zone's age and recruitment as inputs;
   - **(b)** the same readiness work for the personality weights (refit on real "would meet
     again" answers);
   - **(c)** stop here until the app and real data exist.

   I'd recommend (c). The engine is ready for Phase 1, and real data will say what to improve
   next.

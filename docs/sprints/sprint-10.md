# Sprint 10: step-by-step calibration and the nightly cache

**In one paragraph.** Sprint 9's handoff listed two things not covered yet, and this sprint
does both. First, the **photo exercise one question at a time**: the app asks the engine for
the next photo or pair, shows it, and sends back the member's tap. If the member leaves
halfway, the engine picks up exactly where they stopped. The new `CalibrationSession` is now
the only implementation of the exercise; the test runs run through it too, so the app and
the test runs can never drift apart. It asks exactly the questions the old code asked, and
the default city's scores came out byte-identical. Second, a **nightly cache** that keeps the
day's work and refits only the members whose answers changed. For Phase 1 a quiet night now
takes 0.07 seconds instead of 0.5.

**Papers used.** None new. The questions asked are still papers 1 and 2 (the most
informative photo or pair, and a varied set to start).

---

## 1. What got built

| File | What it does |
|---|---|
| `looks.py` | `CalibrationSession` (`next_question`, `answer`, `resume`, `done`) and `Question`. `calibrate` now just runs a session to the end |
| `app.py` | `start_calibration`, `resume_calibration`, `question_photo_ids`, `answer_row` (database ids); `starting_taste`; `MarketCache` |
| `docs/handoff/README.md` | The photo exercise and the nightly job rewritten for the new API, with measured times |

---

## 2. The photo exercise as a conversation

Before, `calibrate` ran the whole exercise in one go, calling back for each answer, which is
awkward for an app where each answer arrives with a tap, seconds or days apart. Now:

```
session = app.start_calibration(market, features, member_id, cache.start, cfg)
q = session.next_question()        # e.g. Question(a=17)        → "Interested?"
                                   #  or Question(a=17, b=42)   → "Who are you more drawn to?"
session.answer(True, question=q)   # her tap
...until session.done
```

What the session guarantees:

- **The same order as before.**
  1. Up to 12 varied single photos, chosen in advance; the model is refitted once after the
     last of them.
  2. Up to 9 pair questions.
  3. The most informative single photos, until 30 photos have been shown.
- **The same question until it's answered.** Asking twice returns the same question, so a
  page reload doesn't skip a photo.
- **No double answers.** `answer(..., question=q)` refuses an answer to a question already
  answered, for example after a double tap.
- **Resuming.** `resume(stored answers)` replays them. If the member's pool or the starting
  taste changed in the meantime, a stored answer the session wouldn't ask now is still learned
  from, and those photos aren't shown again.

### One implementation, not two

`looks.calibrate`, which every test run uses, is now four lines on top of the session. So
whatever the test runs measured is exactly what the app will do. The proof:

| Check | Result |
|---|---|
| Question by question against a frozen copy of the old algorithm (25 members, Phase 1 with pairs and the research setup without) | identical questions, identical taste models |
| The default 1,000-member city's 50/50 scores | byte-identical to before |
| A synthetic month on those scores | exactly the same dates |
| Interrupted after 3, 14, or all but one answer, then resumed | the same remaining questions |

---

## 3. The nightly cache

`MarketCache` lives in the app's server.
- **Every night:** `refresh()` reloads the members, refits the taste models of members whose
  answers changed, and recomputes the scores.
- **If nothing changed at all:** it returns the cached scores straight away.

**Why the compression and the starting taste are frozen.** Photo features come from a PCA
compression of everyone's photo embeddings, and every taste model is a small correction on
top of the shared starting taste. Refitting either one shifts every member's features or
prior, so every taste model would need refitting. The cache keeps both fixed and refits them
only when asked (`refit_start=True`, say once a week). With the same compression and starting
taste, a cached refresh gives **exactly** the result of a full rebuild; the test checks every
taste model and every score.

| Zone | Full rebuild | One member's answers changed | Nothing changed | Scores in memory |
|---|---:|---:|---:|---:|
| 400 members (Phase 1) | 0.50 s | 0.23 s | 0.07 s | 5 MB |
| 2,000 members | 4.9 s | 3.1 s | 0.5 s | 132 MB |

**Memory is the real limit at city scale.** The scores are five grids of members × members, so
memory grows with the square of the zone's size, to about 3.3 GB at 10,000 members. A city
should run as separate zones of a few thousand members, each with its own cache. That fits how
the business grows anyway (Phase 2: more zones).

---

## 4. Two things found on the way

1. **A flawed first measurement.** The cache reported "refitted 0" because I had picked a
   member who never did the exercise (an inactive member), so the timing tested nothing. With
   a member who has answers it refits exactly one.
2. **A quiet night still took 2 seconds** at 2,000 members, because every refresh recomputed
   every score. The cache now notices when neither the members nor any answers changed. A
   test also checks the opposite case: a profile edit with no new answers still recomputes the
   scores, so the shortcut can't serve stale lists.

---

## 5. Honest limits

- **Freezing has a cost.** Between weekly refits, new members' photos are compressed with last
  week's PCA and new taste models start from last week's starting taste. Both change slowly,
  but a big influx of members is a reason to refit early.
- **A changed answer refits that member only;** it doesn't update the starting taste
  everyone else uses until the weekly refit.
- **Scores are recomputed in full** whenever anything changes. Recomputing only the changed
  rows is possible later if zones grow.

---

## 6. What to study

1. **State machines:** the session's phases (varied photos, pairs, informative photos, done)
   and why "the same question until answered" makes the API safe to call twice.
2. **Idempotency:** guarding answers with the question they answer.
3. **Cache invalidation:** what has to be frozen for incremental results to equal a full
   rebuild, and how to detect "nothing changed" without missing a profile edit.
4. **Memory growth:** n × n matrices grow as the square of n, hence zones.

## Try it yourself

```bash
.venv/bin/python -m pytest tests/test_sprint10.py -q
```

---

## Questions for you

1. **Weekly refit:** is once a week fine for refitting the photo compression and starting
   taste, or should it follow growth (say, every 10% new members)?
2. **Zones for the city launch:** how many members per zone do you expect? Up to about 3,000
   is comfortable on one server.
3. **Sprint 11:**
   - **(a)** re-run the gate every cycle with real data as men leave and women join;
   - **(b)** pause engine work until the app exists and real Phase 1 data arrives;
   - **(c)** anything the app developer asks for after reading the handoff.

   My recommendation is (b), unless the app developer asks for something.

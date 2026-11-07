# Sprint 15: the code review findings

**In one paragraph.** You chose option (d): fix the eight findings still open from the full code
review after Sprint 13 (finding 2 was fixed in Sprint 14). Each one now has a test that fails on
the old code and passes on the new. The most serious was a crash: one cancelled date from
before a man's subscription stopped every woman's list that morning. The rest:
- photos saved sideways by phones now reach the photo model upright;
- the nightly cache notices every changed answer, and forgets a forgotten member completely;
- the engine refuses times written without a time zone;
- a pending pick holds the man's evening in the lists, not just at pick time;
- checking a pick reads only the two members' rows (423 ms → 2 ms in a large zone);
- the model check no longer reads encrypted photo vectors;
- the text model no longer keeps old prompts in memory.

None of this changes a test-run result: the test runs keep their own booking state and don't
call the functions that changed.

**Papers used.** None: these are engineering fixes. Two new decisions record the rules behind
them: D24 (forgetting refits the shared fits) and D25 (times are written with their zone).

---

## 1. What got built

| Finding | Fix | Where |
|---|---|---|
| 1. Crash on a date before a subscription (FR-LM-71) | such a booking counts against none of his cycles; `close_cycle` refuses a day before signup by name | `app._counts_against`, `app.close_cycle` |
| 3. Sideways photos (FR-LM-72) | turned upright from the EXIF orientation tag before resizing | `encoders.PhotoModel.pixels` |
| 4. The cache (FR-LM-73, D24) | compares each member's answers, not their count; drops tastes with no answers left; refits the shared fits the night after someone is forgotten | `app.MarketCache` |
| 5. Times without a zone (FR-LM-74, D25) | refused by the row writers; stored in UTC with `+00:00` | `app.utc_text`, `booking_row`, `answer_row`, `forget_statements` |
| 6. Pending picks (FR-LM-75) | the hold moves into `app.capacity`, shared by the lists and the pick check | `app.capacity`, `app.recheck_pick` |
| 7. Slow pick checks (FR-LM-76) | reads only the two members' bookings, payments, subscriptions and blocks | `app.read`, `ids_in`, `pair_blocked`, `load_bookings(members=)`, `signup_days(members=)` |
| 8. The model check (FR-LM-77) | `SELECT DISTINCT model`, no embeddings | `app.check_photo_models` |
| 9. The text model's memory (FR-LM-78) | keeps only the prompts of its last call | `encoders.SentenceEncoder` |

8 new tests. Three older tests now pass times with a zone (`tzinfo=UTC`), which the new rule
requires; what they check is unchanged. Version 0.15.0.

## 2. The crash (finding 1)

A booking uses a man's dates in his billing cycle. To find the cycle, the engine computes it
from his signup day, and there is no cycle before signup. Two ordinary situations put a booking
before his signup:
- an event date before he subscribed;
- a cancelled date before he let his subscription lapse and joined again, which moves his signup
  later.

For a cancellation the engine asked "which of his cycles was this in?", got an error, and the
whole morning's lists stopped. The fix answers the real question first: a date from before his
current subscription isn't one of his subscription's dates, so its cancellation uses none of
them. (A completed date before signup still marks that day and week as taken, as it should.)

## 3. Sideways photos (finding 3)

A phone's sensor is landscape. For a portrait photo many phones save the pixels sideways and
add one number to the file, the EXIF orientation tag, telling viewers to turn it. Every viewer
does, so nobody notices, but the engine read the raw pixels: SigLIP saw a face lying on its
side, and its vector for that member was off. That skewed every taste model that saw their
photo. `ImageOps.exif_transpose` turns the image upright from the tag first. The test saves a
sideways photo with tag 6 ("turn 90 degrees clockwise") and checks that the pixels reaching the
model equal the upright photo's exactly.

## 4. The cache (finding 4, D24)

The nightly cache refits only the members whose answers changed. It decided "changed" by
counting. Forgetting a member deletes everyone's answers about their photo, so a member could
lose one answer and add one new answer the same evening: same count, different answers, no
refit. Their taste was still fitted on the forgotten member's photo. Now the cache keeps each
member's answers themselves (as database ids) and refits on any difference. A member with no
answers left loses their cached taste.

The forgotten member also lingered in two shared fits: the photo compression (fitted on
everyone's photo vectors) and the starting taste (fitted on everyone's answers). Before, they
were refitted only weekly. Now, the night after someone is forgotten, the refresh refits both,
as the weekly refit does (D24). The test checks both shared fits equal a fresh fit without them.

## 5. Times (finding 5, D25)

D19 says timestamps are stored in UTC, and a stored time without an offset is read as UTC. That
reading rule is right for the database, but it made the writers dangerous. A New York server's
`datetime.now()` has no zone, so 2:05 pm in New York was stored as "14:05" and read back as
14:05 UTC, which is 9:05 am in New York, five hours early. The writers now refuse a time
without a zone (the app passes `datetime.now(UTC)`) and convert any zoned time to UTC with an
explicit `+00:00`. Reading stays lenient, so existing rows still load.

## 6. Pending picks (finding 6)

When she picks him, the booking waits for his answer. Sprint 12 made that pending pick hold his
evening, but only inside `recheck_pick`. The next morning's lists didn't know, so other women
were offered the same evening and their picks were moved or refused when they tried. The hold
now lives in `capacity`, the one place that says which days are taken, so lists and pick checks
agree. Only the day is held, not his week, as in Sprint 12.

## 7. Checking a pick (finding 7)

`recheck_pick` runs inside the transaction that writes the booking, while the two members' rows
are locked. It used to read every booking in the zone and build a members-by-members block
grid: 100 MB at 10,000 members, on every pick. Now it reads only the two members' bookings (and
those bookings' payments), their subscriptions and the block rows between them:

| Zone | Other bookings | Before | After |
|---|---:|---:|---:|
| 400 members | 8,000 | 64.8 ms | 1.4 ms |
| 2,000 members | 40,000 | 422.6 ms | 2.0 ms |

The queries add a WHERE built from whole-number ids (`member_a IN (12, 40)`). Ids are passed
through `int()`, so no text can get into the SQL, and the same code still runs on SQLite and
PostgreSQL without the drivers' different placeholder styles. The test compares every one of a
day's picks with the old version and checks the rows read don't grow as other bookings are
added.

## 8. The model check and the text model's memory (findings 8 and 9)

The nightly model check needs only the model names, but it read every photo embedding row,
encrypted vectors included: about 30 MB of biometric data a night at 10,000 members. It now asks
for `DISTINCT model`.

The text model remembered every prompt it had ever read, so a long-running server's memory
grew with every edit, and forgotten members' prompts stayed in memory. It now keeps just the
prompts of its last call. Each night's call includes every current prompt, so unchanged prompts
are still not re-read.

## 8b. After the sprint: the gate and women who seek only women (v0.15.1)

You spotted a loophole. The ratio gate counted every woman as supply for the men it sells places
to, including women who seek only women. In a zone where 7% of members are gay and 3% bisexual,
that was 25 of 286 women. It was also a hole in Sprint 14's fake-account defence: a fake set to
"woman seeking women" is never shown to a man, and with a pool of about 8 she rarely gets a
list, so she never reaches the 14 list days the silent rule needs. The gate and the weekly
alerts now count only women who seek men, and the gate counts only subscribers who seek women
as the men competing for them (FR-LM-79). Applicants who seek only men were already safe: they
have no pool among these women, so the gate defers them. In three 400-member test zones the
gate sold nothing either way (the forecast stops it below 90%), so the fix changes its inputs,
19 to 25 fewer women counted, rather than that day's decision.

## 9. What changed in the plan

- Two decisions: D24 and D25.
- All nine code review findings are closed.

## 10. Honest limits

- The pick-check timings are SQLite in memory. On PostgreSQL the two-member queries rely on
  the schema's indexes on `bookings (member_a, date_on)` and `(member_b, date_on)`, and on the
  payments' unique `(booking_id, member_id)` key; keep them.
- A pending pick holds only his day, so with one date a week, an evening in the same week can
  still be offered and refused if he accepts the first pick. Holding the week would show him
  less often while he decides; that's a product choice.
- The cache refit after a forgetting changes every taste model a little that night, exactly as
  the weekly refit does.

## 11. What to study

1. **EXIF orientation:** the eight orientation values and which rotation or flip each means.
2. **Time zones:** naive versus aware datetimes, why "no zone" can't be fixed after the fact,
   and why storing UTC with an offset makes the stored value unambiguous.
3. **Caching by content:** why a count is a weak fingerprint; comparing the content (or a
   hash of it) instead.
4. **Query cost:** why reading all rows grows with the zone, while reading by an indexed id
   stays flat; what an index on `(member_a, date_on)` does for `WHERE member_a IN (...)`.

## Try it yourself

```bash
python -m pytest tests/test_sprint15.py -q
```

## Questions for you

1. **Should a pending pick hold his whole week instead of just the evening?** The evening is
   enough to stop double bookings. Holding the week would stop other women being offered
   another evening that week (which is refused if he accepts the first pick), at the cost of
   showing him less while he decides.
2. **How long may he take to accept?** This is question 14 in #8, and it matters more now
   that a pending pick hides his evening from everyone else.

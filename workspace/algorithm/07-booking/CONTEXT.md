# Stage 7: booking

Turns a pick into a date: when, where, and the payment hold. Two versions exist: the weekly
batch scheduler from Sprints 1 to 6, and the real app flow from Sprint 7.

## Takes in / passes on

- **In:** her pick and his acceptance; both people's weekly free slots, existing dates and
  caps; neighbourhoods.
- **Out:** a `Booking` with a day, a part of the day (lunch, evening, late) and a venue, moving
  through proposed, held, confirmed, then completed, cancelled, no-show or expired.

## The real flow (Sprint 7)

- **Window:** each member's free windows over the next 14 days are bits in one 64-bit number;
  the proposal is the lowest bit set in both (`booking.window_bits`,
  `booking.earliest_common`). Days with a date, full weeks and full months are blocked.
- **Hold:** both $10 payments must arrive within 15 minutes, or the booking expires
  (`Booking.hold`, `Booking.pay`). Illegal moves raise `BookingError`.
- **Venue:** the one that keeps the longer of the two trips shortest, ties broken by the total
  trip (`geo.venue_for`). Members give a neighbourhood, never an address.
- **Event dates:** `booking.event_booking`, the same rules.

## The batch scheduler (Sprints 1 to 6)

Weekly b-matching: min-cost flow when the graph splits into two sides, an integer program
(SCIP) when it has odd cycles (`scheduler.schedule_week`), with end-of-month urgency.

## Code

`src/lastfirstdate/booking.py`, `geo.py`, `scheduler.py`.

## Settings

`[booking]`, `[dates]` (caps, `max_per_week`, `late_month_bonus`), `[geo]`.

## Tests

FR-LM-10 (`tests/test_scheduler.py`), FR-LM-37, FR-LM-40, FR-LM-41 (`tests/test_sprint7.py`).

## Papers

13, Su et al. (capacity matters); 15 and 16 (congestion).

## Decisions

D4 (dates per month and caps), D17 (his own late cancellation or no-show counts as
delivered).

## What we know

- About one booked date in three falls through: the hold expires, someone cancels, or someone
  doesn't show up. Most are refilled in time.
- Whoever is stood up gets their week and cap slot back (`CycleSimulation.release_slot`).
  Sprint 7 missed this, which understated delivered dates (corrected in Sprint 8).
- Allowing 2 dates a week instead of 1 didn't help.
- Venue trips averaged under 1 km in Phase 1.

## If you change it

Keep `db/schema.sql`'s booking states the same as `booking.STATES` (tested). Rerun
`python -m lastfirstdate.cycle_sim`.

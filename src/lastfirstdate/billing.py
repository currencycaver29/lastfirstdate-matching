"""Billing cycles, delivered dates, refills, credits and date fees (FR-LM-38).

lastfirstdate's rules: $200 per cycle after a 30-day free trial; 3 guaranteed dates per cycle,
pro-rated credit if fewer. A date is delivered when both confirmed a venue and time, both
paid, and it is scheduled before his cycle ends; no-shows and cancellations don't count and the
slot is refilled. Each person pays $10 at booking: full refund 24 hours or more ahead, 50%
later, nothing for a no-show.

The site doesn't say what happens when *he* is the one who cancels late or doesn't come. If
that refilled his slot, a man could skip dates and still collect credits, so by default his
own late cancellation or no-show counts as delivered (`billing.own_absence_counts`, D17).

Refill. During a cycle, his missing count is 3 minus the dates delivered minus the confirmed
dates still to come. When one of those falls through on her side, his missing count goes back
up and so does his bonus on women's lists:

    bonus_j = guarantee_priority * missing_j / weeks left in HIS cycle

Each man has his own cycle, so "weeks left" differs from man to man (Sprints 4 to 6 used one
calendar month for everyone).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from lastfirstdate.booking import Booking
from lastfirstdate.config import Config


@dataclass(frozen=True)
class Cycle:
    member: int
    index: int  # -1 for the free trial, then 0, 1, 2, ... for paid cycles
    start: int  # first day
    end: int  # first day of the next cycle
    price: float  # 0 during the free trial

    def contains(self, day: int) -> bool:
        return self.start <= day < self.end

    def days_left(self, today: int) -> int:
        return max(0, self.end - today)


def cycle_on(member: int, signup_day: int, day: int, cfg: Config) -> Cycle:
    """The billing cycle a member is in on `day`.

    The free trial runs for `billing.trial_days` from signup (index -1, price 0; the
    guarantee still applies, but there is nothing to credit). Paid cycles of
    `billing.cycle_days` follow back to back.
    """
    b = cfg.billing
    if day < signup_day:
        raise ValueError("no cycle before signup")
    paid_from = signup_day + b.trial_days
    if day < paid_from:
        return Cycle(member, -1, signup_day, paid_from, 0.0)
    index = (day - paid_from) // b.cycle_days
    start = paid_from + index * b.cycle_days
    return Cycle(member, index, start, start + b.cycle_days, b.price_per_cycle)


def counts_toward_guarantee(booking: Booking, member: int, cycle: Cycle, cfg: Config) -> bool:
    """Does this booking count as one of `member`'s delivered dates in `cycle`?"""
    if member not in booking.members or not cycle.contains(booking.day):
        return False
    if booking.state == "completed":
        return True
    if not cfg.billing.own_absence_counts:
        return False
    if booking.state == "cancelled":
        late = booking.hours_before < cfg.billing.full_refund_hours
        return booking.cancelled_by == member and late
    if booking.state == "no_show":
        return not booking.showed[member]
    return False


def delivered(bookings: Iterable[Booking], member: int, cycle: Cycle, cfg: Config) -> int:
    return sum(counts_toward_guarantee(bk, member, cycle, cfg) for bk in bookings)


def still_to_come(bookings: Iterable[Booking], member: int, cycle: Cycle) -> int:
    """Confirmed dates in the cycle that haven't happened yet."""
    return sum(
        bk.state == "confirmed" and member in bk.members and cycle.contains(bk.day)
        for bk in bookings
    )


def missing(delivered_dates: int, upcoming: int, guarantee: int) -> int:
    return max(0, guarantee - delivered_dates - upcoming)


def credit(missing_at_end: int, cycle: Cycle, cfg: Config) -> float:
    """Pro-rated credit at the end of a cycle: price * missing / guarantee."""
    g = cfg.dates.guaranteed_per_month_paying
    return cycle.price * min(missing_at_end, g) / g


def refill_bonus(missing_now: np.ndarray, days_left: np.ndarray, weight: float) -> np.ndarray:
    """(n,) list bonus: weight * missing / weeks left in each member's own cycle."""
    weeks_left = np.maximum(1, np.ceil(np.asarray(days_left) / 7))
    return weight * np.asarray(missing_now) / weeks_left


def fee_refunds(booking: Booking, cfg: Config) -> dict[int, float]:
    """Dollars of the $10 date fee given back to each member who paid it.

    Completed: nothing (the fee paid for the date). Cancelled: the one who cancelled gets it
    all back at least `full_refund_hours` ahead, `late_refund_share` of it later; the other
    gets it all back. No-show: nothing for whoever didn't come, all of it for whoever did.
    Expired: whoever had paid gets it all back.
    """
    b = cfg.billing
    out = dict.fromkeys(booking.paid, 0.0)
    for m in booking.paid:
        if booking.state == "cancelled" and m == booking.cancelled_by:
            on_time = booking.hours_before >= b.full_refund_hours
            out[m] = b.date_fee * (1.0 if on_time else b.late_refund_share)
        elif booking.state in ("cancelled", "expired"):
            out[m] = b.date_fee
        elif booking.state == "no_show":
            out[m] = b.date_fee if booking.showed[m] else 0.0
    return out


def hours_before(booking: Booking, cancel_day: float) -> float:
    """Hours between a cancellation on (fractional) day `cancel_day` and the date's day."""
    return max(0.0, (booking.day - cancel_day) * 24.0)


def weeks_left(days: int) -> int:
    return max(1, math.ceil(days / 7))

"""Proposed date windows and the life of a booking (FR-LM-37, FR-LM-40).

lastfirstdate's flow: his profile appears on her daily list with a proposed date window; she
chooses him; both $10 payments must arrive within 15 minutes; then the venue is revealed and
chat opens.

Windows. Days are numbered from a Monday (day 0). Each day has three parts (lunch, evening,
late), so the next `horizon_days` days give 3 * horizon_days possible windows, at most 63, and
a member's free windows fit in one 64-bit number: bit 3k + part is set when they are free on
day today + k at that part of the day. Their weekly free slots repeat every week; days they
cannot date (a date already that day, their week or month already full) are blocked. The
proposal for a pair is the lowest bit set in both numbers: the earliest window both can make.
Proposing at list time replaces the weekly batch schedule of Sprints 1 to 6.

Life of a booking:

    proposed -> held -> confirmed -> completed
        |         |          |-----> cancelled   (by one of them, some hours ahead)
        |         |          '-----> no_show     (one or both did not come)
        '---------'-----> expired               (the payment hold ran out)

Any other move raises BookingError. Event dates (FR-LM-40) are bookings with
source = "event": same rules, they just don't come from a daily list.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from lastfirstdate import native
from lastfirstdate.models import PARTS_OF_DAY

N_PARTS = len(PARTS_OF_DAY)
STATES = ("proposed", "held", "confirmed", "completed", "cancelled", "no_show", "expired")
SOURCES = ("list", "event")
FINAL = frozenset({"completed", "cancelled", "no_show", "expired"})
MOVES = {
    "proposed": frozenset({"held", "expired"}),
    "held": frozenset({"confirmed", "expired"}),
    "confirmed": frozenset({"completed", "cancelled", "no_show"}),
}
NO_WINDOW = -1


class BookingError(ValueError):
    """An illegal move in a booking's life, or a payment after the hold ran out."""


# ---------------------------------------------------------------- proposed windows


def window_bits(
    weekly_slots: np.ndarray,
    today: int,
    lead_days: int,
    horizon_days: int,
    blocked: np.ndarray | None = None,
) -> np.ndarray:
    """(n,) free windows over the next `horizon_days` days, one bit per (day, part).

    `weekly_slots` holds each member's free weekly slots as bits (slot = weekday * 3 + part,
    as in `MemberArrays.slots`); `blocked` (n, horizon_days) marks days a member cannot date.
    Days before `lead_days` are never offered. Rust when available (`native.py`); the loop
    below is the same rule in Python.
    """
    if not 0 <= lead_days < horizon_days <= 21:
        raise ValueError("need 0 <= lead_days < horizon_days <= 21")
    weekly = np.asarray(weekly_slots, dtype=np.int64)
    if native.enabled() and weekly.ndim == 1:
        mask = None if blocked is None else np.ascontiguousarray(blocked, dtype=bool)
        return native.core.window_bits(
            np.ascontiguousarray(weekly), int(today), lead_days, horizon_days, mask
        )
    out = np.zeros(weekly.shape, dtype=np.uint64)
    for k in range(lead_days, horizon_days):
        weekday = (today + k) % 7
        day = ((weekly >> (weekday * N_PARTS)) & 0b111).astype(np.uint64)
        if blocked is not None:
            day = np.where(blocked[:, k], np.uint64(0), day)
        out |= day << np.uint64(k * N_PARTS)
    return out


def earliest_common(bits_a: np.ndarray, bits_b: np.ndarray) -> np.ndarray:
    """The earliest window free for both (an index into window_bits), or NO_WINDOW.

    Broadcasts, so (n, 1) against (1, m) gives every pair at once. Rust (one instruction per
    pair: count the trailing zeros) when available and the inputs are one member against
    many or pair by pair; the lines below are the same rule in NumPy.
    """
    a, b = np.asarray(bits_a, dtype=np.uint64), np.asarray(bits_b, dtype=np.uint64)
    if (
        native.enabled()
        and a.ndim <= 1
        and b.ndim <= 1
        and (1 in (a.size, b.size) or a.size == b.size)
    ):
        out = native.core.earliest_common(
            np.ascontiguousarray(a.reshape(-1)), np.ascontiguousarray(b.reshape(-1))
        )
        return out.reshape(a.shape if a.size > b.size or b.ndim == 0 else b.shape)
    common = a & b
    with np.errstate(over="ignore"):  # ~0 + 1 wraps to 0 on purpose
        lowest = common & (~common + np.uint64(1))  # keeps only the lowest set bit
    with np.errstate(divide="ignore"):
        index = np.rint(np.log2(lowest.astype(np.float64)))
    return np.where(common == 0, NO_WINDOW, index).astype(np.int64)


def window_day_part(window: int, today: int) -> tuple[int, int]:
    """The (day, part of day) a window index stands for."""
    if window < 0:
        raise ValueError("no window")
    k, part = divmod(int(window), N_PARTS)
    return today + k, part


# ---------------------------------------------------------------- the booking itself


@dataclass
class Booking:
    """One date between members a and b (for list dates, a is the one who chose)."""

    a: int
    b: int
    day: int
    part: int
    source: str = "list"
    state: str = "proposed"
    held_at: float | None = None  # minute the payment hold started
    paid: set[int] = field(default_factory=set)
    venue: str | None = None
    cancelled_by: int | None = None
    hours_before: float | None = None
    showed: dict[int, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise BookingError("a booking needs two different members")
        if self.source not in SOURCES:
            raise BookingError(f"unknown source {self.source!r}")
        if not 0 <= self.part < N_PARTS:
            raise BookingError(f"part of day must be 0 to {N_PARTS - 1}")

    @property
    def members(self) -> tuple[int, int]:
        return self.a, self.b

    def other(self, member: int) -> int:
        self._check_member(member)
        return self.b if member == self.a else self.a

    def _check_member(self, member: int) -> None:
        if member not in self.members:
            raise BookingError(f"member {member} is not in this booking")

    def _move(self, new: str) -> None:
        if new not in MOVES.get(self.state, ()):
            raise BookingError(f"a {self.state} booking cannot become {new}")
        self.state = new

    def hold(self, now: float) -> None:
        """Both have said yes: start the payment hold."""
        self._move("held")
        self.held_at = now

    def hold_expired(self, now: float, hold_minutes: float) -> bool:
        return self.state == "held" and now - self.held_at > hold_minutes

    def pay(self, member: int, now: float, hold_minutes: float, venue: str | None = None) -> bool:
        """Record a member's $10 payment. Confirms the date when both have paid.

        A payment after the hold has run out expires the booking instead and returns False.
        """
        self._check_member(member)
        if self.state != "held":
            raise BookingError(f"cannot pay for a {self.state} booking")
        if self.hold_expired(now, hold_minutes):
            self._move("expired")
            return False
        self.paid.add(member)
        if len(self.paid) == 2:
            self._move("confirmed")
            self.venue = venue
        return True

    def expire(self) -> None:
        """The hold ran out (or a proposal was never taken up); the window is free again."""
        self._move("expired")

    def cancel(self, member: int, hours_before: float) -> None:
        self._check_member(member)
        if hours_before < 0:
            raise BookingError("a cancellation must come before the date")
        self._move("cancelled")
        self.cancelled_by = member
        self.hours_before = hours_before

    def finish(self, showed_a: bool, showed_b: bool) -> None:
        """The date's time has come: completed if both came, no_show otherwise."""
        self._move("completed" if showed_a and showed_b else "no_show")
        self.showed = {self.a: bool(showed_a), self.b: bool(showed_b)}

    @property
    def is_final(self) -> bool:
        return self.state in FINAL


def event_booking(a: int, b: int, day: int, part: int) -> Booking:
    """A date between two members who met at a members-only event (FR-LM-40)."""
    return Booking(a, b, day, part, source="event")


def dated_pairs(bookings: Iterable[Booking]) -> set[tuple[int, int]]:
    """Pairs with a booking that went ahead (not expired): never shown to each other again."""
    return {(min(bk.a, bk.b), max(bk.a, bk.b)) for bk in bookings if bk.state != "expired"}

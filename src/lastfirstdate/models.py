"""Member schema (FR-LM-02).

Only fields a real lastfirstdate profile would have. There is deliberately no
field for overall attractiveness, desirability or any global rank (rule R2).
Photo embeddings live in a separate matrix indexed by member id, so they can be
encrypted and deleted on their own (rule R3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Gender = Literal["man", "woman", "nonbinary"]

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
PARTS_OF_DAY = ("lunch", "evening", "late")
N_SLOTS = len(DAYS) * len(PARTS_OF_DAY)  # 21 weekly slots

BIG5_TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")


def slot_id(day: str, part: str) -> int:
    """Weekly slot number for a day and part of day, e.g. ('fri', 'evening') -> 13."""
    return DAYS.index(day) * len(PARTS_OF_DAY) + PARTS_OF_DAY.index(part)


def slot_name(slot: int) -> str:
    """Readable name for a weekly slot number, e.g. 13 -> 'fri evening'."""
    if not 0 <= slot < N_SLOTS:
        raise ValueError(f"slot must be in 0..{N_SLOTS - 1}, got {slot}")
    day, part = divmod(slot, len(PARTS_OF_DAY))
    return f"{DAYS[day]} {PARTS_OF_DAY[part]}"


@dataclass(frozen=True, eq=False)
class Member:
    """One member profile, as the matching engine sees it."""

    id: int
    gender: Gender
    seeking: frozenset[str]
    age: int
    age_min_pref: int
    age_max_pref: int
    lat: float
    lon: float
    max_km: float
    verified: bool
    last_active_days: int
    paying: bool
    free_slots: frozenset[int]
    interests: tuple[str, ...]
    bio: str
    values: dict[str, str]
    accepts: dict[str, frozenset[str]]
    dealbreakers: frozenset[str]
    big5: tuple[float, float, float, float, float]
    # Used ONLY for fairness audits of who gets shown and dated. Never a matching input.
    audit_group: str

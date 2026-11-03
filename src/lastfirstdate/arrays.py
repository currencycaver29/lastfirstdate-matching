"""Member profiles as numpy arrays, so every pair can be scored at once.

With 1,000 members there are about a million pairs. Looping over them in Python is slow;
holding each field as one array lets numpy compare all pairs in a few operations.

Small trick used throughout: sets are stored as bit masks. Member i's free slots
{0, 3, 13} become the integer 2**0 + 2**3 + 2**13. Two members share a slot exactly
when `slots[i] & slots[j]` is not zero, and `np.bitwise_count` counts how many they share.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lastfirstdate.config import ValuesQuestion
from lastfirstdate.models import Member

GENDERS = ("man", "woman", "nonbinary")
GENDER_BIT = {g: 1 << k for k, g in enumerate(GENDERS)}
EARTH_RADIUS_KM = 6371.0


def mask(items, index: dict) -> int:
    """Bit mask with one bit set per item."""
    out = 0
    for item in items:
        out |= 1 << index[item]
    return out


@dataclass(frozen=True)
class MemberArrays:
    n: int
    gender: np.ndarray  # (n,) index into GENDERS
    gender_bit: np.ndarray  # (n,) 1 << gender
    seeking: np.ndarray  # (n,) bit mask of genders sought
    age: np.ndarray
    age_min: np.ndarray
    age_max: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    max_km: np.ndarray
    verified: np.ndarray
    last_active: np.ndarray
    paying: np.ndarray
    slots: np.ndarray  # (n,) bit mask of free weekly slots
    answer: np.ndarray  # (n, q) index of each member's answer per values question
    accepts: np.ndarray  # (n, q) bit mask of accepted answers per question
    dealbreaker: np.ndarray  # (n, q) True where the question is a dealbreaker
    big5: np.ndarray  # (n, 5)

    @classmethod
    def from_members(
        cls, members: Sequence[Member], questions: Sequence[ValuesQuestion]
    ) -> MemberArrays:
        n = len(members)
        for k, m in enumerate(members):
            if m.id != k:
                raise ValueError("members must be ordered so that members[k].id == k")
        q_index = [{opt: k for k, opt in enumerate(q.options)} for q in questions]
        answer = np.array(
            [[q_index[k][m.values[q.key]] for k, q in enumerate(questions)] for m in members],
            dtype=np.int64,
        ).reshape(n, len(questions))
        accepts = np.array(
            [
                [mask(m.accepts[q.key], q_index[k]) for k, q in enumerate(questions)]
                for m in members
            ],
            dtype=np.int64,
        ).reshape(n, len(questions))
        dealbreaker = np.array(
            [[q.key in m.dealbreakers for q in questions] for m in members], dtype=bool
        ).reshape(n, len(questions))
        gender = np.array([GENDERS.index(m.gender) for m in members], dtype=np.int64)
        return cls(
            n=n,
            gender=gender,
            gender_bit=np.left_shift(1, gender),
            seeking=np.array(
                [mask(m.seeking, {g: k for k, g in enumerate(GENDERS)}) for m in members]
            ),
            age=np.array([m.age for m in members]),
            age_min=np.array([m.age_min_pref for m in members]),
            age_max=np.array([m.age_max_pref for m in members]),
            lat=np.array([m.lat for m in members]),
            lon=np.array([m.lon for m in members]),
            max_km=np.array([m.max_km for m in members]),
            verified=np.array([m.verified for m in members]),
            last_active=np.array([m.last_active_days for m in members]),
            paying=np.array([m.paying for m in members]),
            slots=np.array(
                [mask(m.free_slots, {s: s for s in range(64)}) for m in members], dtype=np.int64
            ),
            answer=answer,
            accepts=accepts,
            dealbreaker=dealbreaker,
            big5=np.array([m.big5 for m in members], dtype=float).reshape(n, 5),
        )


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on scalars or broadcastable arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def distance_matrix(arr: MemberArrays) -> np.ndarray:
    """(n, n) straight-line distance in km between every pair of members."""
    return haversine_km(arr.lat[:, None], arr.lon[:, None], arr.lat[None, :], arr.lon[None, :])


def shared_slots(arr: MemberArrays) -> np.ndarray:
    """(n, n) number of free weekly slots each pair has in common."""
    return np.bitwise_count(arr.slots[:, None] & arr.slots[None, :]).astype(np.int64)

"""Neighbourhoods and the venue midpoint (FR-LM-41).

Members give a neighbourhood, never an address, so the app never holds anyone's exact
location. Distances use neighbourhood centres. lastfirstdate picks a venue at the "location
midpoint"; with a real list of venues the closest one to the geographic midpoint can still be
much closer to one person than the other, so the rule here is fairer and just as simple:

    venue = the one that makes the longer of the two trips as short as possible,
            ties broken by the shorter total trip

(a "minimax" choice). For two people in the same neighbourhood it is simply the venue nearest
to them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lastfirstdate.arrays import haversine_km
from lastfirstdate.config import Place


def coordinates(places: Sequence[Place]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([p.lat for p in places]), np.array([p.lon for p in places])


def nearest_place(lat, lon, places: Sequence[Place]) -> np.ndarray:
    """Index of the nearest place for each (lat, lon) given (scalars or arrays)."""
    plat, plon = coordinates(places)
    d = haversine_km(np.asarray(lat)[..., None], np.asarray(lon)[..., None], plat, plon)
    return np.argmin(d, axis=-1)


def venue_for(a: Place, b: Place, venues: Sequence[Place]) -> tuple[int, float, float]:
    """The venue index for a date between people at a and b, and each one's trip in km."""
    vlat, vlon = coordinates(venues)
    da = haversine_km(a.lat, a.lon, vlat, vlon)
    db = haversine_km(b.lat, b.lon, vlat, vlon)
    longer, total = np.maximum(da, db), da + db
    order = np.lexsort((total, longer))  # by the longer trip, then the total
    best = int(order[0])
    return best, float(da[best]), float(db[best])

"""FR-LM-04: Stage 1 hard filter."""

import dataclasses

import numpy as np
import pytest

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.filter import eligibility_matrix, pair_passes


def _matrix(members, cfg, excluded=None):
    return eligibility_matrix(MemberArrays.from_members(members, cfg.values), cfg, excluded)


def test_filter_is_symmetric_with_empty_diagonal(cfg, small_pop):
    """FR-LM-04: if i may meet j then j may meet i, and nobody is matched with themselves."""
    e = _matrix(small_pop.members, cfg)
    assert np.array_equal(e, e.T)
    assert not e.diagonal().any()
    assert e.any()


def test_filter_matches_plain_python_reference(cfg, small_pop):
    """FR-LM-04: the fast numpy filter agrees with the one-pair-at-a-time reference."""
    members = small_pop.members[:150]
    e = _matrix(members, cfg)
    for a in members:
        for b in members:
            assert e[a.id, b.id] == pair_passes(a, b, cfg), (a.id, b.id)


@pytest.fixture()
def passing_pair(cfg, small_pop):
    """Two members (ids 0 and 1) who pass every rule."""
    e = _matrix(small_pop.members, cfg)
    i, j = map(int, np.argwhere(e)[0])
    a = dataclasses.replace(small_pop.members[i], id=0)
    b = dataclasses.replace(small_pop.members[j], id=1)
    assert _matrix([a, b], cfg)[0, 1]
    return a, b


def _flip_dealbreaker(a, b, cfg):
    """Make b's answer to one question something a refuses, and make it a dealbreaker for a."""
    q = cfg.values[0]
    refused = next(o for o in q.options if o != a.values[q.key])
    accepts = dict(a.accepts)
    accepts[q.key] = frozenset({a.values[q.key]})
    values = dict(b.values)
    values[q.key] = refused
    return (
        dataclasses.replace(a, accepts=accepts, dealbreakers=a.dealbreakers | {q.key}),
        dataclasses.replace(b, values=values),
    )


BREAKERS = {
    "unverified": lambda a, b, cfg: (a, dataclasses.replace(b, verified=False)),
    "inactive": lambda a, b, cfg: (
        a,
        dataclasses.replace(b, last_active_days=cfg.filter.active_within_days + 1),
    ),
    "orientation": lambda a, b, cfg: (
        dataclasses.replace(a, seeking=frozenset({"nonbinary"})),
        b,
    ),
    "age": lambda a, b, cfg: (a, dataclasses.replace(b, age=a.age_max_pref + 1)),
    "distance": lambda a, b, cfg: (a, dataclasses.replace(b, lat=b.lat + 1.0)),
    "dealbreaker": _flip_dealbreaker,
    "no shared slot": lambda a, b, cfg: (
        dataclasses.replace(a, free_slots=frozenset({0})),
        dataclasses.replace(b, free_slots=frozenset({1})),
    ),
}


@pytest.mark.parametrize("rule", sorted(BREAKERS))
def test_each_rule_removes_its_counterexample(cfg, passing_pair, rule):
    """FR-LM-04: breaking any single rule removes the pair, in both directions."""
    a, b = BREAKERS[rule](*passing_pair, cfg)
    e = _matrix([a, b], cfg)
    assert not e[0, 1] and not e[1, 0]
    assert not pair_passes(a, b, cfg)


def test_already_dated_or_blocked_pairs_are_excluded(cfg, passing_pair):
    """FR-LM-04: pairs marked as already dated or blocked never pass."""
    excluded = np.array([[False, True], [True, False]])
    assert not _matrix(list(passing_pair), cfg, excluded)[0, 1]


def test_unverified_allowed_when_not_required(cfg, passing_pair):
    """FR-LM-04: the verified rule follows filter.require_verified."""
    a, b = passing_pair
    b = dataclasses.replace(b, verified=False)
    relaxed = dataclasses.replace(
        cfg, filter=dataclasses.replace(cfg.filter, require_verified=False)
    )
    assert _matrix([a, b], relaxed)[0, 1]

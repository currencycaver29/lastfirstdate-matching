"""FR-LM-05 to FR-LM-08: Stage 4 personality and interests score."""

import dataclasses

import numpy as np

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.personality import (
    TagEncoder,
    big5_similarity,
    interest_similarity,
    personality_matrix,
    values_compatibility,
)


def _with(member, i, **changes):
    return dataclasses.replace(member, id=i, **changes)


def test_interest_similarity_bounds(small_pop):
    """FR-LM-05: identical interests give 1, no overlap gives 0, always within 0..1."""
    base = small_pop.members[0]
    a = _with(base, 0, interests=("hiking", "jazz", "coffee"))
    b = _with(base, 1, interests=("hiking", "jazz", "coffee"))
    c = _with(base, 2, interests=("opera", "chess"))
    t = interest_similarity(TagEncoder().encode([a, b, c]))
    assert np.isclose(t[0, 1], 1.0)
    assert np.isclose(t[0, 2], 0.0)
    t_all = interest_similarity(TagEncoder().encode(small_pop.members))
    assert t_all.min() >= 0.0 and t_all.max() <= 1.0 + 1e-12


def test_rare_shared_interest_counts_more(small_pop):
    """FR-LM-05: sharing a rare interest scores higher than sharing a common one."""
    base = small_pop.members[0]
    people = [_with(base, 0, interests=("coffee", "opera")), _with(base, 1, interests=("coffee",))]
    people.append(_with(base, 2, interests=("opera",)))
    # five more members who all like coffee make coffee common
    people += [_with(base, k, interests=("coffee", f"x{k}")) for k in range(3, 8)]
    t = interest_similarity(TagEncoder().encode(people))
    assert t[0, 2] > t[0, 1]


def _values_pair(cfg, small_pop, a_answer, b_answer, a_accepts, b_accepts):
    base = small_pop.members[0]
    keys = [q.key for q in cfg.values]
    va = {k: base.values[k] for k in keys}
    vb = dict(va)
    acc_a = {k: frozenset(q.options) for k, q in zip(keys, cfg.values, strict=True)}
    acc_b = dict(acc_a)
    k0 = keys[0]
    va[k0], vb[k0] = a_answer, b_answer
    acc_a[k0], acc_b[k0] = frozenset(a_accepts), frozenset(b_accepts)
    a = _with(base, 0, values=va, accepts=acc_a, dealbreakers=frozenset())
    b = _with(base, 1, values=vb, accepts=acc_b, dealbreakers=frozenset())
    return values_compatibility(MemberArrays.from_members([a, b], cfg.values))


def test_values_compatibility_needs_both_sides(cfg, small_pop):
    """FR-LM-06: a question counts only if each one's answer is accepted by the other."""
    nq = len(cfg.values)
    both = _values_pair(cfg, small_pop, "want", "open", {"want", "open"}, {"want", "open"})
    assert np.isclose(both[0, 1], 1.0)
    one_way = _values_pair(
        cfg, small_pop, "want", "dont_want", {"want", "dont_want"}, {"dont_want"}
    )
    assert np.isclose(one_way[0, 1], (nq - 1) / nq)
    assert np.isclose(one_way[1, 0], one_way[0, 1])


def test_big5_similarity_bounds(cfg, small_pop):
    """FR-LM-07: identical Big Five gives 1, opposite corners give 0."""
    base = small_pop.members[0]
    a = _with(base, 0, big5=(0.0,) * 5)
    b = _with(base, 1, big5=(0.0,) * 5)
    c = _with(base, 2, big5=(1.0,) * 5)
    p = big5_similarity(MemberArrays.from_members([a, b, c], cfg.values))
    assert np.isclose(p[0, 1], 1.0)
    assert np.isclose(p[0, 2], 0.0)


def test_personality_score_is_the_weighted_sum(cfg, small_pop):
    """FR-LM-08: P = w_text t + w_values A + w_big5 p, symmetric and within 0..1."""
    m = small_pop.members
    arr = MemberArrays.from_members(m, cfg.values)
    p = personality_matrix(m, arr, cfg)
    w = cfg.personality
    expected = (
        w.w_text * interest_similarity(TagEncoder().encode(m))
        + w.w_values * values_compatibility(arr)
        + w.w_big5 * big5_similarity(arr)
    )
    np.testing.assert_allclose(p, expected)
    np.testing.assert_allclose(p, p.T)
    assert p.min() >= 0.0 and p.max() <= 1.0 + 1e-12


def test_personality_weights_come_from_config(cfg, small_pop):
    """FR-LM-08: changing the weights in config changes the score."""
    m = small_pop.members
    arr = MemberArrays.from_members(m, cfg.values)
    only_big5 = dataclasses.replace(
        cfg, personality=dataclasses.replace(cfg.personality, w_text=0.0, w_values=0.0, w_big5=1.0)
    )
    np.testing.assert_allclose(personality_matrix(m, arr, only_big5), big5_similarity(arr))

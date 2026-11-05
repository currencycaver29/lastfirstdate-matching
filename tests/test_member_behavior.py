"""FR-LM-11: behavior test harness (fake members decide from hidden truth only)."""

import numpy as np

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.member_behavior import Behavior


def _behavior(cfg, pop, seed=0):
    arr = MemberArrays.from_members(pop.members, cfg.values)
    return Behavior(pop, arr, cfg, np.random.default_rng(seed))


def test_member_behavior_is_deterministic(cfg, small_pop):
    """FR-LM-11 / R7: the same seed gives the same true appeal and the same choices."""
    a, b = _behavior(cfg, small_pop, 3), _behavior(cfg, small_pop, 3)
    np.testing.assert_array_equal(a.utility, b.utility)
    shown = list(range(1, 31))
    assert a.choose(0, shown, np.random.default_rng(9)) == b.choose(
        0, shown, np.random.default_rng(9)
    )


def test_members_pick_profiles_they_truly_prefer(cfg, small_pop):
    """FR-LM-11: across many choosers, truly preferred profiles get picked far more often."""
    beh = _behavior(cfg, small_pop)
    rng = np.random.default_rng(4)
    top_picks = bottom_picks = 0
    for i in range(100):
        u = beh.utility[i].copy()
        u[i] = np.nan
        order = np.argsort(np.nan_to_num(u, nan=-np.inf))
        bottom, top = order[1:11], order[-10:]  # order[0] is i itself
        for _ in range(5):
            top_picks += len(beh.choose(i, top, rng))
            bottom_picks += len(beh.choose(i, bottom, rng))
    assert top_picks > 5 * max(bottom_picks, 1)


def test_earlier_passes_lower_the_chance_of_a_pick(cfg, small_pop):
    """FR-LM-11: someone already passed over is less likely to be picked when shown again."""
    beh = _behavior(cfg, small_pop)
    shown = np.arange(1, 31)
    fresh = again = 0
    for k in range(300):
        fresh += len(beh.choose(0, shown, np.random.default_rng(k)))
        again += len(beh.choose(0, shown, np.random.default_rng(k), np.full(30, 2)))
    assert again < fresh


def test_picks_respect_the_daily_cap(cfg, small_pop):
    """FR-LM-11: a chooser never picks more than test_runs.picks_per_day profiles in a day."""
    beh = _behavior(cfg, small_pop)
    for k in range(50):
        assert len(beh.choose(k, np.arange(100, 130), np.random.default_rng(k))) <= (
            cfg.test_runs.picks_per_day
        )


def test_calibration_swipes_follow_true_looks(cfg, small_pop):
    """FR-LM-11: photo swipes depend on true looks appeal: better-looking-to-her gets more likes."""
    beh = _behavior(cfg, small_pop)
    rng = np.random.default_rng(5)
    top = bottom = 0
    for i in range(100):
        order = np.argsort(beh.looks[i])
        order = order[order != i]
        top += sum(beh.swipe(i, int(j), rng) for j in order[-10:])
        bottom += sum(beh.swipe(i, int(j), rng) for j in order[:10])
    assert top > 3 * max(bottom, 1)

"""FR-LM-13 to FR-LM-17: the looks model, calibration and the looks report."""

import dataclasses

import numpy as np
import pytest

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.looks import (
    PCA,
    ArrayPhotoEncoder,
    CalibrationIncomplete,
    PersonalTaste,
    StartingTaste,
    Swipe,
    calibrate,
    calibration_pool,
    fit_logistic,
    photo_features,
    select_varied,
    sigmoid,
    with_intercept,
)
from lastfirstdate.looks_report import auc, flag_groups, run_looks_report
from lastfirstdate.synthetic import generate


@pytest.fixture(scope="module")
def report(cfg):
    return run_looks_report(generate(cfg), cfg, max_viewers=200)


# ---------------------------------------------------------------- FR-LM-13


def test_photo_features_shape_and_scale(cfg, small_pop):
    """FR-LM-13: PCA shrinks embeddings to pca_dim numbers plus an intercept column."""
    pca, x = photo_features(ArrayPhotoEncoder(small_pop.photo_embeddings), len(small_pop), cfg)
    assert x.shape == (len(small_pop), cfg.calibration.pca_dim + 1)
    np.testing.assert_allclose(x[:, -1], 1.0)
    np.testing.assert_allclose(x[:, :-1].mean(axis=0), 0.0, atol=1e-9)
    assert np.isclose(np.mean(np.sum(x[:, :-1] ** 2, axis=1)), 1.0)
    np.testing.assert_allclose(
        pca.components @ pca.components.T, np.eye(pca.components.shape[0]), atol=1e-9
    )


def test_pca_keeps_most_of_the_variance(cfg):
    """FR-LM-13: on synthetic photos, 64 of 128 numbers keep most of the variation."""
    pop = generate(cfg)
    pca = PCA.fit(pop.photo_embeddings, cfg.calibration.pca_dim)
    assert pca.explained_total > 0.75
    assert np.all(np.diff(pca.explained) <= 1e-12)  # largest first


def test_array_encoder_returns_requested_rows(small_pop):
    """FR-LM-13: the offline encoder returns the stored embedding for each id."""
    enc = ArrayPhotoEncoder(small_pop.photo_embeddings)
    np.testing.assert_array_equal(enc.encode([3, 1]), small_pop.photo_embeddings[[3, 1]])


# ---------------------------------------------------------------- logistic core


def test_logistic_fit_recovers_known_weights():
    """FR-LM-15: with plenty of data the fit recovers the weights that generated it."""
    rng = np.random.default_rng(0)
    true_w = np.array([1.5, -2.0, 0.5])
    x = with_intercept(rng.normal(size=(20000, 2)))
    y = rng.random(20000) < sigmoid(x @ true_w)
    w, cov = fit_logistic(x, y, l2=1e-3)
    np.testing.assert_allclose(w, true_w, atol=0.08)
    assert np.all(np.diag(cov) < 0.01)


def test_penalty_keeps_weights_near_the_prior_with_little_data():
    """FR-LM-15: with three swipes the personal correction stays near zero."""
    x = with_intercept(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    w_strong, _ = fit_logistic(x, np.array([1, 0, 1]), l2=100.0)
    w_weak, _ = fit_logistic(x, np.array([1, 0, 1]), l2=0.01)
    assert np.linalg.norm(w_strong) < 0.05 < np.linalg.norm(w_weak)


# ---------------------------------------------------------------- FR-LM-14


def _toy_swipes(n, viewer_gender, candidate_gender, w, rng):
    x = with_intercept(rng.normal(size=(n, 2)))
    y = rng.random(n) < sigmoid(x @ w)
    return [Swipe(viewer_gender, candidate_gender, x[k], bool(y[k])) for k in range(n)]


def test_starting_taste_has_one_prior_per_group_and_shrinks_small_groups():
    """FR-LM-14: each orientation group gets its own w_0; small groups lean on the pooled one."""
    rng = np.random.default_rng(1)
    big = _toy_swipes(4000, 1, 0, np.array([2.0, 0.0, 0.0]), rng)
    small = _toy_swipes(20, 0, 0, np.array([-2.0, 0.0, 0.0]), rng)
    start = StartingTaste.fit(big + small, l2=10.0)
    assert set(start.by_group) == {(1, 0), (0, 0)}
    assert start.by_group[(1, 0)][0] > 1.5
    # the small group's own data points the other way, but 20 swipes cannot pull it far
    assert start.by_group[(0, 0)][0] > start.pooled[0] - 1.5
    x = with_intercept(np.array([[1.0, 0.0]]))
    unseen = start.offsets(2, x, np.array([0]))
    np.testing.assert_allclose(unseen, x @ start.pooled)


def test_starting_taste_beats_a_coin_flip(report):
    """FR-LM-14: for a brand-new member, w_0 alone predicts likes better than chance."""
    assert report.mean("auc", "start") > 0.6


def test_looks_scores_refuse_to_rank_lists_before_calibration(cfg, small_pop):
    """FR-LM-14 / R2: w_0 alone is never used to rank a daily list."""
    rng = np.random.default_rng(2)
    start = StartingTaste.fit(_toy_swipes(500, 1, 0, np.array([1.0, 0.0, 0.0]), rng), l2=1.0)
    taste = PersonalTaste(1, start, 1.0, min_swipes_for_lists=3)
    x = with_intercept(np.array([[0.5, 0.5]]))
    with pytest.raises(CalibrationIncomplete):
        taste.list_scores(x, np.array([0]))
    for liked in (True, False):
        taste.add_swipe(x[0], 0, liked)
    with pytest.raises(CalibrationIncomplete):
        taste.list_scores(x, np.array([0]))
    taste.add_swipe(x[0], 0, True)
    assert taste.list_scores(x, np.array([0])).shape == (1,)


# ---------------------------------------------------------------- FR-LM-15


def test_personal_taste_beats_the_starting_taste(report):
    """FR-LM-15: after 50 swipes the personal model predicts held-out likes better than w_0."""
    diff, se = report.paired("auc", "calibrated", "start")
    assert diff > 2 * se and diff > 0.03
    assert report.mean("corr", "calibrated") > report.mean("corr", "start")


def test_more_swipes_mean_less_uncertainty(cfg, small_pop):
    """FR-LM-15: each swipe shrinks the model's uncertainty about a profile."""
    _, feats = photo_features(ArrayPhotoEncoder(small_pop.photo_embeddings), len(small_pop), cfg)
    rng = np.random.default_rng(3)
    start = StartingTaste.fit(
        [Swipe(1, 0, feats[j], bool(rng.random() < 0.4)) for j in range(200)], l2=10.0
    )
    taste = PersonalTaste(1, start, 1.0, 50)
    probe = feats[300:301]
    before = taste.uncertainty(probe)[0]
    for j in range(200, 230):
        taste.add_swipe(feats[j], 0, bool(rng.random() < 0.4))
    assert taste.uncertainty(probe)[0] < before


# ---------------------------------------------------------------- FR-LM-16


def test_varied_selection_picks_different_directions_first():
    """FR-LM-16: the greedy DPP covers new directions before repeating a look."""
    near = np.array([[1.0, 0.0, 0.0], [0.99, 0.1, 0.0], [0.98, 0.0, 0.1]])
    others = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    x = np.vstack([near, others])
    picks = select_varied(x, 3)
    assert len(set(picks)) == 3
    assert {3, 4} <= set(picks)  # both new directions come before a second near-duplicate


def test_calibration_shows_the_planned_profiles(cfg, small_pop):
    """FR-LM-16: 50 distinct profiles from the pool, the first 20 chosen for variety."""
    arr = MemberArrays.from_members(small_pop.members, cfg.values)
    _, feats = photo_features(ArrayPhotoEncoder(small_pop.photo_embeddings), len(small_pop), cfg)
    viewer = next(m.id for m in small_pop.members if m.gender == "woman" and m.seeking == {"man"})
    pool = calibration_pool(arr, cfg, viewer)
    assert viewer not in pool and pool.size > cfg.calibration.n_profiles
    assert all(small_pop.members[j].gender == "man" for j in pool)
    rng = np.random.default_rng(4)
    start = StartingTaste.fit(
        [Swipe(1, 0, feats[j], bool(rng.random() < 0.4)) for j in pool[:100]], l2=10.0
    )
    taste = PersonalTaste(1, start, cfg.calibration.l2_penalty, cfg.calibration.n_profiles)
    shown = calibrate(taste, pool, feats, arr.gender, lambda j: j % 3 == 0, cfg)
    assert len(shown) == len(set(shown)) == cfg.calibration.n_profiles
    assert set(shown) <= set(pool.tolist())
    k = cfg.calibration.n_variety
    expected = [int(pool[r]) for r in select_varied(feats[pool][:, :-1], k)]
    assert shown[:k] == expected
    assert taste.n_swipes == cfg.calibration.n_profiles


def test_calibration_beats_random_profiles(report):
    """FR-LM-16 (acceptance): 50 calibration profiles teach more than 50 random ones."""
    diff, se = report.paired("auc", "calibrated", "random")
    assert diff > 2 * se


def test_variety_profiles_cover_more_look_types(report):
    """FR-LM-16: the first 20 profiles show more kinds of looks than 20 random ones."""
    assert report.coverage("calibrated") > report.coverage("random")


# ---------------------------------------------------------------- FR-LM-17


def test_report_has_learning_curve_and_group_audit(cfg, report):
    """FR-LM-17: the report shows accuracy by swipes and the per-group breakdown."""
    text = report.text()
    for g in range(cfg.population.n_audit_groups):
        assert f"G{g + 1}" in text
    assert "AUC by number of swipes" in text and "flagged" in text
    curve = report.curve("calibrated")
    assert curve[50] > curve[10]
    assert report.mean("auc", "truth") >= report.mean("auc", "calibrated")


def test_audit_flags_a_biased_model():
    """FR-LM-17: a model that shows one group far less than true taste would is flagged."""
    truth = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    biased = np.array([0.3, 0.25, 0.2, 0.2, 0.05])
    assert flag_groups(biased, truth, 0.05) == [0, 4]
    assert flag_groups(truth, truth, 0.05) == []


def test_default_model_is_not_flagged(report):
    """FR-LM-17: on the default population the looks model does not skew any group."""
    assert report.flagged() == []


def test_auc_helper():
    """FR-LM-17: AUC is 1 for a perfect ranking, 0.5 for ties, nan with one class."""
    assert auc(np.array([0.1, 0.9]), np.array([False, True])) == 1.0
    assert auc(np.array([0.5, 0.5]), np.array([False, True])) == 0.5
    assert np.isnan(auc(np.array([0.1, 0.2]), np.array([True, True])))


def test_report_is_deterministic(cfg):
    """FR-LM-17 / R7: same seed, same report."""
    pop = generate(cfg, n=300)
    a = run_looks_report(pop, cfg, max_viewers=10)
    b = run_looks_report(pop, cfg, max_viewers=10)
    assert a.text() == b.text()


def test_pca_dim_comes_from_config(cfg, small_pop):
    """FR-LM-13 / R5: the number of PCA dimensions follows the config."""
    small = dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, pca_dim=8))
    _, x = photo_features(ArrayPhotoEncoder(small_pop.photo_embeddings), len(small_pop), small)
    assert x.shape[1] == 9

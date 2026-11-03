"""FR-LM-03: the synthetic population is deterministic, valid and realistic enough to test on."""

import dataclasses

import numpy as np

from lastfirstdate.models import N_SLOTS, Member
from lastfirstdate.synthetic import GroundTruth, generate, summary


def _as_dicts(pop):
    return [dataclasses.asdict(m) for m in pop.members]


def test_same_seed_gives_same_population(cfg):
    """FR-LM-03 / rule R7: the same seed always gives the same members and embeddings."""
    a = generate(cfg, seed=3, n=150)
    b = generate(cfg, seed=3, n=150)
    assert _as_dicts(a) == _as_dicts(b)
    np.testing.assert_array_equal(a.photo_embeddings, b.photo_embeddings)
    np.testing.assert_array_equal(a.truth.taste, b.truth.taste)


def test_different_seed_gives_different_population(cfg):
    """FR-LM-03: changing the seed changes the population."""
    a = generate(cfg, seed=3, n=150)
    b = generate(cfg, seed=4, n=150)
    assert _as_dicts(a) != _as_dicts(b)


def test_population_matches_config_shares(cfg):
    """FR-LM-03: gender, paying and verified shares follow the config."""
    pop = generate(cfg, seed=5, n=2000)
    m = pop.members
    assert len(pop) == 2000
    share_men = np.mean([x.gender == "man" for x in m])
    assert abs(share_men - cfg.population.share_men) < 0.04
    men = [x for x in m if x.gender == "man"]
    women = [x for x in m if x.gender == "woman"]
    assert abs(np.mean([x.paying for x in men]) - cfg.population.paying_share["man"]) < 0.05
    assert abs(np.mean([x.paying for x in women]) - cfg.population.paying_share["woman"]) < 0.05
    assert abs(np.mean([x.verified for x in m]) - cfg.population.verified_share) < 0.03


def test_every_member_is_valid(cfg, small_pop):
    """FR-LM-03: every field is inside its allowed range."""
    pop_cfg = cfg.population
    keys = {q.key: q.options for q in cfg.values}
    for x in small_pop.members:
        assert x.gender in ("man", "woman")
        assert x.seeking and x.seeking <= {"man", "woman"}
        assert pop_cfg.age_min <= x.age <= pop_cfg.age_max
        assert x.age_min_pref <= x.age <= x.age_max_pref
        assert pop_cfg.max_km_min <= x.max_km <= pop_cfg.max_km_max
        assert 40.4 < x.lat < 41.0 and -74.4 < x.lon < -73.6
        assert pop_cfg.slots_min <= len(x.free_slots) <= pop_cfg.slots_max
        assert all(0 <= s < N_SLOTS for s in x.free_slots)
        assert pop_cfg.interests_min <= len(x.interests) <= pop_cfg.interests_max
        assert len(set(x.interests)) == len(x.interests)
        assert x.bio
        assert set(x.values) == set(keys) == set(x.accepts)
        for k, answer in x.values.items():
            assert answer in keys[k]
            assert answer in x.accepts[k]
            assert x.accepts[k] <= set(keys[k])
        assert x.dealbreakers <= set(keys)
        assert len(x.big5) == 5 and all(0.0 <= v <= 1.0 for v in x.big5)
        assert x.audit_group in {f"G{g + 1}" for g in range(pop_cfg.n_audit_groups)}


def test_embeddings_are_unit_vectors_clustered_by_look_type(cfg, small_pop):
    """FR-LM-03: fake photo embeddings behave like a real image model's: similar looks sit close."""
    e = small_pop.photo_embeddings
    assert e.shape == (len(small_pop), cfg.population.embedding_dim)
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-9)
    t = small_pop.truth.look_type
    sims = e @ e.T
    same = t[:, None] == t[None, :]
    off_diag = ~np.eye(len(t), dtype=bool)
    assert sims[same & off_diag].mean() > sims[~same].mean() + 0.2


def test_groups_lean_toward_look_types(cfg, small_pop):
    """FR-LM-03: fake groups correlate with looks, so a photo-taste bias is detectable by audits."""
    pop = generate(cfg, seed=8, n=1500)
    t = pop.truth.look_type
    k = cfg.population.n_look_types
    groups = np.array([int(x.audit_group[1:]) - 1 for x in pop.members])
    for g in range(cfg.population.n_audit_groups):
        favored = {(2 * g) % k, (2 * g + 1) % k}
        in_group = groups == g
        share = np.isin(t[in_group], list(favored)).mean()
        assert share > 2 * len(favored) / k  # well above an even spread


def test_ground_truth_is_kept_out_of_member_profiles(small_pop):
    """FR-LM-03 / rule R2: hidden truth never appears on the profiles the engine reads."""
    member_fields = {f.name for f in dataclasses.fields(Member)}
    truth_fields = {f.name for f in dataclasses.fields(GroundTruth)}
    assert not member_fields & truth_fields
    n = len(small_pop)
    truth = small_pop.truth
    assert truth.taste.shape == small_pop.photo_embeddings.shape
    np.testing.assert_allclose(np.linalg.norm(truth.taste, axis=1), 1.0, atol=1e-9)
    assert truth.looks_weight.shape == truth.pickiness.shape == (n,)
    assert np.all((truth.looks_weight > 0) & (truth.looks_weight < 1))


def test_summary_mentions_size(small_pop, cfg):
    """FR-LM-03: the summary describes the population in plain text."""
    text = summary(small_pop, cfg)
    assert "400 members" in text
    assert "paying" in text

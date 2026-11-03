"""FR-LM-01: every weight and cap comes from config/lastfirstdate.toml and is validated."""

import copy
import tomllib

import pytest

from lastfirstdate.config import DEFAULT_PATH, ConfigError, from_dict, load


@pytest.fixture()
def raw():
    with open(DEFAULT_PATH, "rb") as fh:
        return tomllib.load(fh)


def test_default_config_loads(cfg):
    """FR-LM-01: the shipped config loads and holds the agreed defaults."""
    assert cfg.mix.looks_weight == 0.5
    assert cfg.lists.daily_size == 30
    assert cfg.dates.guaranteed_per_month_paying == 3
    assert cfg.calibration.n_profiles == 50
    assert cfg.calibration.n_variety == 20
    p = cfg.personality
    assert (p.w_text, p.w_values, p.w_big5) == (0.4, 0.3, 0.3)
    assert len(cfg.values) >= 1


def test_load_accepts_explicit_path():
    """FR-LM-01: load() reads the file it is given."""
    assert load(DEFAULT_PATH).lists.daily_size == 30


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("personality", "w_text", 0.5, "add up to 1"),
        ("mix", "looks_weight", 1.5, "between 0 and 1"),
        ("calibration", "n_variety", 60, "n_variety"),
        ("calibration", "pca_dim", 4096, "pca_dim"),
        ("dates", "guaranteed_per_month_paying", 9, "guaranteed_per_month_paying"),
        ("lists", "exploration_share", -0.1, "between 0 and 1"),
        ("population", "slots_max", 30, "21"),
    ],
)
def test_out_of_range_values_are_rejected(raw, section, key, value, message):
    """FR-LM-01: a bad value fails loudly instead of running with it."""
    bad = copy.deepcopy(raw)
    bad[section][key] = value
    with pytest.raises(ConfigError, match=message):
        from_dict(bad)


def test_missing_section_is_rejected(raw):
    """FR-LM-01: every section must be present."""
    bad = copy.deepcopy(raw)
    del bad["dates"]
    with pytest.raises(ConfigError, match=r"\[dates\]"):
        from_dict(bad)


def test_missing_and_unknown_keys_are_rejected(raw):
    """FR-LM-01: typos in key names are caught."""
    bad = copy.deepcopy(raw)
    del bad["lists"]["daily_size"]
    with pytest.raises(ConfigError, match="daily_size"):
        from_dict(bad)
    bad = copy.deepcopy(raw)
    bad["lists"]["daily_sise"] = 30
    with pytest.raises(ConfigError, match="unknown"):
        from_dict(bad)


def test_values_questions_are_validated(raw):
    """FR-LM-01: values questions need unique keys and at least two options."""
    bad = copy.deepcopy(raw)
    bad["values"].append(dict(bad["values"][0]))
    with pytest.raises(ConfigError, match="unique"):
        from_dict(bad)
    bad = copy.deepcopy(raw)
    bad["values"][0]["options"] = ["only"]
    with pytest.raises(ConfigError, match="2\\+"):
        from_dict(bad)

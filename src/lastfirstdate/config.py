"""Load and validate config/lastfirstdate.toml (FR-LM-01).

Every weight, cap and threshold the engine uses comes from here. Nothing in the
engine hard-codes a tunable number.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "lastfirstdate.toml"


class ConfigError(ValueError):
    """Raised when the config file is missing a value or a value is out of range."""


@dataclass(frozen=True)
class FilterConfig:
    require_verified: bool
    active_within_days: int
    min_shared_slots: int


@dataclass(frozen=True)
class PersonalityConfig:
    w_text: float
    w_values: float
    w_big5: float


@dataclass(frozen=True)
class MixConfig:
    looks_weight: float


@dataclass(frozen=True)
class CalibrationConfig:
    n_profiles: int
    n_variety: int
    pca_dim: int
    l2_penalty: float
    w0_l2_penalty: float
    group_share_tolerance: float


@dataclass(frozen=True)
class ListsConfig:
    daily_size: int
    reshow_after_days: int
    max_appearances_per_day: int
    exploration_share: float


@dataclass(frozen=True)
class DatesConfig:
    guaranteed_per_month_paying: int
    cap_per_month: int
    weeks_per_month: int
    max_per_week: int
    date_value: float
    late_month_bonus: float


@dataclass(frozen=True)
class SimulationConfig:
    seed: int
    guarantee_target: float
    min_pool_for_guarantee: int
    open_app_rate: float
    picks_per_day: int
    personality_signal: float
    like_threshold: float
    accept_threshold: float
    again_threshold: float
    choice_temperature: float
    show_up_rate: float
    meeting_noise: float
    repeat_pick_factor: float
    swipe_threshold: float
    training_swipes: int


@dataclass(frozen=True)
class ValuesQuestion:
    key: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class PopulationConfig:
    n_members: int
    share_men: float
    orientation: dict[str, float]
    paying_share: dict[str, float]
    verified_share: float
    active_share: float
    age_min: int
    age_max: int
    max_km_min: float
    max_km_max: float
    embedding_dim: int
    n_look_types: int
    n_audit_groups: int
    interests_min: int
    interests_max: int
    slots_min: int
    slots_max: int
    dealbreaker_rate: float


@dataclass(frozen=True)
class Config:
    filter: FilterConfig
    personality: PersonalityConfig
    mix: MixConfig
    calibration: CalibrationConfig
    lists: ListsConfig
    dates: DatesConfig
    simulation: SimulationConfig
    population: PopulationConfig
    values: tuple[ValuesQuestion, ...]


def _section(raw: dict, name: str, cls: type):
    if name not in raw:
        raise ConfigError(f"missing section [{name}]")
    fields = cls.__dataclass_fields__
    data = raw[name]
    missing = [f for f in fields if f not in data]
    if missing:
        raise ConfigError(f"[{name}] is missing: {', '.join(missing)}")
    unknown = [k for k in data if k not in fields]
    if unknown:
        raise ConfigError(f"[{name}] has unknown keys: {', '.join(unknown)}")
    return cls(**{k: data[k] for k in fields})


def _check_share(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{name} must be between 0 and 1, got {value}")


def _check_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")


def validate(cfg: Config) -> None:
    """Raise ConfigError if any value is out of range or inconsistent."""
    p = cfg.personality
    for name, w in (("w_text", p.w_text), ("w_values", p.w_values), ("w_big5", p.w_big5)):
        _check_share(f"personality.{name}", w)
    if not math.isclose(p.w_text + p.w_values + p.w_big5, 1.0, abs_tol=1e-9):
        raise ConfigError("personality weights must add up to 1")

    _check_share("mix.looks_weight", cfg.mix.looks_weight)

    c = cfg.calibration
    _check_positive("calibration.n_profiles", c.n_profiles)
    if not 0 <= c.n_variety <= c.n_profiles:
        raise ConfigError("calibration.n_variety must be between 0 and n_profiles")
    _check_positive("calibration.pca_dim", c.pca_dim)
    _check_positive("calibration.l2_penalty", c.l2_penalty)
    _check_positive("calibration.w0_l2_penalty", c.w0_l2_penalty)
    _check_share("calibration.group_share_tolerance", c.group_share_tolerance)

    lists = cfg.lists
    _check_positive("lists.daily_size", lists.daily_size)
    _check_positive("lists.reshow_after_days", lists.reshow_after_days)
    if lists.max_appearances_per_day < 0:
        raise ConfigError("lists.max_appearances_per_day must be 0 (auto) or positive")
    _check_share("lists.exploration_share", lists.exploration_share)

    d = cfg.dates
    _check_positive("dates.cap_per_month", d.cap_per_month)
    _check_positive("dates.weeks_per_month", d.weeks_per_month)
    if not 0 <= d.guaranteed_per_month_paying <= d.cap_per_month:
        raise ConfigError("dates.guaranteed_per_month_paying must be between 0 and cap_per_month")
    if d.late_month_bonus < 0:
        raise ConfigError("dates.late_month_bonus must not be negative")
    if d.date_value < 0:
        raise ConfigError("dates.date_value must not be negative")
    if d.max_per_week != 1:
        raise ConfigError("dates.max_per_week: only 1 date per person per week is supported")

    sim = cfg.simulation
    for name in (
        "guarantee_target",
        "open_app_rate",
        "personality_signal",
        "show_up_rate",
        "repeat_pick_factor",
    ):
        _check_share(f"simulation.{name}", getattr(sim, name))
    _check_positive("simulation.picks_per_day", sim.picks_per_day)
    _check_positive("simulation.training_swipes", sim.training_swipes)
    _check_positive("simulation.choice_temperature", sim.choice_temperature)
    if sim.meeting_noise < 0:
        raise ConfigError("simulation.meeting_noise must not be negative")
    if sim.min_pool_for_guarantee < 0:
        raise ConfigError("simulation.min_pool_for_guarantee must not be negative")

    f = cfg.filter
    if f.active_within_days < 0 or f.min_shared_slots < 0:
        raise ConfigError("filter values must not be negative")

    pop = cfg.population
    _check_positive("population.n_members", pop.n_members)
    for name in ("share_men", "verified_share", "active_share", "dealbreaker_rate"):
        _check_share(f"population.{name}", getattr(pop, name))
    if set(pop.orientation) != {"straight", "gay", "bi"}:
        raise ConfigError("population.orientation needs exactly: straight, gay, bi")
    if not math.isclose(sum(pop.orientation.values()), 1.0, abs_tol=1e-9):
        raise ConfigError("population.orientation shares must add up to 1")
    if set(pop.paying_share) != {"man", "woman"}:
        raise ConfigError("population.paying_share needs exactly: man, woman")
    for k, v in pop.paying_share.items():
        _check_share(f"population.paying_share.{k}", v)
    if not 18 <= pop.age_min < pop.age_max:
        raise ConfigError("population ages must satisfy 18 <= age_min < age_max")
    if not 0 < pop.max_km_min <= pop.max_km_max:
        raise ConfigError("population max_km range is invalid")
    for lo, hi in (("interests_min", "interests_max"), ("slots_min", "slots_max")):
        if not 0 < getattr(pop, lo) <= getattr(pop, hi):
            raise ConfigError(f"population.{lo} must be positive and <= {hi}")
    if pop.slots_max > 21:
        raise ConfigError("population.slots_max cannot exceed 21 weekly slots")
    for name in ("embedding_dim", "n_look_types", "n_audit_groups"):
        _check_positive(f"population.{name}", getattr(pop, name))
    if c.pca_dim > pop.embedding_dim:
        raise ConfigError("calibration.pca_dim cannot exceed population.embedding_dim")

    if not cfg.values:
        raise ConfigError("at least one [[values]] question is required")
    keys = [q.key for q in cfg.values]
    if len(keys) != len(set(keys)):
        raise ConfigError("[[values]] question keys must be unique")
    for q in cfg.values:
        if len(q.options) < 2 or len(set(q.options)) != len(q.options):
            raise ConfigError(f"values question {q.key!r} needs 2+ unique options")


def from_dict(raw: dict) -> Config:
    """Build and validate a Config from a parsed TOML dict."""
    values_raw = raw.get("values", [])
    try:
        values = tuple(
            ValuesQuestion(key=q["key"], options=tuple(q["options"])) for q in values_raw
        )
    except KeyError as exc:
        raise ConfigError(f"[[values]] entry is missing {exc}") from exc
    cfg = Config(
        filter=_section(raw, "filter", FilterConfig),
        personality=_section(raw, "personality", PersonalityConfig),
        mix=_section(raw, "mix", MixConfig),
        calibration=_section(raw, "calibration", CalibrationConfig),
        lists=_section(raw, "lists", ListsConfig),
        dates=_section(raw, "dates", DatesConfig),
        simulation=_section(raw, "simulation", SimulationConfig),
        population=_section(raw, "population", PopulationConfig),
        values=values,
    )
    validate(cfg)
    return cfg


def load(path: str | Path = DEFAULT_PATH) -> Config:
    """Read the TOML file at `path` and return a validated Config."""
    with open(path, "rb") as fh:
        return from_dict(tomllib.load(fh))

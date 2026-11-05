"""Load and validate config/lastfirstdate.toml (FR-LM-01).

Every weight, cap and threshold the engine uses comes from here. Nothing in the
engine hard-codes a tunable number.
"""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
    text_encoder: str


@dataclass(frozen=True)
class MixConfig:
    looks_weight: float
    chooser_weight: float


@dataclass(frozen=True)
class CalibrationConfig:
    n_profiles: int
    n_variety: int
    n_pairs: int
    pca_dim: int
    l2_penalty: float
    w0_l2_penalty: float
    group_share_tolerance: float


@dataclass(frozen=True)
class ListsConfig:
    daily_size: int
    reshow_after_days: int
    policy: str
    exposure_multiple: float
    guarantee_priority: float
    hide_booked: bool
    nsw_epsilon: float
    holdback_yeses: int
    exploration_share: float


@dataclass(frozen=True)
class LearningConfig:
    months: int
    taste_date_weight: float
    lambda_date_weight: float
    lambda_choice_weight: float
    lambda_prior_slope: float
    lambda_prior_strength: float
    lambda_min: float
    lambda_max: float


@dataclass(frozen=True)
class GateConfig:
    target: float
    min_women_per_man: float
    min_pool: int
    at_risk_threshold: float
    forecast_l2: float
    forecast: str


@dataclass(frozen=True)
class FairnessConfig:
    min_rate_ratio: float


@dataclass(frozen=True)
class DatesConfig:
    guaranteed_per_month_paying: int
    cap_per_month: int
    cap_per_month_free: int
    free_cap_genders: list[str]
    weeks_per_month: int
    max_per_week: int
    date_value: float
    late_month_bonus: float


@dataclass(frozen=True)
class TestRunsConfig:
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
    rating_cutoffs: list[float]
    repeat_pick_factor: float
    swipe_threshold: float
    training_swipes: int


@dataclass(frozen=True)
class BookingConfig:
    lead_days: int
    horizon_days: int
    hold_minutes: int
    timezone: str  # the zone's clock: date windows are local times (D19)
    part_starts: list[int]  # hour each part of the day begins: lunch, evening, late


@dataclass(frozen=True)
class BillingConfig:
    price_per_cycle: float
    trial_days: int
    cycle_days: int
    date_fee: float
    full_refund_hours: int
    late_refund_share: float
    own_absence_counts: bool


@dataclass(frozen=True)
class TrustConfig:
    prior_mean: float
    prior_strength: float
    reliability_weight: float
    flag_below: float
    flag_min_ratings: int
    ranking_weight: float


@dataclass(frozen=True)
class RefitConfig:
    min_man_cycles: int
    holdout_share: float
    min_improvement: float


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class GeoConfig:
    neighbourhoods: tuple[Place, ...]
    venues: tuple[Place, ...]


@dataclass(frozen=True)
class ModelsConfig:
    """The production photo and text models, pinned (decision D20, FR-LM-59)."""

    cache_dir: str  # where downloaded model files live ("~" allowed)
    photo_model: str  # the model's name, recorded with each photo embedding
    photo_repo: str  # where the ONNX export is downloaded from
    photo_revision: str  # exact revision (commit) of that repository
    photo_file: str
    photo_sha256: str
    photo_size: int  # photos are resized to this many pixels square
    text_model: str
    text_revision: str
    text_file: str
    text_sha256: str
    tokenizer_sha256: str
    text_max_tokens: int  # longer texts are cut to this many word pieces


@dataclass(frozen=True)
class CycleRunsConfig:
    cycles: int
    stagger: bool
    pay_in_time_chooser: float
    pay_in_time_chosen: float
    cancel_rate: float
    late_cancel_share: float
    event_rate: float
    new_women_per_cycle: float
    bad_actor_share: float
    bad_actor_conduct: float
    rating_center: float
    conduct_effect: float
    rating_noise: float
    rating_halo: float


@dataclass(frozen=True)
class ValuesQuestion:
    """One values or intent question.

    mode "accepts": each member gives an answer and the answers they accept (Sprints 1-5).
    mode "closeness": each member gives one answer; options are listed in order and two
    answers are as compatible as they are close. `incompatible` lists answer pairs that
    rule a pair out for everyone (a shared dealbreaker), in either order.
    """

    key: str
    options: tuple[str, ...]
    mode: str = "accepts"
    incompatible: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Area:
    name: str
    lat: float
    lon: float
    share: float


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
    area_jitter_deg: float
    prompt_style: str
    exact_counts: bool
    areas: tuple[Area, ...]


@dataclass(frozen=True)
class Config:
    filter: FilterConfig
    personality: PersonalityConfig
    mix: MixConfig
    calibration: CalibrationConfig
    lists: ListsConfig
    fairness: FairnessConfig
    learning: LearningConfig
    gate: GateConfig
    dates: DatesConfig
    test_runs: TestRunsConfig
    population: PopulationConfig
    values: tuple[ValuesQuestion, ...]
    booking: BookingConfig
    billing: BillingConfig
    trust: TrustConfig
    geo: GeoConfig
    cycle_runs: CycleRunsConfig
    refit: RefitConfig
    models: ModelsConfig


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
    if p.text_encoder not in ("tags", "words", "sentence"):
        raise ConfigError("personality.text_encoder must be tags, words or sentence")

    _check_share("mix.looks_weight", cfg.mix.looks_weight)
    if not 0.0 < cfg.mix.chooser_weight < 1.0:
        raise ConfigError("mix.chooser_weight must be strictly between 0 and 1")

    c = cfg.calibration
    _check_positive("calibration.n_profiles", c.n_profiles)
    if not 0 <= c.n_variety <= c.n_profiles:
        raise ConfigError("calibration.n_variety must be between 0 and n_profiles")
    if c.n_pairs < 0 or c.n_variety + 2 * c.n_pairs > c.n_profiles:
        raise ConfigError("calibration: n_variety + 2 * n_pairs must not exceed n_profiles")
    _check_positive("calibration.pca_dim", c.pca_dim)
    _check_positive("calibration.l2_penalty", c.l2_penalty)
    _check_positive("calibration.w0_l2_penalty", c.w0_l2_penalty)
    _check_share("calibration.group_share_tolerance", c.group_share_tolerance)

    lists = cfg.lists
    _check_positive("lists.daily_size", lists.daily_size)
    _check_positive("lists.reshow_after_days", lists.reshow_after_days)
    if lists.policy not in ("basic", "fair", "nsw"):
        raise ConfigError("lists.policy must be basic, fair or nsw")
    _check_positive("lists.exposure_multiple", lists.exposure_multiple)
    _check_positive("lists.nsw_epsilon", lists.nsw_epsilon)
    if lists.guarantee_priority < 0 or lists.holdback_yeses < 0:
        raise ConfigError("lists.guarantee_priority and lists.holdback_yeses must not be negative")
    _check_share("fairness.min_rate_ratio", cfg.fairness.min_rate_ratio)
    gate = cfg.gate
    _check_share("gate.target", gate.target)
    _check_share("gate.at_risk_threshold", gate.at_risk_threshold)
    _check_positive("gate.min_women_per_man", gate.min_women_per_man)
    _check_positive("gate.forecast_l2", gate.forecast_l2)
    if gate.min_pool < 0:
        raise ConfigError("gate.min_pool must not be negative")
    if gate.forecast not in ("booked", "delivered"):
        raise ConfigError("gate.forecast must be booked or delivered")
    learn = cfg.learning
    _check_positive("learning.months", learn.months)
    for name in (
        "taste_date_weight",
        "lambda_date_weight",
        "lambda_choice_weight",
        "lambda_prior_strength",
    ):
        if getattr(learn, name) < 0:
            raise ConfigError(f"learning.{name} must not be negative")
    if not 0.0 <= learn.lambda_min <= 0.5 <= learn.lambda_max <= 1.0:
        raise ConfigError("learning: need 0 <= lambda_min <= 0.5 <= lambda_max <= 1")
    cuts = cfg.test_runs.rating_cutoffs
    if len(cuts) != 4 or sorted(cuts) != list(cuts):
        raise ConfigError("test_runs.rating_cutoffs needs 4 increasing numbers")
    _check_share("lists.exploration_share", lists.exploration_share)

    d = cfg.dates
    _check_positive("dates.cap_per_month", d.cap_per_month)
    if not 0 <= d.cap_per_month_free <= d.cap_per_month:
        raise ConfigError("dates.cap_per_month_free must be between 0 and cap_per_month")
    if not set(d.free_cap_genders) <= {"man", "woman", "nonbinary"}:
        raise ConfigError("dates.free_cap_genders may only list man, woman, nonbinary")
    _check_positive("dates.weeks_per_month", d.weeks_per_month)
    if not 0 <= d.guaranteed_per_month_paying <= d.cap_per_month:
        raise ConfigError("dates.guaranteed_per_month_paying must be between 0 and cap_per_month")
    if d.late_month_bonus < 0:
        raise ConfigError("dates.late_month_bonus must not be negative")
    if d.date_value < 0:
        raise ConfigError("dates.date_value must not be negative")
    if d.max_per_week != 1:
        raise ConfigError("dates.max_per_week: only 1 date per person per week is supported")

    sim = cfg.test_runs
    for name in (
        "guarantee_target",
        "open_app_rate",
        "personality_signal",
        "show_up_rate",
        "repeat_pick_factor",
    ):
        _check_share(f"test_runs.{name}", getattr(sim, name))
    _check_positive("test_runs.picks_per_day", sim.picks_per_day)
    _check_positive("test_runs.training_swipes", sim.training_swipes)
    _check_positive("test_runs.choice_temperature", sim.choice_temperature)
    if sim.meeting_noise < 0:
        raise ConfigError("test_runs.meeting_noise must not be negative")
    if sim.min_pool_for_guarantee < 0:
        raise ConfigError("test_runs.min_pool_for_guarantee must not be negative")

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
    if not pop.areas:
        raise ConfigError("population needs at least one [[population.areas]] entry")
    if any(a.share <= 0 for a in pop.areas):
        raise ConfigError("population.areas shares must be positive")
    if pop.prompt_style not in ("bio", "prompts"):
        raise ConfigError("population.prompt_style must be bio or prompts")
    if pop.area_jitter_deg < 0:
        raise ConfigError("population.area_jitter_deg must not be negative")
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
        if q.mode not in ("accepts", "closeness"):
            raise ConfigError(f"values question {q.key!r}: mode must be accepts or closeness")
        for pair in q.incompatible:
            if len(pair) != 2 or not set(pair) <= set(q.options):
                raise ConfigError(f"values question {q.key!r}: bad incompatible pair {pair!r}")
        if q.incompatible and q.mode != "closeness":
            raise ConfigError(f"values question {q.key!r}: incompatible needs mode closeness")

    bk = cfg.booking
    if not 0 <= bk.lead_days < bk.horizon_days <= 21:
        raise ConfigError("booking needs 0 <= lead_days < horizon_days <= 21")
    _check_positive("booking.hold_minutes", bk.hold_minutes)
    try:
        ZoneInfo(bk.timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ConfigError(f"booking.timezone {bk.timezone!r} is not a known time zone") from exc
    starts = list(bk.part_starts)
    if len(starts) != 3 or starts != sorted(set(starts)) or not 0 <= starts[0] <= starts[-1] <= 23:
        raise ConfigError("booking.part_starts needs 3 increasing hours between 0 and 23")
    bl = cfg.billing
    for name in ("price_per_cycle", "cycle_days", "date_fee", "full_refund_hours"):
        _check_positive(f"billing.{name}", getattr(bl, name))
    if bl.trial_days < 0:
        raise ConfigError("billing.trial_days must not be negative")
    _check_share("billing.late_refund_share", bl.late_refund_share)
    tr = cfg.trust
    if not 0 < tr.prior_mean <= 1:
        raise ConfigError("trust.prior_mean must be in (0, 1]")
    _check_positive("trust.prior_strength", tr.prior_strength)
    _check_share("trust.reliability_weight", tr.reliability_weight)
    _check_share("trust.ranking_weight", tr.ranking_weight)
    if not 0 <= tr.flag_below < tr.prior_mean:
        raise ConfigError("trust.flag_below must be at least 0 and below trust.prior_mean")
    if tr.flag_min_ratings < 1:
        raise ConfigError("trust.flag_min_ratings must be at least 1")
    for kind in ("neighbourhoods", "venues"):
        places = getattr(cfg.geo, kind)
        if not places:
            raise ConfigError(f"geo needs at least one [[geo.{kind}]] entry")
        if len({pl.name for pl in places}) != len(places):
            raise ConfigError(f"geo.{kind} names must be unique")
        if any(not (-90 <= pl.lat <= 90 and -180 <= pl.lon <= 180) for pl in places):
            raise ConfigError(f"geo.{kind} coordinates are out of range")
    rf = cfg.refit
    _check_positive("refit.min_man_cycles", rf.min_man_cycles)
    if not 0 < rf.holdout_share < 1:
        raise ConfigError("refit.holdout_share must be between 0 and 1")
    if rf.min_improvement < 0:
        raise ConfigError("refit.min_improvement must not be negative")
    md = cfg.models
    for name in ("photo_revision", "text_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", getattr(md, name)):
            raise ConfigError(f"models.{name} must be a full 40-character commit hash")
    for name in ("photo_sha256", "text_sha256", "tokenizer_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", getattr(md, name)):
            raise ConfigError(f"models.{name} must be a 64-character sha256")
    _check_positive("models.photo_size", md.photo_size)
    _check_positive("models.text_max_tokens", md.text_max_tokens)
    cs = cfg.cycle_runs
    _check_positive("cycle_runs.cycles", cs.cycles)
    if cs.new_women_per_cycle < 0:
        raise ConfigError("cycle_runs.new_women_per_cycle must not be negative")
    for name in (
        "pay_in_time_chooser",
        "pay_in_time_chosen",
        "cancel_rate",
        "late_cancel_share",
        "event_rate",
        "bad_actor_share",
    ):
        _check_share(f"cycle_runs.{name}", getattr(cs, name))
    if cs.rating_noise < 0 or cs.rating_halo < 0 or cs.conduct_effect < 0:
        raise ConfigError(
            "cycle_runs.rating_noise, rating_halo and conduct_effect must not be negative"
        )
    if not 1 <= cs.rating_center <= 5:
        raise ConfigError("cycle_runs.rating_center must be between 1 and 5")


def _population(raw: dict) -> PopulationConfig:
    if "population" not in raw:
        raise ConfigError("missing section [population]")
    data = dict(raw["population"])
    try:
        data["areas"] = tuple(Area(**a) for a in data.get("areas", []))
    except TypeError as exc:
        raise ConfigError(
            f"[[population.areas]] entries need name, lat, lon, share: {exc}"
        ) from exc
    return _section({"population": data}, "population", PopulationConfig)


def _geo(raw: dict) -> GeoConfig:
    if "geo" not in raw:
        raise ConfigError("missing section [geo]")
    data = raw["geo"]
    unknown = [k for k in data if k not in ("neighbourhoods", "venues")]
    if unknown:
        raise ConfigError(f"[geo] has unknown keys: {', '.join(unknown)}")
    try:
        return GeoConfig(
            neighbourhoods=tuple(Place(**pl) for pl in data.get("neighbourhoods", [])),
            venues=tuple(Place(**pl) for pl in data.get("venues", [])),
        )
    except TypeError as exc:
        raise ConfigError(f"[[geo.*]] entries need name, lat, lon: {exc}") from exc


def from_dict(raw: dict) -> Config:
    """Build and validate a Config from a parsed TOML dict."""
    values_raw = raw.get("values", [])
    try:
        values = tuple(
            ValuesQuestion(
                key=q["key"],
                options=tuple(q["options"]),
                mode=q.get("mode", "accepts"),
                incompatible=tuple(tuple(pair) for pair in q.get("incompatible", [])),
            )
            for q in values_raw
        )
    except KeyError as exc:
        raise ConfigError(f"[[values]] entry is missing {exc}") from exc
    cfg = Config(
        filter=_section(raw, "filter", FilterConfig),
        personality=_section(raw, "personality", PersonalityConfig),
        mix=_section(raw, "mix", MixConfig),
        calibration=_section(raw, "calibration", CalibrationConfig),
        lists=_section(raw, "lists", ListsConfig),
        fairness=_section(raw, "fairness", FairnessConfig),
        learning=_section(raw, "learning", LearningConfig),
        gate=_section(raw, "gate", GateConfig),
        dates=_section(raw, "dates", DatesConfig),
        test_runs=_section(raw, "test_runs", TestRunsConfig),
        population=_population(raw),
        values=values,
        booking=_section(raw, "booking", BookingConfig),
        billing=_section(raw, "billing", BillingConfig),
        trust=_section(raw, "trust", TrustConfig),
        geo=_geo(raw),
        cycle_runs=_section(raw, "cycle_runs", CycleRunsConfig),
        refit=_section(raw, "refit", RefitConfig),
        models=_section(raw, "models", ModelsConfig),
    )
    validate(cfg)
    return cfg


def load(path: str | Path = DEFAULT_PATH) -> Config:
    """Read the TOML file at `path` and return a validated Config."""
    with open(path, "rb") as fh:
        return from_dict(tomllib.load(fh))


# ---------------------------------------------------------------- what changing a setting needs

# FR-LM-48: every setting is classified by what changing it in production requires.
BUSINESS = "business terms: change any time (with legal review where money is involved)"
TEST_FIRST = "test first: rerun the reports and check fulfilment before changing"
REFIT = "refit the forecast: python -m lastfirstdate.forecast_fit --delivered, then rerun the tests"
TEST_ONLY = "test runs only: the app never reads it"
ENGINEERING = "engineering only: changes the data the app collects or how a model is fitted"
IMPACTS = (BUSINESS, TEST_FIRST, REFIT, TEST_ONLY, ENGINEERING)

_SECTION_IMPACT = {
    "filter": TEST_FIRST,
    "personality": TEST_FIRST,
    "mix": TEST_FIRST,
    "calibration": TEST_FIRST,
    "lists": TEST_FIRST,
    "fairness": BUSINESS,
    "learning": TEST_FIRST,
    "gate": BUSINESS,
    "dates": REFIT,
    "test_runs": TEST_ONLY,
    "population": TEST_ONLY,
    "values": ENGINEERING,
    "booking": REFIT,
    "billing": BUSINESS,
    "trust": TEST_FIRST,
    "geo": BUSINESS,
    "cycle_runs": TEST_ONLY,
    "refit": ENGINEERING,
    "models": ENGINEERING,
}
_KEY_IMPACT = {
    "personality.w_big5": ENGINEERING,  # the database holds no Big Five scores
    "personality.text_encoder": ENGINEERING,
    "calibration.pca_dim": ENGINEERING,
    "calibration.l2_penalty": ENGINEERING,
    "calibration.w0_l2_penalty": ENGINEERING,
    "calibration.group_share_tolerance": BUSINESS,  # an audit threshold
    "gate.forecast": ENGINEERING,
    "gate.forecast_l2": ENGINEERING,
    "dates.weeks_per_month": ENGINEERING,  # the batch scheduler and booked forecast only
    "dates.date_value": ENGINEERING,
    "dates.late_month_bonus": ENGINEERING,
    "billing.cycle_days": REFIT,  # the forecast is fitted on 28-day cycles
    "booking.timezone": ENGINEERING,  # the zone's clock
    "booking.part_starts": BUSINESS,  # the times members see
    "trust.flag_below": BUSINESS,  # safety policy
    "trust.flag_min_ratings": BUSINESS,
}


def setting_names() -> list[str]:
    """Every setting as "section.key" ("values" for the intent questions)."""
    names = []
    for section, f in Config.__dataclass_fields__.items():
        if section == "values":
            names.append("values")
            continue
        cls = globals()[f.type] if isinstance(f.type, str) else f.type
        names += [f"{section}.{key}" for key in cls.__dataclass_fields__]
    return names


def setting_impact(name: str) -> str:
    """What changing setting `name` ("section.key") requires (FR-LM-48)."""
    if name in _KEY_IMPACT:
        return _KEY_IMPACT[name]
    section = name.split(".", 1)[0]
    if section not in _SECTION_IMPACT:
        raise KeyError(f"unknown setting {name!r}")
    return _SECTION_IMPACT[section]


def settings_table() -> str:
    """The handoff's settings table, grouped by what a change requires (Markdown)."""
    lines = []
    for impact in IMPACTS:
        names = [n for n in setting_names() if setting_impact(n) == impact]
        lines.append(f"**{impact[0].upper()}{impact[1:]}**\n")
        lines.append(", ".join(f"`{n}`" for n in names) + "\n")
    return "\n".join(lines)

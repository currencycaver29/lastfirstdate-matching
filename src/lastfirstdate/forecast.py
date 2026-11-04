"""Guarantee forecast, at-risk alerts and the ratio gate (FR-LM-33, FR-LM-34).

The business page's rule is "if the ratio isn't there, male sales pause". This module turns
it into numbers.

Forecast. For each paying man j, the chance he reaches his guarantee this cycle:

    P_j = sigmoid(b0 + b1 * log(r) + b2 * log(1 + pool_j) + b3 * log(1 + share_j)
                  + b4 * log(min(r, 1)))

- r, women per man: active committed women / paying men in the zone
- pool_j: women who could be matched with him (Stage 1)
- share_j: his expected share of women's attention, the sum over the women in his pool of
  1 / (number of men each of them could be matched with)
- log(min(r, 1)): zero while there is at least one woman per man, negative below that. With
  fewer women than men their date caps run out and fulfilment falls off a cliff (in the
  Phase 1 simulation, 61% of men reach 3 dates at 1 woman per man but 11% at 0.5); this
  term lets the curve bend there instead of extrapolating the gentle slope above 1.

The at-risk model adds his progress mid-cycle: dates booked / guarantee,
log(1 + yeses waiting) and weeks left / weeks. Both are logistic regressions, fitted on
simulated cycles before launch (`forecast_fit.py` writes `config/forecast_model.toml`) and
refitted on real cycles once they exist, with the same `fit_forecast`. The model file records
the range of women per man it was fitted on; the gate never relies on it outside that range.

Gate. Applicants on the men's waitlist are considered in order. One is admitted only if, with
him added, women per man stays at or above `gate.min_women_per_man` and inside the forecast's
fitted range, and the forecast average fulfilment stays at or above `gate.target`. Applicants
whose pool is below `gate.min_pool` are deferred rather than sold a guarantee they would likely
miss. Pool size depends only on the member's own settings and location, never on how often
anyone is picked (rule R2).
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lastfirstdate import native
from lastfirstdate.config import Config, GateConfig
from lastfirstdate.looks import fit_logistic, sigmoid

START_FEATURES = (
    "intercept",
    "log women per man",
    "log possible matches",
    "log attention share",
    "log women per man below 1",
)
PROGRESS_FEATURES = START_FEATURES + (
    "dates so far / guarantee",
    "log waiting yeses",
    "weeks left / weeks",
)
# Sprint 8: the same start features, fitted on delivered dates in the real booking flow
DELIVERED_PROGRESS_FEATURES = START_FEATURES + (
    "delivered so far / guarantee",
    "log confirmed dates to come",
    "days left / cycle days",
)
KINDS = ("booked", "delivered")
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DEFAULT_MODEL_PATH = CONFIG_DIR / "forecast_model.toml"  # booked dates (Sprint 6)
DELIVERED_MODEL_PATH = CONFIG_DIR / "forecast_delivered.toml"  # delivered dates (Sprint 8)


def progress_features(kind: str) -> tuple[str, ...]:
    return DELIVERED_PROGRESS_FEATURES if kind == "delivered" else PROGRESS_FEATURES


@dataclass(frozen=True)
class MarketFeatures:
    men: np.ndarray  # ids of the paying men described
    women_per_man: float
    pool: np.ndarray  # (men,) possible matches
    share: np.ndarray  # (men,) expected share of women's attention


def market_features(
    eligible: np.ndarray, women: Sequence[int], men: Sequence[int]
) -> MarketFeatures:
    """Features of each man in `men` against the women in `women`."""
    women = np.asarray(women, dtype=int)
    men = np.asarray(men, dtype=int)
    e = eligible[np.ix_(women, men)].astype(float)  # women x men
    per_woman = e.sum(axis=1)  # men each woman could be matched with
    share = (e / np.maximum(per_woman, 1.0)[:, None]).sum(axis=0)
    return MarketFeatures(
        men=men,
        women_per_man=len(women) / max(len(men), 1),
        pool=e.sum(axis=0),
        share=share,
    )


def start_matrix(mf: MarketFeatures) -> np.ndarray:
    m = mf.men.size
    log_ratio = np.log(max(mf.women_per_man, 1e-9))
    return np.column_stack(
        [
            np.ones(m),
            np.full(m, log_ratio),
            np.log1p(mf.pool),
            np.log1p(mf.share),
            np.full(m, min(log_ratio, 0.0)),
        ]
    )


def progress_matrix(
    mf: MarketFeatures,
    dates_so_far: np.ndarray,
    waiting: np.ndarray,
    weeks_left: int,
    cfg: Config,
) -> np.ndarray:
    """Start features plus progress, for the men in `mf` (dates and waiting indexed by id)."""
    d = cfg.dates
    m = mf.men.size
    return np.column_stack(
        [
            start_matrix(mf),
            dates_so_far[mf.men] / d.guaranteed_per_month_paying,
            np.log1p(waiting[mf.men]),
            np.full(m, weeks_left / d.weeks_per_month),
        ]
    )


def delivered_progress_matrix(
    mf: MarketFeatures,
    delivered: np.ndarray,
    upcoming: np.ndarray,
    days_left: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Start features plus progress in each man's own cycle (FR-LM-44).

    `delivered`, `upcoming` and `days_left` are indexed by member id: dates delivered so far
    this cycle, confirmed dates still to come in it, and days until it ends.
    """
    return np.column_stack(
        [
            start_matrix(mf),
            delivered[mf.men] / cfg.dates.guaranteed_per_month_paying,
            np.log1p(upcoming[mf.men]),
            days_left[mf.men] / cfg.billing.cycle_days,
        ]
    )


@dataclass(frozen=True)
class ForecastModel:
    start: np.ndarray  # coefficients for START_FEATURES
    progress: np.ndarray  # coefficients for progress_features(kind)
    note: str = ""
    fitted_women_per_man: tuple[float, float] = (0.0, math.inf)  # range seen when fitting
    kind: str = "booked"  # what it forecasts: booked dates (Sprint 6) or delivered (Sprint 8)
    assumptions: dict = field(default_factory=dict)  # settings the fit depended on

    def p_start(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(x @ self.start)

    def p_progress(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(x @ self.progress)

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        def row(names, coefs):
            return "\n".join(f'"{n}" = {c:.6f}' for n, c in zip(names, coefs, strict=True))

        lo, hi = self.fitted_women_per_man
        what = "FR-LM-43, FR-LM-44" if self.kind == "delivered" else "FR-LM-33"
        flag = " --delivered" if self.kind == "delivered" else ""
        text = (
            f"# Guarantee forecast coefficients ({what}). Generated by\n"
            f"# `python -m lastfirstdate.forecast_fit{flag}`; refit on real cycles once they"
            " exist.\n"
            f'kind = "{self.kind}"\n'
            f'note = "{self.note}"\n'
            f"fitted_women_per_man = [{lo:.4f}, {hi:.4f}]\n\n"
        )
        if self.assumptions:
            text += (
                "[assumptions]\n"
                + "".join(
                    f"{k} = {str(v).lower() if isinstance(v, bool) else v}\n"
                    for k, v in self.assumptions.items()
                )
                + "\n"
            )
        text += (
            f"[start]\n{row(START_FEATURES, self.start)}\n\n"
            f"[progress]\n{row(progress_features(self.kind), self.progress)}\n"
        )
        Path(path).write_text(text)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> ForecastModel:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        kind = raw.get("kind", "booked")
        if kind not in KINDS:
            raise ValueError(f"forecast kind must be one of {KINDS}, not {kind!r}")
        return cls(
            start=np.array([raw["start"][n] for n in START_FEATURES]),
            progress=np.array([raw["progress"][n] for n in progress_features(kind)]),
            note=raw.get("note", ""),
            fitted_women_per_man=tuple(raw.get("fitted_women_per_man", (0.0, math.inf))),
            kind=kind,
            assumptions=dict(raw.get("assumptions", {})),
        )


def gate_model(cfg: Config) -> ForecastModel:
    """The forecast the gate uses, chosen by `gate.forecast` (FR-LM-45)."""
    path = DELIVERED_MODEL_PATH if cfg.gate.forecast == "delivered" else DEFAULT_MODEL_PATH
    return ForecastModel.load(path)


def fit_forecast(
    x_start: np.ndarray,
    y_start: np.ndarray,
    x_progress: np.ndarray,
    y_progress: np.ndarray,
    l2: float,
    note: str = "",
    fitted_women_per_man: tuple[float, float] = (0.0, math.inf),
    kind: str = "booked",
    assumptions: dict | None = None,
) -> ForecastModel:
    """Fit both logistic regressions (a small penalty keeps them stable).

    `y_start` may hold shares between 0 and 1 (for example, the share of a man's cycles that
    reached 3 delivered dates); logistic regression fits those as they are.
    """
    start, _ = fit_logistic(x_start, y_start, l2)
    progress, _ = fit_logistic(x_progress, y_progress, l2)
    return ForecastModel(start, progress, note, fitted_women_per_man, kind, assumptions or {})


def forecast_fulfilment(model: ForecastModel, mf: MarketFeatures) -> float:
    """Forecast share of the men in `mf` who reach their guarantee this cycle."""
    if mf.men.size == 0:
        return 1.0
    return float(model.p_start(start_matrix(mf)).mean())


def at_risk(
    model: ForecastModel,
    mf: MarketFeatures,
    dates_so_far: np.ndarray,
    waiting: np.ndarray,
    weeks_left: int,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Men below `gate.at_risk_threshold` chance of reaching the guarantee, and every chance."""
    p = model.p_progress(progress_matrix(mf, dates_so_far, waiting, weeks_left, cfg))
    return mf.men[p < cfg.gate.at_risk_threshold], p


def at_risk_delivered(
    model: ForecastModel,
    mf: MarketFeatures,
    delivered: np.ndarray,
    upcoming: np.ndarray,
    days_left: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Men below `gate.at_risk_threshold` chance of ending their own cycle with 3 delivered
    dates, and every chance (FR-LM-44). Needs a model fitted on delivered dates."""
    if model.kind != "delivered":
        raise ValueError("delivered-date alerts need a forecast fitted on delivered dates")
    p = model.p_progress(delivered_progress_matrix(mf, delivered, upcoming, days_left, cfg))
    return mf.men[p < cfg.gate.at_risk_threshold], p


@dataclass
class GateDecision:
    admit: list[int] = field(default_factory=list)
    deferred_small_pool: list[int] = field(default_factory=list)
    women_per_man: float = 0.0
    forecast: float = 1.0
    stopped_because: str = "waitlist empty"

    def text(self) -> str:
        return (
            f"admit {len(self.admit)} men; deferred {len(self.deferred_small_pool)} with small "
            f"pools; then {self.women_per_man:.2f} women per man, forecast fulfilment "
            f"{self.forecast:.1%}; stopped because: {self.stopped_because}"
        )


def gate_decision(
    model: ForecastModel,
    eligible: np.ndarray,
    women: Sequence[int],
    men: Sequence[int],
    applicants: Sequence[int],
    cfg: Config,
) -> GateDecision:
    """How many waitlisted men to admit now, in waitlist order (FR-LM-34).

    Runs in Rust when the compiled core is present (`native.py`), updating each man's
    attention share as men are added instead of recounting it; the loop below is the
    reference version in Python.
    """
    g = cfg.gate
    women = list(women)
    current = list(men)
    if native.enabled():
        return _gate_decision_rust(model, eligible, women, current, list(applicants), g)
    out = GateDecision()
    mf = market_features(eligible, women, current)
    out.women_per_man, out.forecast = mf.women_per_man, forecast_fulfilment(model, mf)
    for a in applicants:
        if eligible[women, a].sum() < g.min_pool:
            out.deferred_small_pool.append(int(a))
            continue
        trial = current + out.admit + [int(a)]
        ratio = len(women) / len(trial)
        if ratio < g.min_women_per_man:
            out.stopped_because = _stop_message("ratio", model, g)
            break
        if ratio < model.fitted_women_per_man[0]:
            out.stopped_because = _stop_message("fitted", model, g)
            break
        trial_mf = market_features(eligible, women, trial)
        f = forecast_fulfilment(model, trial_mf)
        if f < g.target:
            out.stopped_because = _stop_message("forecast", model, g)
            break
        out.admit.append(int(a))
        out.women_per_man, out.forecast = ratio, f
    return out


def _stop_message(kind: str, model: ForecastModel, g: GateConfig) -> str:
    return {
        "waitlist empty": "waitlist empty",
        "ratio": f"women per man would fall below {g.min_women_per_man:g}",
        "fitted": f"the forecast was only fitted down to {model.fitted_women_per_man[0]:.2f} "
        "women per man",
        "forecast": f"forecast fulfilment would fall below {g.target:.0%}",
    }[kind]


def _gate_decision_rust(
    model: ForecastModel,
    eligible: np.ndarray,
    women: list[int],
    current: list[int],
    applicants: list[int],
    g: GateConfig,
) -> GateDecision:
    cols = [int(c) for c in current] + [int(a) for a in applicants]
    e = eligible[np.ix_(np.asarray(women, dtype=int), np.asarray(cols, dtype=int))]
    admit, deferred, ratio, forecast, stop = native.core.gate_scan(
        np.ascontiguousarray(e, dtype=bool),
        len(current),
        np.ascontiguousarray(model.start, dtype=float),
        float(g.min_pool),
        float(g.min_women_per_man),
        float(model.fitted_women_per_man[0]),
        float(g.target),
    )
    return GateDecision(
        admit=[cols[c] for c in admit],
        deferred_small_pool=[cols[c] for c in deferred],
        women_per_man=ratio,
        forecast=forecast,
        stopped_because=_stop_message(stop, model, g),
    )

"""Stages 2 and 3: the looks model (FR-LM-13 to FR-LM-16).

Stage 3, how much member i likes how member j looks:

    V_i(j) = sigmoid( (w_0[g] + delta_i) . x_j )

- x_j: j's photo embedding, shrunk to `calibration.pca_dim` numbers with PCA, plus a
  constant 1 so the model has an intercept (FR-LM-13).
- w_0[g]: the starting taste for i's orientation group g (for example women looking at
  men), fitted on everyone's pooled swipes (FR-LM-14). It is only a starting point under a
  member's own swipes and is never used to rank anyone by itself (rule R2).
- delta_i: i's personal correction, learned from their own swipes (FR-LM-15). A penalty
  (l2/2) * |delta_i|^2 keeps it near zero until there is evidence to move it. In Bayesian
  terms delta_i has a Gaussian prior centered on 0, so the fit is a Bayesian logistic
  regression; a Laplace approximation gives its uncertainty, which Stage 2 uses.

Stage 2, which ~50 profiles a new member swipes on (FR-LM-16):

- Profiles 1 to 20 cover as many different looks as possible: greedy selection for a
  determinantal point process (paper 2). Each next profile is the one least explained by
  those already chosen.
- Profiles 21 to 50 are the ones the model can learn most from: the largest expected
  information, log(1 + p(1 - p) * s^2), where p is the predicted like chance and s^2 the
  model's uncertainty about that profile (papers 1 and 3). Swipes that are near-certain
  teach little; so do profiles in directions the model already knows.

The engine never reads ground truth: swipes come in through a callback.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from lastfirstdate import native
from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import Config

MAX_NEWTON_STEPS = 50
NEWTON_TOL = 1e-8
MAX_STEP_HALVINGS = 30
OBJECTIVE_NOISE = 1e-12  # relative rise in the objective treated as rounding, not overshoot


class CalibrationIncomplete(RuntimeError):
    """Raised when looks scores are requested for a list before calibration is finished."""


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


# ---------------------------------------------------------------- FR-LM-13 photo embeddings


class PhotoEncoder(Protocol):
    """Turns members' photos into embeddings, one row per member.

    Production plugs in a pretrained image model such as CLIP. Tests use the synthetic
    embeddings through ArrayPhotoEncoder, so nothing is downloaded (R4).
    """

    def encode(self, member_ids: Sequence[int]) -> np.ndarray: ...


class ArrayPhotoEncoder:
    """Photo embeddings that were computed earlier and stored in an array."""

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = np.asarray(embeddings, dtype=float)

    def encode(self, member_ids: Sequence[int]) -> np.ndarray:
        return self.embeddings[np.asarray(member_ids, dtype=int)]


@dataclass(frozen=True)
class PCA:
    """Principal component analysis: keep the directions in which photos differ most."""

    mean: np.ndarray  # (d,)
    components: np.ndarray  # (k, d) unit directions, largest variance first
    explained: np.ndarray  # (k,) share of total variance each one keeps
    scale: float  # divides the output so the average squared length is 1

    @classmethod
    def fit(cls, embeddings: np.ndarray, k: int) -> PCA:
        x = np.asarray(embeddings, dtype=float)
        mean = x.mean(axis=0)
        _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
        var = s**2
        z = (x - mean) @ vt[:k].T
        scale = float(np.sqrt(np.mean(np.sum(z**2, axis=1))))
        return cls(mean=mean, components=vt[:k], explained=var[:k] / var.sum(), scale=scale)

    @property
    def explained_total(self) -> float:
        return float(self.explained.sum())

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return ((np.asarray(embeddings, dtype=float) - self.mean) @ self.components.T) / self.scale


def with_intercept(x: np.ndarray) -> np.ndarray:
    """Append a column of ones, so a weight vector also carries an intercept."""
    return np.hstack([x, np.ones((x.shape[0], 1))])


def photo_features(encoder: PhotoEncoder, n_members: int, cfg: Config) -> tuple[PCA, np.ndarray]:
    """Fit PCA on all members' photos; return it and every member's features (n, k + 1)."""
    raw = encoder.encode(range(n_members))
    pca = PCA.fit(raw, cfg.calibration.pca_dim)
    return pca, with_intercept(pca.transform(raw))


# ---------------------------------------------------------------- logistic regression core


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    l2: float,
    prior_mean: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """MAP logistic regression by Newton's method.

    Minimizes  sum_k s_k * logloss(y_k, sigmoid(offset_k + x_k . w))  +  (l2/2) |w - m|^2,
    where m is `prior_mean`. Returns the weights and the Laplace covariance (the inverse
    of the Hessian at the optimum), which measures how unsure the weights are.

    Runs in Rust when the compiled core is present (`native.py`); `_fit_logistic_python`
    is the same algorithm in NumPy.
    """
    x = np.ascontiguousarray(x, dtype=float)
    n, d = x.shape
    y = np.ascontiguousarray(y, dtype=float)
    m = np.zeros(d) if prior_mean is None else np.ascontiguousarray(prior_mean, dtype=float)
    o = np.zeros(n) if offset is None else np.ascontiguousarray(offset, dtype=float)
    s = np.ones(n) if sample_weight is None else np.ascontiguousarray(sample_weight, dtype=float)
    w = m.copy() if start is None else np.array(start, dtype=float)
    if native.enabled():
        return native.core.fit_logistic(
            x, y, float(l2), m, o, s, w, MAX_NEWTON_STEPS, NEWTON_TOL, MAX_STEP_HALVINGS
        )
    return _fit_logistic_python(x, y, l2, m, o, s, w)


def _logistic_objective(x, y, l2, m, o, s, w) -> float:
    """The penalised log loss that fit_logistic minimises (logits clipped like sigmoid)."""
    z = np.clip(o + x @ w, -35.0, 35.0)
    softplus = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))  # log(1 + e^z)
    return float(s @ (softplus - y * z) + 0.5 * l2 * np.sum((w - m) ** 2))


def _fit_logistic_python(x, y, l2, m, o, s, w) -> tuple[np.ndarray, np.ndarray]:
    """The Newton loop in NumPy: the reference the Rust version is checked against.

    Step halving: if a full Newton step would make the objective worse, try half of it, then
    a quarter, and so on. Plain Newton can overshoot and diverge on hard problems. Steps
    already below the tolerance are always taken in full, and a rise smaller than
    OBJECTIVE_NOISE (relative) counts as rounding, so near the optimum, and whenever every
    full step helps (all calibration fits in the simulation), the path is plain Newton's.
    """
    eye = np.eye(x.shape[1])
    f = _logistic_objective(x, y, l2, m, o, s, w)
    for _ in range(MAX_NEWTON_STEPS):
        p = sigmoid(o + x @ w)
        grad = x.T @ (s * (p - y)) + l2 * (w - m)
        hess = (x.T * (s * p * (1 - p))) @ x + l2 * eye
        step = np.linalg.solve(hess, grad)
        if np.max(np.abs(step)) < NEWTON_TOL:
            w = w - step
            break
        t = 1.0
        for _ in range(MAX_STEP_HALVINGS):
            trial = w - t * step
            f_trial = _logistic_objective(x, y, l2, m, o, s, trial)
            if f_trial <= f + OBJECTIVE_NOISE * (abs(f) + 1.0):
                break
            t *= 0.5
        w, f = trial, f_trial
    p = sigmoid(o + x @ w)
    hess = (x.T * (s * p * (1 - p))) @ x + l2 * eye
    return w, np.linalg.inv(hess)


# ---------------------------------------------------------------- FR-LM-14 starting taste


@dataclass(frozen=True)
class Swipe:
    viewer_gender: int
    candidate_gender: int
    features: np.ndarray  # (k + 1,) the candidate's photo features
    liked: bool


@dataclass(frozen=True)
class StartingTaste:
    """w_0 for each (viewer gender, candidate gender) group. A prior, never a ranking (R2)."""

    by_group: dict[tuple[int, int], np.ndarray]
    pooled: np.ndarray

    @classmethod
    def fit(cls, swipes: Sequence[Swipe], l2: float) -> StartingTaste:
        """Fit one pooled taste, then each group's taste shrunk toward the pooled one.

        Shrinking toward the pooled fit means small groups (for example two women) borrow
        strength from everyone instead of overfitting their few swipes.
        """
        if not swipes:
            raise ValueError("the starting taste needs some swipes to learn from")
        x = np.array([s.features for s in swipes])
        y = np.array([s.liked for s in swipes], dtype=float)
        pooled, _ = fit_logistic(x, y, l2)
        groups: dict[tuple[int, int], list[int]] = {}
        for k, s in enumerate(swipes):
            groups.setdefault((s.viewer_gender, s.candidate_gender), []).append(k)
        by_group = {
            g: fit_logistic(x[idx], y[idx], l2, prior_mean=pooled, start=pooled)[0]
            for g, idx in groups.items()
        }
        return cls(by_group=by_group, pooled=pooled)

    def offsets(
        self, viewer_gender: int, features: np.ndarray, candidate_genders: np.ndarray
    ) -> np.ndarray:
        """w_0[g] . x_j for each candidate, using the pooled taste for unseen groups."""
        out = np.empty(features.shape[0])
        for g in np.unique(candidate_genders):
            rows = candidate_genders == g
            w0 = self.by_group.get((viewer_gender, int(g)), self.pooled)
            out[rows] = features[rows] @ w0
        return out


# ---------------------------------------------------------------- FR-LM-15 personal taste


@dataclass
class PersonalTaste:
    """One member's looks taste: the starting taste plus their personal correction."""

    viewer_gender: int
    start: StartingTaste
    l2: float
    min_swipes_for_lists: int
    features: list[np.ndarray] = field(default_factory=list)
    candidate_genders: list[int] = field(default_factory=list)
    liked: list[bool] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)  # 1.0 per swipe; more for date outcomes
    delta: np.ndarray | None = None
    cov: np.ndarray | None = None
    complete: bool = False  # set when calibration has finished

    def __post_init__(self) -> None:
        d = next(iter(self.start.by_group.values()), self.start.pooled).shape[0]
        if self.delta is None:
            self.delta = np.zeros(d)
        if self.cov is None:
            self.cov = np.eye(d) / self.l2

    @property
    def n_swipes(self) -> int:
        return len(self.liked)

    def add_swipe(
        self, features: np.ndarray, candidate_gender: int, liked: bool, refit: bool = True
    ) -> None:
        """Record one swipe and (by default) refit the personal correction."""
        self.add_label(features, candidate_gender, liked, 1.0, refit)

    def add_label(
        self,
        features: np.ndarray,
        candidate_gender: int,
        liked: bool,
        weight: float,
        refit: bool = True,
    ) -> None:
        """Record one labeled example with a weight (a date outcome counts more than a swipe)."""
        self._pad_weights()
        self.features.append(np.asarray(features, dtype=float))
        self.candidate_genders.append(int(candidate_gender))
        self.liked.append(bool(liked))
        self.weights.append(float(weight))
        if refit:
            self.refit()

    def add_pair(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        candidate_gender: int,
        chose_a: bool,
        weight: float = 1.0,
        refit: bool = True,
    ) -> None:
        """Record a pair question ("Who are you more drawn to?").

        Bradley-Terry: P(A over B) = sigmoid((w_0 + delta) . (x_A - x_B)). That is one more
        labeled example whose features are the difference of the two photos (the intercept
        cancels), so the same fit learns from singles and pairs together. Both photos must
        come from the same orientation group.
        """
        diff = np.asarray(features_a, dtype=float) - np.asarray(features_b, dtype=float)
        self.add_label(diff, candidate_gender, chose_a, weight, refit)

    def _pad_weights(self) -> None:
        # examples added without a weight (for example set directly) count as swipes
        self.weights.extend([1.0] * (len(self.liked) - len(self.weights)))

    def refit(self) -> None:
        self._pad_weights()
        x = np.array(self.features)
        g = np.array(self.candidate_genders)
        offset = self.start.offsets(self.viewer_gender, x, g)
        self.delta, self.cov = fit_logistic(
            x,
            np.array(self.liked),
            self.l2,
            offset=offset,
            sample_weight=np.array(self.weights),
            start=self.delta,
        )

    def logits(self, features: np.ndarray, candidate_genders: np.ndarray) -> np.ndarray:
        return (
            self.start.offsets(self.viewer_gender, features, candidate_genders)
            + features @ self.delta
        )

    def like_chance(self, features: np.ndarray, candidate_genders: np.ndarray) -> np.ndarray:
        """Predicted chance this member likes each candidate's looks."""
        return sigmoid(self.logits(features, candidate_genders))

    def uncertainty(self, features: np.ndarray) -> np.ndarray:
        """s^2 = x' cov x: how unsure the model is about each candidate's logit."""
        if native.enabled():
            return native.core.quad_forms(
                np.ascontiguousarray(features, dtype=float),
                np.ascontiguousarray(self.cov, dtype=float),
            )
        return np.einsum("ij,jk,ik->i", features, self.cov, features)

    def information(self, features: np.ndarray, candidate_genders: np.ndarray) -> np.ndarray:
        """Expected information from one more swipe on each candidate."""
        p = self.like_chance(features, candidate_genders)
        return np.log1p(p * (1 - p) * self.uncertainty(features))

    def list_scores(self, features: np.ndarray, candidate_genders: np.ndarray) -> np.ndarray:
        """V_i(j) for ranking a daily list. Refuses until calibration is complete (R2).

        Complete means `min_swipes_for_lists` swipes, or calibration finished early because
        the member's pool ran out of profiles.
        """
        if not self.complete and self.n_swipes < self.min_swipes_for_lists:
            raise CalibrationIncomplete(
                f"{self.n_swipes} swipes so far; lists need {self.min_swipes_for_lists}"
            )
        return self.like_chance(features, candidate_genders)


# ---------------------------------------------------------------- FR-LM-16 calibration


def calibration_pool(arr: MemberArrays, cfg: Config, viewer: int) -> np.ndarray:
    """Ids a member may see while calibrating (decision D12).

    Matchable members of a gender the viewer seeks and inside the viewer's age range.
    Distance and the other filters are ignored: the goal is to learn taste, and the
    viewer's own match pool is often too small to supply 50 varied profiles.
    """
    ok = arr.last_active <= cfg.filter.active_within_days
    if cfg.filter.require_verified:
        ok = ok & arr.verified
    ok &= (arr.seeking[viewer] & arr.gender_bit) != 0
    ok &= (arr.age_min[viewer] <= arr.age) & (arr.age <= arr.age_max[viewer])
    ok[viewer] = False
    return np.flatnonzero(ok)


def select_varied(features: np.ndarray, k: int) -> list[int]:
    """Row indices of k varied profiles: greedy determinantal point process (paper 2).

    With similarity = cosine between photo directions, a DPP favors sets whose photo
    vectors span the most volume. Greedily, each next pick is the profile whose photo is
    least explained by (has the largest part outside the span of) those already chosen.
    The first pick is the profile closest to the average look.
    """
    v = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    n = v.shape[0]
    k = min(k, n)
    if k == 0:
        return []
    first = int(np.argmax(v @ v.mean(axis=0)))
    chosen = [first]
    residual = v.copy()
    while len(chosen) < k:
        b = residual[chosen[-1]]
        norm = np.linalg.norm(b)
        if norm > 1e-12:
            b = b / norm
            residual = residual - np.outer(residual @ b, b)
        score = np.sum(residual**2, axis=1)
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return chosen


def select_pair(
    taste: PersonalTaste,
    remaining: np.ndarray,
    features: np.ndarray,
    genders: np.ndarray,
    top: int = 30,
) -> tuple[int, int] | None:
    """The most informative pair question from the remaining profiles.

    Among the `top` profiles the model is least sure about, every same-gender pair (A, B) is
    scored by the expected information from asking it, log(1 + p(1 - p) s^2), where p is the
    predicted chance of choosing A and s^2 the model's uncertainty about the difference
    x_A - x_B. Pairs whose answer is a coin flip AND that the model knows little about win.
    """
    if remaining.size < 2:
        return None
    unc = taste.uncertainty(features[remaining])
    cand = remaining[np.argsort(-unc, kind="stable")[:top]]
    logit = taste.logits(features[cand], genders[cand])
    if native.enabled():
        pos = native.core.best_pair(
            np.ascontiguousarray(features[cand], dtype=float),
            np.ascontiguousarray(logit, dtype=float),
            np.ascontiguousarray(genders[cand], dtype=np.int64),
            np.ascontiguousarray(taste.cov, dtype=float),
        )
        return None if pos is None else (int(cand[pos[0]]), int(cand[pos[1]]))
    best, pick = -np.inf, None
    for a_pos in range(cand.size):
        for b_pos in range(a_pos + 1, cand.size):
            a, b = int(cand[a_pos]), int(cand[b_pos])
            if genders[a] != genders[b]:
                continue
            diff = features[a] - features[b]
            p = sigmoid(logit[a_pos] - logit[b_pos])
            s2 = float(diff @ taste.cov @ diff)
            info = np.log1p(p * (1 - p) * s2)
            if info > best:
                best, pick = info, (a, b)
    return pick


def calibrate(
    taste: PersonalTaste,
    pool_ids: np.ndarray,
    features: np.ndarray,
    genders: np.ndarray,
    swipe: Callable[[int], bool],
    cfg: Config,
    strategy: str = "calibrated",
    rng: np.random.Generator | None = None,
    pair: Callable[[int, int], bool] | None = None,
) -> list[int]:
    """Show up to `calibration.n_profiles` photos from the pool and learn from each answer.

    strategy "calibrated": `calibration.n_variety` varied single photos; then, if
    `calibration.n_pairs` > 0 and a `pair` callback is given, that many pair questions
    (two photos each); then informative single photos for whatever is left. strategy
    "random": single photos drawn at random (the comparison baseline). `features` and
    `genders` cover all members; `swipe(j)` says whether the member liked j and
    `pair(a, b)` whether they chose a over b. Returns the ids shown, in order.
    """
    pool_ids = np.asarray(pool_ids, dtype=int)
    k = min(cfg.calibration.n_profiles, pool_ids.size)
    shown: list[int] = []
    if strategy == "random":
        if rng is None:
            raise ValueError("strategy 'random' needs an rng")
        order = rng.permutation(pool_ids)[:k]
        for j in order:
            taste.add_swipe(features[j], genders[j], swipe(int(j)))
            shown.append(int(j))
        taste.complete = True
        return shown
    if strategy != "calibrated":
        raise ValueError(f"unknown strategy {strategy!r}")
    pool_x = features[pool_ids][:, :-1]  # without the intercept column
    for row in select_varied(pool_x, min(cfg.calibration.n_variety, k)):
        j = int(pool_ids[row])
        # variety picks don't depend on the model, so fit once after all of them
        taste.add_swipe(features[j], genders[j], swipe(j), refit=False)
        shown.append(j)
    if shown:
        taste.refit()
    remaining = np.setdiff1d(pool_ids, shown)
    n_pairs = cfg.calibration.n_pairs if pair is not None else 0
    for _ in range(n_pairs):
        if len(shown) + 2 > k:
            break
        chosen = select_pair(taste, remaining, features, genders)
        if chosen is None:
            break
        a, b = chosen
        taste.add_pair(features[a], features[b], genders[a], pair(a, b))
        shown += [a, b]
        remaining = remaining[(remaining != a) & (remaining != b)]
    while len(shown) < k and remaining.size:
        info = taste.information(features[remaining], genders[remaining])
        j = int(remaining[int(np.argmax(info))])
        taste.add_swipe(features[j], genders[j], swipe(j))
        shown.append(j)
        remaining = remaining[remaining != j]
    taste.complete = True
    return shown

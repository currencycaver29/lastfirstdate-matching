"""Behavior simulator: how fake members pick, accept and feel about dates (FR-LM-11).

This is the stand-in for real people. It decides only from the hidden ground truth and the
profiles, never from the engine's scores, so the engine can be judged against it.

Each fake member i has a true appeal for every other member j, in standard-deviation units:

    U_ij = looks_weight_i * looks_ij + (1 - looks_weight_i) * fit_ij

- looks_ij: how i's hidden taste u_i lines up with j's photo, u_i . e_j, standardized.
- fit_ij: true personality fit = s * trait_fit_ij + sqrt(1 - s^2) * chemistry_ij.
  trait_fit is computed here from the profiles with equal weights (deliberately not the
  engine's 0.4 / 0.3 / 0.3), and chemistry is random noise shared by both people that
  no profile can predict. s = simulation.personality_signal (0.5): paper 9 found that
  traits measured before meeting predict desire for a specific person only weakly.

Choices use a logistic curve: P(yes) = 1 / (1 + exp(-(U - threshold - pickiness) / T)).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial.distance import cdist

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import Config
from lastfirstdate.synthetic import Population


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _standardize_rows(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    off = ~np.eye(n, dtype=bool)
    vals = np.where(off, x, np.nan)
    mean = np.nanmean(vals, axis=1, keepdims=True)
    std = np.nanstd(vals, axis=1, keepdims=True)
    return np.where(off, (x - mean) / np.where(std == 0, 1.0, std), 0.0)


def _standardize(x: np.ndarray) -> np.ndarray:
    off = ~np.eye(x.shape[0], dtype=bool)
    mean, std = x[off].mean(), x[off].std()
    return np.where(off, (x - mean) / (std if std > 0 else 1.0), 0.0)


def trait_fit(pop: Population, arr: MemberArrays) -> np.ndarray:
    """(n, n) how well two profiles fit on interests, values and Big Five, equally weighted.

    Simulator-only code, written separately from the engine's personality score.
    """
    m = pop.members
    tags = sorted({t for x in m for t in x.interests})
    index = {t: k for k, t in enumerate(tags)}
    x = np.zeros((len(m), len(tags)))
    for row, member in enumerate(m):
        x[row, [index[t] for t in member.interests]] = 1.0
    inter = x @ x.T
    size = x.sum(axis=1)
    jaccard = inter / np.maximum(size[:, None] + size[None, :] - inter, 1.0)
    agree = np.zeros_like(jaccard)
    for q in range(arr.answer.shape[1]):
        ok = ((arr.accepts[:, q][:, None] >> arr.answer[:, q][None, :]) & 1).astype(bool)
        agree += ok & ok.T
    agree /= arr.answer.shape[1]
    big5 = 1.0 - cdist(arr.big5, arr.big5) / np.sqrt(5.0)
    return (_standardize(jaccard) + _standardize(agree) + _standardize(big5)) / 3.0


class Behavior:
    """Decisions of fake members. Reads ground truth; the engine never does."""

    def __init__(self, pop: Population, arr: MemberArrays, cfg: Config, rng: np.random.Generator):
        sim = cfg.simulation
        self.cfg = sim
        n = len(pop)
        looks = _standardize_rows(pop.truth.taste @ pop.photo_embeddings.T)
        chem = rng.normal(size=(n, n))
        chem = (chem + chem.T) / np.sqrt(2.0)
        s = sim.personality_signal
        fit = s * _standardize(trait_fit(pop, arr)) + np.sqrt(1.0 - s * s) * chem
        lw = pop.truth.looks_weight[:, None]
        self.looks = looks  # true looks appeal, standardized per viewer
        self.utility = lw * looks + (1.0 - lw) * fit
        np.fill_diagonal(self.utility, -np.inf)
        self.pickiness = pop.truth.pickiness

    def _p(self, u, threshold, pickiness):
        return sigmoid((u - threshold - pickiness) / self.cfg.choice_temperature)

    def choose(
        self,
        i: int,
        shown: Sequence[int],
        rng: np.random.Generator,
        passes: np.ndarray | None = None,
    ) -> list[int]:
        """Profiles chooser i picks from today's list: liked ones, best first, up to the cap.

        `passes` counts how often i already saw each shown profile without picking it. Each
        earlier pass multiplies the chance of picking by `simulation.repeat_pick_factor`, so
        showing someone again is not treated as a fresh chance.
        """
        shown = np.asarray(shown, dtype=int)
        if shown.size == 0:
            return []
        u = self.utility[i, shown]
        p = self._p(u, self.cfg.like_threshold, self.pickiness[i])
        if passes is not None:
            p = p * self.cfg.repeat_pick_factor ** np.asarray(passes, dtype=float)
        liked = rng.random(shown.size) < p
        picks = shown[liked][np.argsort(-u[liked], kind="stable")]
        return [int(j) for j in picks[: self.cfg.picks_per_day]]

    def swipe_outcomes(self, i: int, seed: int) -> np.ndarray:
        """(n,) every calibration swipe outcome for member i, fixed in advance.

        Drawn from a generator seeded by (seed, i), so the same member gives the same
        answer to the same profile however many times or in whatever order it is shown
        (common random numbers).
        """
        n = self.looks.shape[0]
        u = np.random.default_rng([seed, int(i)]).random(n)
        return u < self._p(self.looks[i], self.cfg.swipe_threshold, self.pickiness[i])

    def swipe(self, i: int, j: int, rng: np.random.Generator) -> bool:
        """Calibration swipe: does i like j's photos? Decided on true looks appeal only."""
        p = self._p(self.looks[i, j], self.cfg.swipe_threshold, self.pickiness[i])
        return bool(rng.random() < p)

    def accepts(self, j: int, i: int, rng: np.random.Generator) -> bool:
        """Does j accept being picked by i?"""
        p = self._p(self.utility[j, i], self.cfg.accept_threshold, self.pickiness[j])
        return bool(rng.random() < p)

    def shows_up(self, rng: np.random.Generator) -> bool:
        return bool(rng.random() < self.cfg.show_up_rate)

    def wants_again(self, i: int, j: int, rng: np.random.Generator) -> bool:
        """After meeting, does i want a second date with j? Meeting adds new information."""
        u = self.utility[i, j] + rng.normal(0.0, self.cfg.meeting_noise)
        return bool(rng.random() < self._p(u, self.cfg.again_threshold, 0.0))

"""Private ratings and the trust score (FR-LM-39).

After a date, each person privately rates the other on respectfulness, profile accuracy and
conversation (1 to 5) and says whether they'd meet again. lastfirstdate uses "a private trust
score [that] feeds ranking and safety; men whose dates consistently rate poorly are flagged
for review".

What trust may use. Only behaviour: was he respectful, did he look and act like his profile,
did he come and not cancel at the last minute. Conversation and "would meet again" are about
chemistry, which is personal (one person's dull date is another's great one); they feed
learning (Stage 8), never trust. Otherwise trust would become a popularity score, which rule
R2 forbids.

The maths, in plain words:

    conduct     = (k * prior + sum of scaled ratings) / (k + number of ratings)
    reliability = (k * prior + dates kept) / (k + dates scheduled)
    trust       = (1 - w) * conduct + w * reliability

A rating r becomes (r - 1) / 4, so 1 to 5 maps to 0 to 1, and each rating contributes the
mean of its respectfulness and accuracy. k = `trust.prior_strength`: the prior counts as k
ratings, so one bad review can't sink anyone and a newcomer starts at the prior.

Ranking: low trust lowers a member's scores on everyone's lists,

    factor = 1 - ranking_weight * max(0, prior - trust) / prior

which is 1 for anyone at or above the prior: no boost, so trust never becomes something to
compete on, and newcomers are never penalised. Safety: conduct below `trust.flag_below` with
at least `trust.flag_min_ratings` ratings flags the member for review and pauses their profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lastfirstdate.config import Config

RATING_FIELDS = ("respect", "accuracy", "conversation")
TRUST_FIELDS = ("respect", "accuracy")  # the only ratings trust may read


@dataclass(frozen=True)
class Rating:
    rater: int
    rated: int
    respect: int
    accuracy: int
    conversation: int
    again: bool

    def __post_init__(self) -> None:
        if self.rater == self.rated:
            raise ValueError("members cannot rate themselves")
        for name in RATING_FIELDS:
            if getattr(self, name) not in (1, 2, 3, 4, 5):
                raise ValueError(f"{name} must be a whole number from 1 to 5")


class TrustBook:
    """Everyone's conduct ratings and attendance, and the scores built from them."""

    def __init__(self, n: int, cfg: Config):
        self.cfg = cfg.trust
        self.conduct_sum = np.zeros(n)
        self.n_ratings = np.zeros(n, dtype=int)
        self.kept = np.zeros(n)
        self.scheduled = np.zeros(n, dtype=int)

    def add_rating(self, rating: Rating) -> None:
        scaled = [(getattr(rating, f) - 1) / 4 for f in TRUST_FIELDS]
        self.conduct_sum[rating.rated] += float(np.mean(scaled))
        self.n_ratings[rating.rated] += 1

    def add_attendance(self, member: int, kept: bool) -> None:
        """One scheduled date: kept, or missed (no-show or late cancellation by them)."""
        self.scheduled[member] += 1
        self.kept[member] += float(kept)

    def _shrunk(self, total: np.ndarray, count: np.ndarray) -> np.ndarray:
        k = self.cfg.prior_strength
        return (k * self.cfg.prior_mean + total) / (k + count)

    def conduct(self) -> np.ndarray:
        return self._shrunk(self.conduct_sum, self.n_ratings)

    def reliability(self) -> np.ndarray:
        return self._shrunk(self.kept, self.scheduled)

    def trust(self) -> np.ndarray:
        # written around the prior, so a member with no data gets exactly the prior
        w, prior = self.cfg.reliability_weight, self.cfg.prior_mean
        return prior + (1 - w) * (self.conduct() - prior) + w * (self.reliability() - prior)

    def flagged(self) -> np.ndarray:
        """Members to review: conduct below the threshold with enough ratings."""
        return (self.n_ratings >= self.cfg.flag_min_ratings) & (
            self.conduct() < self.cfg.flag_below
        )

    def ranking_factor(self) -> np.ndarray:
        prior = self.cfg.prior_mean
        return 1.0 - self.cfg.ranking_weight * np.maximum(0.0, prior - self.trust()) / prior


def with_trust(scores: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """List scores with each candidate's (column's) trust factor applied."""
    return scores * factor[None, :]

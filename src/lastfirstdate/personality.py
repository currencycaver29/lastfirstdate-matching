"""Stage 4: the personality and interests score (FR-LM-05 to FR-LM-08).

    P_i(j) = w_text * t_ij + w_values * A_ij + w_big5 * p_ij

- t: interest similarity, the cosine of two interest vectors (FR-LM-05). Today the
  vectors come from interest tags; Sentence-BERT (paper 8) can replace them later through
  the TextEncoder interface without changing anything else.
- A: share of values questions where each answer is one the other accepts (FR-LM-06).
- p: Big Five similarity, 1 - distance / sqrt(5) (FR-LM-07).

All three run from 0 to 1, so P does too. The 0.4 / 0.3 / 0.3 weights are a starting
guess: paper 9 found traits measured before meeting predict attraction to one specific
person only weakly, so Sprint 5 retunes them from real date outcomes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from scipy.spatial.distance import cdist

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import Config
from lastfirstdate.models import Member


class TextEncoder(Protocol):
    """Turns each member's interests into a vector. Rows must have length 1."""

    def encode(self, members: Sequence[Member]) -> np.ndarray: ...


class TagEncoder:
    """Interest tags as a weighted bag of words (TF-IDF style).

    Each tag gets weight idf = log((1 + N) / (1 + df)) + 1, where df is how many members
    list it. A shared rare interest ("opera") then counts for more than a shared common
    one ("coffee"). Offline and deterministic, so tests can use it (R4).
    """

    def encode(self, members: Sequence[Member]) -> np.ndarray:
        vocab = sorted({t for m in members for t in m.interests})
        index = {t: k for k, t in enumerate(vocab)}
        x = np.zeros((len(members), len(vocab)))
        for row, m in enumerate(members):
            for t in m.interests:
                x[row, index[t]] = 1.0
        df = x.sum(axis=0)
        idf = np.log((1 + len(members)) / (1 + df)) + 1.0
        x *= idf
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms == 0, 1.0, norms)


def interest_similarity(vectors: np.ndarray) -> np.ndarray:
    """(n, n) t_ij: cosine similarity of unit interest vectors, clipped to 0..1."""
    return np.clip(vectors @ vectors.T, 0.0, 1.0)


def values_compatibility(arr: MemberArrays) -> np.ndarray:
    """(n, n) A_ij: share of questions where each one's answer is accepted by the other."""
    n, nq = arr.answer.shape
    total = np.zeros((n, n))
    for q in range(nq):
        accepted = ((arr.accepts[:, q][:, None] >> arr.answer[:, q][None, :]) & 1).astype(bool)
        total += accepted & accepted.T
    return total / nq


def big5_similarity(arr: MemberArrays) -> np.ndarray:
    """(n, n) p_ij = 1 - Euclidean distance between Big Five vectors / sqrt(5)."""
    return 1.0 - cdist(arr.big5, arr.big5) / np.sqrt(5.0)


def personality_matrix(
    members: Sequence[Member],
    arr: MemberArrays,
    cfg: Config,
    encoder: TextEncoder | None = None,
) -> np.ndarray:
    """(n, n) P_i(j). Symmetric, between 0 and 1."""
    w = cfg.personality
    t = interest_similarity((encoder or TagEncoder()).encode(members))
    a = values_compatibility(arr)
    p = big5_similarity(arr)
    return w.w_text * t + w.w_values * a + w.w_big5 * p

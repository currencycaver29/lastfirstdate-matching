"""Stage 4: the personality and interests score (FR-LM-05 to FR-LM-08).

    P_i(j) = w_text * t_ij + w_values * A_ij + w_big5 * p_ij

- t: interest similarity, the cosine of two interest vectors (FR-LM-05). The vectors come
  from interest tags, from word counts of free-text prompts, or in production from MiniLM, a
  Sentence-BERT model (paper 8, `encoders.SentenceEncoder`, FR-LM-61). With
  `personality.center_text` the zone's average vector is taken out first (FR-LM-64).
- A: share of values questions where each answer is one the other accepts (FR-LM-06).
- p: Big Five similarity, 1 - distance / sqrt(5) (FR-LM-07).

All three run from 0 to 1, so P does too. The 0.4 / 0.3 / 0.3 weights are a starting
guess: paper 9 found traits measured before meeting predict attraction to one specific
person only weakly, so Sprint 5 retunes them from real date outcomes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from scipy.spatial.distance import cdist

from lastfirstdate.arrays import MemberArrays
from lastfirstdate.config import Config, ValuesQuestion
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


STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "i",
        "i'm",
        "in",
        "is",
        "it",
        "it's",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "there's",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "with",
        "you",
        "your",
    ]
)


class WordEncoder:
    """Free-text prompts and bios as weighted word counts (TF-IDF), offline.

    The lastfirstdate app collects free-text prompts ("A perfect Sunday looks like…"). This
    is a simple stand-in that tests can run offline (R4): each member's text becomes a vector
    of word counts times idf = log((1 + N) / (1 + df)) + 1, scaled to length 1. Sentence-BERT
    (paper 8) replaces it in production through the same TextEncoder interface; it also
    understands that "hiking" and "trail runs" mean similar things, which word counts cannot.
    """

    def encode(self, members: Sequence[Member]) -> np.ndarray:
        docs = [self.tokens(m.bio) for m in members]
        vocab = sorted({w for d in docs for w in d})
        index = {w: k for k, w in enumerate(vocab)}
        x = np.zeros((len(members), len(vocab)))
        for row, words in enumerate(docs):
            for w in words:
                x[row, index[w]] += 1.0
        df = (x > 0).sum(axis=0)
        x *= np.log((1 + len(members)) / (1 + df)) + 1.0
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms == 0, 1.0, norms)

    @staticmethod
    def tokens(text: str) -> list[str]:
        words = re.findall(r"[a-z][a-z'-]*", text.lower())
        return [w for w in words if w not in STOPWORDS]


ENCODERS = {"tags": TagEncoder, "words": WordEncoder}


def text_encoder_for(cfg: Config) -> TextEncoder:
    """The encoder `personality.text_encoder` names; "sentence" is the production model."""
    if cfg.personality.text_encoder == "sentence":
        from lastfirstdate.encoders import sentence_encoder

        return sentence_encoder(cfg.models)
    return ENCODERS[cfg.personality.text_encoder]()


def interest_similarity(vectors: np.ndarray, center: bool = False) -> np.ndarray:
    """(n, n) t_ij: cosine similarity of unit interest vectors, clipped to 0..1.

    With `center` (FR-LM-64, D21) the zone's average vector is taken away from every member's
    first, and the rows are scaled back to length 1:

        t_ij = cos(x_i - mean, x_j - mean)

    What everyone writes then counts for nothing, and only what two people share beyond the
    zone's average makes them similar. So a prompt stuffed with popular words lands near the
    middle and gains nothing (the text version of the "average attack" on recommenders; the
    centring is Mu and Viswanath's "all-but-the-top", which also removes the common direction
    sentence models put in every vector). Empty texts stay zero vectors and score 0.
    """
    x = np.asarray(vectors, dtype=float)
    if center:
        has = np.linalg.norm(x, axis=1) > 0
        if has.any():
            x = np.where(has[:, None], x - x[has].mean(axis=0), 0.0)
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            x = x / np.where(norms == 0, 1.0, norms)
    return np.clip(x @ x.T, 0.0, 1.0)


def values_compatibility(
    arr: MemberArrays, questions: Sequence[ValuesQuestion] | None = None
) -> np.ndarray:
    """(n, n) A_ij: average compatibility over the values questions, from 0 to 1.

    "accepts" questions score 1 when each one's answer is accepted by the other, else 0.
    "closeness" questions score 1 - |rank_i - rank_j| / (options - 1): the same answer gives
    1, opposite ends give 0, neighbors something in between.
    """
    n, nq = arr.answer.shape
    total = np.zeros((n, n))
    for q in range(nq):
        question = questions[q] if questions is not None else None
        if question is not None and question.mode == "closeness":
            a = arr.answer[:, q]
            total += 1.0 - np.abs(a[:, None] - a[None, :]) / (len(question.options) - 1)
            continue
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
    encoder = encoder or text_encoder_for(cfg)
    t = interest_similarity(encoder.encode(members), center=w.center_text)
    a = values_compatibility(arr, cfg.values)
    p = big5_similarity(arr)
    return w.w_text * t + w.w_values * a + w.w_big5 * p

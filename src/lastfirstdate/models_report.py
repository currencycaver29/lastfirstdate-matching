"""Sprint 13 report: the real photo and text models, checked outside the tests.

Usage (after `pip install -e ".[models]"` and `python -m lastfirstdate.encoders download`):

    python -m lastfirstdate.models_report [cache_dir]

No real photos or member text are used (R3): the texts are made up, the photos are generated
patterns, and the population is synthetic.

1. Text: how close MiniLM and the offline word counts think some made-up prompts are.
2. Text on the synthetic population: how well each separates pairs who share a lifestyle
   (the hidden truth) from pairs who don't, as an AUC.
3. Photos: SigLIP's output size and length, repeatability, and speed.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from types import SimpleNamespace

import numpy as np

from lastfirstdate.config import load
from lastfirstdate.encoders import PhotoModel, SentenceEncoder, photo_model_id
from lastfirstdate.looks_report import auc
from lastfirstdate.personality import WordEncoder
from lastfirstdate.synthetic import generate

PAIRS = [
    ("Weekends are for rock climbing and long trail runs.", "I spend Sundays bouldering."),
    ("Weekends are for rock climbing and long trail runs.", "Opera nights and a good novel."),
    ("I cook Italian food for friends.", "My pasta carbonara is famous."),
    ("I cook Italian food for friends.", "I'm training for a marathon."),
    ("Looking for something serious, ready to settle down.", "I want a long-term partner."),
]


def member(text: str) -> SimpleNamespace:
    """A stand-in profile carrying only a prompt (both encoders read only `bio`)."""
    return SimpleNamespace(bio=text)


def pair_table(sentence: SentenceEncoder) -> list[tuple[str, str, float, float]]:
    rows = []
    words = WordEncoder()
    for a, b in PAIRS:
        w = words.encode([member(a), member(b)])
        s = sentence.embed_texts([a, b])
        rows.append((a, b, float(w[0] @ w[1]), float(s[0] @ s[1])))
    return rows


def lifestyle_auc(sentence: SentenceEncoder, seed: int = 7) -> dict[str, float]:
    """AUC of each encoder's similarity for same-lifestyle pairs among man-woman pairs."""
    cfg = load("config/phase1.toml")
    pop = generate(cfg, seed=seed)
    life = pop.truth.lifestyle
    men = np.array([m.gender == "man" for m in pop.members])
    same = (life[:, None] == life[None, :])[np.ix_(men, ~men)].ravel()
    out = {}
    for name, encoder in (("words", WordEncoder()), ("sentence", sentence)):
        t0 = time.perf_counter()
        vectors = encoder.encode(pop.members)
        out[f"{name}_ms"] = 1000 * (time.perf_counter() - t0) / len(pop.members)
        sim = (vectors @ vectors.T)[np.ix_(men, ~men)].ravel()
        out[name] = auc(sim, same)
    return out


def patterns(n: int, size: int = 256, seed: int = 0) -> list[np.ndarray]:
    """Generated test images: colour gradients with a few shapes (no real photos, R3)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    images = []
    for _ in range(n):
        c = rng.random((3, 3))
        img = np.stack([c[k, 0] + c[k, 1] * xx + c[k, 2] * yy for k in range(3)], axis=-1)
        for _ in range(3):
            cx, cy, r = rng.random(3) * [size, size, size / 4]
            disc = (np.mgrid[0:size, 0:size][1] - cx) ** 2 + (
                np.mgrid[0:size, 0:size][0] - cy
            ) ** 2 < r**2
            img[disc] = rng.random(3)
        images.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return images


def photo_check(photo: PhotoModel) -> dict[str, float]:
    images = patterns(8)
    t0 = time.perf_counter()
    vecs = np.array([photo.embed(im) for im in images])
    seconds = (time.perf_counter() - t0) / len(images)
    again = photo.embed(images[0])
    brighter = photo.embed(np.clip(images[0].astype(int) + 20, 0, 255).astype(np.uint8))
    others = vecs[1:] @ vecs[0]
    return {
        "dimension": vecs.shape[1],
        "length": float(np.linalg.norm(vecs, axis=1).mean()),
        "same_photo_difference": float(np.abs(again - vecs[0]).max()),
        "brighter_copy_similarity": float(brighter @ vecs[0]),
        "other_patterns_similarity": float(others.mean()),
        "seconds_per_photo": seconds,
    }


def main() -> None:
    m = load("config/phase1.toml").models
    if len(sys.argv) > 1:
        m = dataclasses.replace(m, cache_dir=sys.argv[1])
    sentence = SentenceEncoder.from_settings(m)
    print("1. Made-up prompts: similarity (0 = unrelated, 1 = the same)")
    print("   words  MiniLM  pair")
    for a, b, w, s in pair_table(sentence):
        print(f"   {w:5.2f}  {s:6.2f}  {a!r} / {b!r}")
    life = lifestyle_auc(sentence)
    print(
        f"\n2. Synthetic population, man-woman pairs: AUC for sharing a lifestyle\n"
        f"   words {life['words']:.3f}, MiniLM {life['sentence']:.3f}; time per prompt: words "
        f"{life['words_ms']:.2f} ms, MiniLM {life['sentence_ms']:.1f} ms"
    )
    photo = PhotoModel.from_settings(m)
    p = photo_check(photo)
    print(
        f"\n3. Photos ({photo_model_id(m)}): {p['dimension']:.0f} numbers, length "
        f"{p['length']:.3f}; the same photo twice differs by at most "
        f"{p['same_photo_difference']:.1e}; a brighter copy scores "
        f"{p['brighter_copy_similarity']:.2f} "
        f"against other patterns' {p['other_patterns_similarity']:.2f}; "
        f"about {p['seconds_per_photo']:.2f} s per photo"
    )


if __name__ == "__main__":
    main()

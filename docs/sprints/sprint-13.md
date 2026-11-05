# Sprint 13: the real photo and text models

**In one paragraph.** Until now the engine ran on stand-ins: synthetic photo vectors and word
counts for prompts. You chose the production models:
- **Google SigLIP base** for photos;
- **MiniLM** (all-MiniLM-L6-v2) for prompts.

Both are free for commercial use (Apache 2.0), run on a normal server's processor with
ONNX Runtime, and are pinned to an exact version with a fingerprint for every file. The tests
still run offline (rule R4), and no real photos or text enter the repo (R3). Checked once for
real on made-up texts and generated images, MiniLM scores "rock climbing" and "bouldering" at
0.67 where word counts scored 0, and SigLIP turns a photo into 768 numbers in about 0.1
seconds.

**Papers used.**
- **Paper 8**, Sentence-BERT: MiniLM is trained as a Sentence-BERT model.
- **Zhai et al. (2023), SigLIP**, and **Wang et al. (2020), MiniLM:** both now listed in
  `docs/research.md`.

---

## 1. What got built

| Change | Where |
|---|---|
| `[models]` settings: each model, exact revision, every file's sha256 | `config.py`, both TOMLs |
| `python -m lastfirstdate.encoders download`: fetch and check the files | `encoders.py` |
| `PhotoModel`: one photo in, 768 numbers out; `photo_model_id` | `encoders.py` |
| `SentenceEncoder`: prompts through MiniLM; `text_encoder = "sentence"` | `encoders.py`, `personality.py`, `app.py` |
| `app.check_photo_models`: one photo model at a time | `app.py` |
| Optional install: `pip install -e ".[models]"` (ONNX Runtime, tokenizers, Pillow) | `pyproject.toml` |
| `python -m lastfirstdate.models_report`: the real-model check | `models_report.py` |
| Decision D20; handoff section "The photo and text models" | `docs/SPEC.md`, `docs/handoff/README.md` |

---

## 2. Why these models

| | Licence | Notes |
|---|---|---|
| **Google SigLIP base** (chosen) | Apache 2.0 | Commercial use allowed; 768 numbers per photo |
| OpenAI CLIP | MIT code, but its model card says any deployed use, commercial or not, is "out of scope" | Risky for a paid app |
| **MiniLM, all-MiniLM-L6-v2** (chosen) | Apache 2.0 | 384 numbers per text; small and fast; English only |

**ONNX Runtime instead of PyTorch.** ONNX is a standard file format for trained models, and
ONNX Runtime is a small library that runs them on an ordinary processor. PyTorch would add
gigabytes and usually wants a graphics card. Here the whole install is a few tens of
megabytes plus the 460 MB of model files.

---

## 3. Pinned and checked files (FR-LM-59)

Every file is named by repository, **exact revision** (a 40-character commit, never "latest")
and **sha256**, a fingerprint of its contents. `download` fetches each file and compares
fingerprints:
- **Match:** it keeps the file.
- **No match:** it deletes the download and stops.
- **Already in place and matching:** it skips the file.

This protects against two real risks:
- **A silent update:** a model changing under you would quietly change every member's
  matches.
- **A tampered download:** a supply-chain attack.

The settings refuse anything that isn't a full commit hash and a full fingerprint.

---

## 4. Photos (FR-LM-60)

SigLIP expects photos prepared one exact way, and `PhotoModel` follows it:
1. Convert to RGB, resize to 224 by 224 (bicubic).
2. Scale each colour from 0..255 to -1..1: x / 255, minus 0.5, divided by 0.5.
3. Run the vision model; take its pooled output (768 numbers).
4. Scale the vector to length 1.

The app runs this when a member adds a photo, encrypts the vector (it's biometric data), and
stores it with `photo_model_id`, for example `google/siglip-base-patch16-224@4649052661e5`.

**One model at a time (FR-LM-62).** Vectors from different models live in different spaces,
like coordinates on two different maps. Mixing them would make nonsense matches, so the
nightly refresh refuses to run if any stored embedding names another model. Switching models
later means re-embedding every photo, then a refresh with `refit_start=True`.

**What the check showed.**
- 768 numbers of length 1, identical for the same photo, about 0.09 s per photo.
- Generated test patterns all scored about 0.93 with each other: SigLIP's vectors share a
  large common direction. The engine's photo compression (PCA) subtracts the average before
  compressing, which removes that shared part.

---

## 5. Prompts (FR-LM-61)

MiniLM turns a text into 384 numbers so that texts with similar meanings land close
together:
1. Split the prompt into word pieces (at most 256).
2. Run the model: one vector per word piece.
3. Average those vectors, **counting only the real word pieces**. Texts in a batch are padded
   to the same length, and padding must never change a text's meaning; a test checks this.
4. Scale to length 1. An empty prompt gives zeros, as with word counts.

| Made-up prompts | Word counts | MiniLM |
|---|---:|---:|
| rock climbing / bouldering | 0.00 | **0.67** |
| rock climbing / opera | 0.00 | 0.13 |
| Italian cooking / carbonara | 0.00 | **0.52** |
| Italian cooking / marathon | 0.00 | 0.20 |
| settle down / long-term partner | 0.00 | **0.45** |

**On the synthetic population** the gain is small: AUC 0.888 against 0.869 for telling pairs
who share a lifestyle from pairs who don't. That's expected, because the fake prompts reuse a
small vocabulary, which word counts already handle. Real people describe the same hobby in
many different words, which is where MiniLM helps.

**In the nightly refresh:**
- **Time:** the first night takes 1.3 s instead of 0.3 s; later nights about 0.3 s, since
  unchanged prompts are remembered.
- **Effect:** each member's top 10 keeps a median 80% of the same profiles, so MiniLM
  changes about 2 of them.

**Switching it on:** set `[personality] text_encoder = "sentence"` in production. Tests and
test runs keep `"words"`, which needs no model files.

---

## 6. Honest limits

- **The settings were tuned on synthetic photo vectors** (128 numbers; SigLIP gives 768).
  The photo compression and the penalties need rechecking on real vectors, which means a
  small consented pilot (Sprint 14 option (a)). No real faces can be used in this repo.
- **The SigLIP file is a community export** (Xenova) of Google's Apache-2.0 weights. It's
  pinned and fingerprinted; to remove the dependency, export it yourself once from Google's
  official repository, which needs PyTorch on one machine.
- **MiniLM is English only.** Prompts in other languages would match poorly; the multilingual
  MiniLM is a settings change away.
- **The personality weights were set with word counts.** MiniLM gives higher similarities
  overall (unrelated prompts score about 0.1 to 0.2 instead of 0). Rankings within a member's
  list are what matter, but the weights should be retuned on real outcomes (paper 9).
- **Looks models can carry bias.** Image models learn from internet photos and can encode
  stereotypes. Your matching uses each member's own taste on top (rule R2: no attractiveness
  score), but the fairness audits ([#16](https://github.com/currencycaver29/lastfirstdate-matching/issues/16))
  matter even more with real embeddings.

---

## 7. What to study

1. **Embeddings:** how a model turns a photo or a sentence into a list of numbers where
   "close" means "similar".
2. **Cosine similarity:** the angle between two vectors, and why vectors are scaled to
   length 1 first.
3. **Mean pooling with a mask:** averaging token vectors without letting padding in.
4. **Pinning and checksums:** why exact versions and fingerprints protect a system that
   learns from data.

## Try it yourself

```bash
.venv/bin/python -m pytest tests/test_sprint13.py -q
.venv/bin/pip install -e ".[models]"
.venv/bin/python -m lastfirstdate.encoders download
.venv/bin/python -m lastfirstdate.models_report
```

---

## Questions for you

1. **Languages:** will members write prompts only in English? If not, switch to the
   multilingual MiniLM.
2. **A consented pilot:** can you find 30 to 100 volunteers to do the photo exercise before
   launch, so the photo settings can be rechecked on real vectors? (Your lawyer should
   approve the consent wording.)
3. **Sprint 14:**
   - **(a)** the pilot recheck;
   - **(b)** re-run the gate every cycle;
   - **(c)** retune the personality weights on real outcomes;
   - **(d)** wait for the app and real data.

   I'd recommend (a) if a pilot is possible, otherwise (d).

# Sprint 13 results: the real photo and text models

Tests (offline, stand-ins): `python -m pytest tests/test_sprint13.py -q` (9 tests).
Real models, outside the tests (needs `pip install -e ".[models]"` and
`python -m lastfirstdate.encoders download`): `python -m lastfirstdate.models_report`. Made-up
texts, generated images and the synthetic population only (R3).

## Downloads (FR-LM-59)

| File | Size | sha256 |
|---|---:|---|
| SigLIP vision model (ONNX, Xenova export, revision 4649052) | 372 MB | f89d41ba… matches |
| MiniLM (ONNX, revision 1110a24) | 90 MB | 6fd5d72f… matches |
| MiniLM tokenizer | 0.5 MB | be50c362… matches |

A second `download` fetched nothing.

## Text (FR-LM-61)

| Made-up prompts | Word counts | MiniLM |
|---|---:|---:|
| rock climbing and trail runs / bouldering | 0.00 | 0.67 |
| rock climbing and trail runs / opera and a novel | 0.00 | 0.13 |
| cooking Italian food / pasta carbonara | 0.00 | 0.52 |
| cooking Italian food / marathon training | 0.00 | 0.20 |
| something serious, settle down / long-term partner | 0.00 | 0.45 |

Synthetic population (seed 7), man-woman pairs, AUC for sharing a lifestyle: word counts 0.869,
MiniLM 0.888. About 2.6 ms per prompt on this CPU.

Through the app's nightly refresh (Phase 1 zone, 400 members): first night 1.3 s with MiniLM
against 0.3 s with word counts; later nights about 0.3 s, because unchanged prompts are
remembered. Personality rankings agree with the word counts at a median Spearman of 0.87, and
each member's top 10 keeps a median 80% of the same profiles.

## Photos (FR-LM-60)

768 numbers of length 1; the same photo twice is identical; about 0.09 s per photo. A brighter
copy of a generated pattern scores 1.00, and other generated patterns 0.93 on average: SigLIP's
vectors are all fairly alike, which the photo compression handles by centring them first.

## Earlier results

Unchanged: tests and test runs use `text_encoder = "words"` and synthetic embeddings, and
every earlier test passes.

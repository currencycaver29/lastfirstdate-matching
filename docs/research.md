# Research behind the engine

The engine builds on published work in recommender systems and matching markets. Two papers
shaped it most.

| Paper | What it gives the engine | Where |
|---|---|---|
| Pizzato, Rej, Chung, Koprinska, Kay, *RECON: A Reciprocal Recommender for Online Dating*, ACM RecSys 2010 ([dblp](https://dblp.org/rec/conf/recsys/PizzatoRCKK10)) | The idea at the heart of the engine: a dating match must be scored in both directions. The engine scores how much each person would like the other, then combines the two with the harmonic mean, so a pair ranks high only when the interest is mutual. Every daily list is ordered by this mutual score | Stage 5, the mutual score: `mutual.py`, FR-LM-20 |
| Singh, Joachims, *Fairness of Exposure in Rankings*, KDD 2018 ([arXiv](https://arxiv.org/abs/1802.07281)) | Being seen is a resource to share fairly. No one appears on more than a fair share of each day's lists (the exposure cap), and members owed dates get a bonus. This spreads dates across members, keeps anyone from being ranked by overall attractiveness (rule R2), and stopped fake accounts from reading a ranking off their lists (Sprint 14) | Stage 6, the daily lists: `fairness.py`, FR-LM-21, FR-LM-23, decision D7 |

The mutual score, in the engine's terms:

```
S(i -> j) = 0.5 * percentile_i(looks) + 0.5 * percentile_i(personality)
M(i, j)   = 2 * S(i -> j) * S(j -> i) / (S(i -> j) + S(j -> i))
```

The exposure cap: nobody appears on more than `ceil(exposure_multiple * list slots today /
candidates today)` lists in a day, with lists filled one slot at a time in turn.

The rest of the engine uses standard statistics and machine learning (logistic regression with
a prior, Platt scaling, the Brier score, centring of text vectors), explained in plain words in
the sprint notes (`docs/sprints/`).

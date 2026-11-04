# Stage 3: looks taste

Learns what each member finds attractive, as a personal model: never one score for how
attractive someone is in general (R2).

## Takes in / passes on

- **In:** photo embeddings (reduced by PCA to `pca_dim` features), the calibration answers
  (stage 2), and later, date outcomes (stage 8).
- **Out:** `V_i(j)`, member i's predicted chance of liking j's looks, for every candidate.

## The model

    P(i likes j) = sigmoid((w_0[group] + delta_i) · x_j)

- **w_0** is a starting taste per (viewer gender, candidate gender) group, fitted on
  everyone's swipes (`StartingTaste`). It is a prior, never a ranking on its own.
- **delta_i** is the member's own correction, fitted by Bayesian logistic regression
  (`PersonalTaste`).
- The fit runs Newton's method with step halving. The Laplace covariance it returns says how
  sure the model is.
- Lists refuse looks scores until calibration is complete (`CalibrationIncomplete`).

## Code

`src/lastfirstdate/looks.py`: `photo_features`, `PCA`, `fit_logistic` (with a Rust twin),
`StartingTaste`, `PersonalTaste`. Production plugs a real image model (such as CLIP) into the
`PhotoEncoder` interface; tests use synthetic embeddings (R4).

## Settings

`[calibration]`: `pca_dim`, `l2_penalty`, `w0_l2_penalty`.

## Tests

FR-LM-13 to FR-LM-15 (`tests/test_looks.py`), FR-LM-35 (`tests/test_native.py`).

## Papers

- 3, Chu and Ghahramani; 4, Ren et al.; 5, Rothe et al.: a general model plus a personal
  correction.
- 6 and 7, Neve and McConville: photo-based reciprocal matching.

## What we know

- AUC 0.844 against a ceiling of 0.875. A personal model adds about 0.09 AUC over w_0 alone.
- In 9,444 real calibration fits, Newton needed 3 to 6 steps. Step halving prevents runaway
  on hard problems and changes nothing when every full step helps.

## If you change it

Keep `_fit_logistic_python` and the Rust fit agreeing (`tests/test_native.py`). The forecast
(stage 9) and learning (stage 8) use the same fit.

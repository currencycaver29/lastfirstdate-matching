# Sprint 11 results: ready for real data

Command: `python -m pytest tests/test_sprint11.py -q`. The Phase 1 market (seed 7, 15% new women
per cycle) is simulated over three cycles with 80% show-ups instead of the forecast's assumed
92%, and written into SQLite as app records (`app_sim.write_history`).

## Outcomes from records (FR-LM-51)

| Check | Result |
|---|---|
| Cycle outcomes, 291 simulated cycles | all identical when rebuilt from records |
| Weekly delivered-so-far and confirmed-to-come, 838 snapshots | all identical |
| Extra cycles in the records | 6 genuine fourth cycles (men whose cycles started on launch day) |

## Recalibration and the switch (FR-LM-52, FR-LM-53)

| | Share of cycles with 3 delivered dates |
|---|---:|
| Actual | 76.1% |
| Current forecast | 83.3% |
| Recalibrated on all 297 cycles (a = −0.48, b = 0.96) | 76.3% |

Switch rule (train on the earlier 70% of cycles, test on the latest 30%): Brier 0.2179
(current) against 0.2112 (recalibrated), so the switch is recommended.

## Surprise

Split at the median, the recalibration did worse on the later half (16.1 points off against
the current model's 15.0). Fulfilment falls as a zone ages (about 88% in first cycles, 71% in
third cycles), so first-cycle data can't foresee the later level. Recalibrate monthly on recent
cycles, and switch only after testing on the latest cycles.

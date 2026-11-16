# First results — 10 November 2026

## What was tested

- **1,620 completed matches**, 2022-08-05 to 2026-11-08.
- Training: 978 matches, through 2025-01-19.
- Calibration: 331 matches, through 2025-12-21.
- Unseen final test: **311 matches**, 2025-12-22 to 2026-11-08.
- Split by distinct date, approximately 60/20/20. No random shuffling.
- Model parameters/scaling are fit on training data. Temperature is fit only on calibration data. Earlier observed test results update features for later test dates; parameters are frozen.

## Final test results

| Measure | Model | Training-frequency baseline |
|---|---:|---:|
| Correct largest-probability pick | 43.1% (134/311) | 39.5% |
| Log loss, lower is better | 1.0700 | 1.1032 |
| Multiclass Brier sum, lower is better | 0.6453 | 0.6689 |

Always choosing the home team also scored 39.5%. The frequency baseline uses only training-period outcome frequencies and has home win as its largest probability. This is an elementary benchmark, not a bookmaker comparison.

The model shows a modest improvement on this particular split. **No betting edge or profit is established.** No odds, bookmaker margin, stake simulation or live trial is included.

## Weakness visible in this run

The model rarely makes draw its largest probability: 3 of 311 test matches, only 1 correct. Draw probabilities are still present for every match, but this is a weak single-pick draw predictor. The JSON report includes the full H/D/A confusion matrix and per-class reliability bins. Small bins should not be overinterpreted.

One chronological test is not enough to establish stability. A stronger goal/Elo baseline, seasonal walk-forward tests, carefully timed bookmaker prices and an untouched paper trial are next. Tuning repeatedly against this same test set would weaken its status as evidence.

## What the included model means

The saved model retains the evaluated training coefficients and calibration. Its team-state history has been advanced through 2026-11-08. It was not refit on the held-out test results. Its forecasts exclude injuries, lineups, xG and news. It should be treated as a learning baseline, not a production tipping service.

## Verification

12 automated tests passed. They check current/future label leakage, same-day feature freezing, coefficient/scaler/calibration isolation from test labels, JSON inference parity, probability/class ordering, input validation, forecast guardrails and paired journal scoring. Synthetic test data exercises correctness; the figures above use actual downloaded match results.

The CLI training command, team-list command and labelled forecast demo were also run. No web app or API has been built or tested yet.

# v0.2 measured results — 10 November 2026

1,620 completed matches, 5 August 2022 to 8 November 2026. All have shots/on-target statistics. **Possession coverage: zero**, so no possession benefit is claimed.

## Same-fixture candidate comparison

| Candidate | Development accuracy (760 matches) | Development log loss | Final-season audit accuracy (100 matches) | Audit log loss |
|---|---:|---:|---:|---:|
| Frequency baseline | 41.7% | 1.0860 | 42.0% | 1.0854 |
| Logistic regression | 51.3% | 1.0265 | 44.0% | 1.0453 |
| Poisson goal model | 52.4% | 1.0177 | 45.0% | 1.0482 |
| XGBoost | 50.5% | 1.0235 | 42.0% | 1.0422 |

Development consists of 2024/25 and 2025/26. The audit is the partial 2026/27 season. Lower log loss is the selection criterion. The Poisson goal model wins on development loss. XGBoost has slightly better audit loss, but that does not override the development-only choice. Accuracy and probability quality are different measures.

The selected goal model picked 45 of 100 audit results correctly; the home/frequency baseline picked 42. It never selected draw as its highest probability in that audit. Its draw probabilities exist, but draw picking is weak. Small model differences over 100 matches do not establish reliable superiority, profitability or a bookmaker edge.

## Evaluation boundaries

- 2024/25 (development): fit 608 matches through 2024-02-05; calibrate 152 through 2024-05-19; test 380 from 2024-08-16 to 2025-05-25.
- 2025/26 (development): fit 908 matches through 2024-12-08; calibrate 232 through 2025-05-25; test 380 from 2025-08-15 to 2026-05-24.
- 2026/27 (audit): fit 1219 matches through 2025-10-19; calibrate 301 through 2026-05-24; test 100 from 2026-08-21 to 2026-11-08.

Parameters are frozen within each historical test season; earlier test outcomes update later team features. Temperature calibration uses only the preceding calibration dates. Candidate settings were fixed before inspecting the v0.2 audit, with no tuning to improve that audit.

These dates overlap data already viewed in v0.1. This is exploratory historical evaluation, not a pristine external holdout. No bookmaker prices or ROI have been evaluated.

## What is saved for future use

`artifacts/model.json` is the selected goal model refit on newer history: coefficients through 21 December 2025, calibration and team history through 8 November 2026. It is labelled as a latest refit for a future paper trial, with no independent out-of-sample result. Do not attach the frozen audit accuracy to this latest refit.

`artifacts/audit_models/` preserves the fixed final-audit parameter sets. `artifacts/walk_forward_predictions.csv` contains every evaluated prediction, model and role. `artifacts/benchmark.json` includes coverage, provenance, dates, metrics and caveats.

## Validation

23 tests passed, including future-result/stat leakage, delayed possession availability, prediction-time cutoff, same-day feature freezing, JSON model roundtrip, H/D/A class order, Poisson probability math, chronological boundaries and audit-label isolation from development selection. The XGBoost roundtrip was exercised in this environment.

The comparison, data-check, team listing, forecast demo and journal scoring commands were run. No web app or API is part of this release.

# v0.3 results — 2015-onward research

Run date: 10 November 2026. **The 55–60% all-match target has not been verified.** More history did not improve the selected model's latest-season hit rate.

## Data and scope

- 4,280 completed results: 8 August 2015 to 8 November 2026.
- 11 complete seasons of 380 matches, plus the 100 completed 2026/27 matches.
- 760 warm-up-season matches; 3,420 historical development predictions; 100 final-season audit predictions.
- 59 active prematch features: goals, points, Elo, venue form, shots, longer form, discipline, corners and fouls.
- 4,279 fully usable shot rows; all 4,280 discipline rows; no results dropped.
- Possession observations: zero. Verified paired injury/suspension snapshots: zero. No injury benefit is claimed.

One source row, Newcastle–West Ham on 15 August 2021, records away total shots 8 and away shots on target 9. We retained the outcome, marked those two away shot values missing under the explicit quarantine option, and retained original source values/hashes in provenance. No correction was guessed.

Source season identity preserves July 2020 within 2019/20. A calendar July split would mislabel those COVID-delayed fixtures.

## Candidate comparison

The recent selection window is 2023/24–2025/26 (1,140 matches). The latest partial season is excluded from model choice. Lower selection log loss is the prespecified criterion.

| Candidate | All development accuracy (3,420) | Recent selection accuracy (1,140) | Recent selection log loss | Latest audit accuracy (100) | Audit log loss |
|---|---:|---:|---:|---:|---:|
| Frequency baseline | 44.1% | 43.2% | 1.0729 | 42.0% | 1.0846 |
| Logistic regression | 51.8% | 53.1% | 0.9918 | 42.0% | 1.0425 |
| Poisson goals | 53.5% | 54.4% | 0.9807 | 41.0% | 1.0356 |
| XGBoost | 52.6% | 52.4% | 0.9954 | 45.0% | 1.0426 |
| Recent-weighted logistic | 51.8% | 53.2% | 0.9914 | 43.0% | 1.0385 |
| Recent-weighted Poisson | 53.7% | 54.2% | 0.9797 | 40.0% | 1.0343 |
| Recent-weighted XGBoost | 53.1% | 52.6% | 0.9922 | 44.0% | 1.0438 |

The development-only rule selected **recent-weighted Poisson**, not the candidate with the best audit accuracy. Its 54.2% recent-selection result is not a measured 55% or 60% all-match result. It picked 40/100 latest audit outcomes correctly; frequency/home picked 42/100, and the stored bookmaker-favourite comparator picked 48/100. No bookmaker edge is established.

In v0.2 the selected frozen goal model picked 45/100 on this same latest-season sample. v0.3 improves its audit log loss from approximately 1.0482 to 1.0343, but its top-pick accuracy falls. Better probability loss and better pick accuracy are different goals. The previous release is preserved; do not call this an unqualified accuracy improvement.

## Selected candidate, every held-out season

| Season | Role | Test matches | Accuracy | Log loss |
|---|---|---:|---:|---:|
| 2017/18 | development | 380 | 53.4% | 0.9714 |
| 2018/19 | development | 380 | 57.4% | 0.9235 |
| 2019/20 | development | 380 | 52.9% | 0.9889 |
| 2020/21 | development | 380 | 49.7% | 1.0246 |
| 2021/22 | development | 380 | 53.4% | 0.9727 |
| 2022/23 | development | 380 | 53.4% | 0.9833 |
| 2023/24 | development | 380 | 59.7% | 0.9194 |
| 2024/25 | development | 380 | 54.5% | 0.9902 |
| 2025/26 | development | 380 | 48.4% | 1.0294 |
| 2026/27 | audit | 100 | 40.0% | 1.0343 |

The variation across seasons matters: an older season near 60% does not establish that rate today. All these historical periods have been inspected during development. The current audit was already seen in older versions; it is not a pristine external holdout.

## Method

- The first two source seasons warm up history.
- Per test season, earlier matches fit parameters; about the last 190 prior matches calibrate probabilities, with complete date groups kept together.
- No fitting/calibration row is later than the test season's first fixture. Parameters stay frozen for that season; completed earlier dates update later team features.
- Recent-weighted candidates retain the 2015 history but give fitting rows a three-year weight half-life, normalized to mean weight one.
- Select by pooled log loss on up to the latest three completed development seasons. Do not select by the audit's winning model.
- The latest paper refit fits parameters through 4 March 2026 and calibrates/updates history through 8 November 2026. It has no independent future-trial result yet.

## Confidence subsets are not the headline score

Fixed probability thresholds 0.50, 0.60 and 0.70 were declared before the proper run. Subsets report their own coverage and Wilson uncertainty intervals in benchmark.json. On this small audit, only 16, 7 and 2 matches qualify respectively. Even a very high hit rate in those tiny subsets is not the all-match accuracy and is not evidence of a stable betting system. The all-match audit result remains 40/100.

## What was verified

36 automated tests passed with XGBoost installed. Tests cover current/future outcome/stat leakage, prior red/yellow visibility, bad-form streaks, COVID season identity, raw-preserving quarantine, chronological split boundaries, development-selection isolation from audit labels, delayed availability timestamps, unknown versus healthy states, rejection of retrospective aggregates, rejection of injury-aware calls on untrained models, JSON model roundtrips, recency candidates, Poisson probability math, cache-only downloads and confidence coverage.

The package install, cached downloader, data-check, complete 12-season comparison, demo and journal scoring were checked. No API or app is included.

## Remaining blocker

An injury/suspension import is implemented but no verified historical pre-match feed was connected. See INJURY_DATA_AUDIT.md. Until reliable snapshots are supplied and the injury-feature model is retrained/evaluated on paired fixtures, this remains a form/strength/discipline model.

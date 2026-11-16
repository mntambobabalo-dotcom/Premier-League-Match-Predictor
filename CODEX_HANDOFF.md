# Codex handoff for Premier League Markets v0.4

## Paste this task into Codex

Read AGENTS.md and docs/MARKETS_V0_4.md. Help me run `python -m epl_predictor markets --demo`, then show me how to enter a verified fixture and its Betway decimal prices. Continue with market-level probabilities for double chance, straight outcomes, goals, correct scores and recorded cards. Use the 2015/16 onward history and evaluate quoted prices with dated paper tests. Card prices need a verified match between archive statistics and settlement rules. Work through one change at a time.

## Current market workflow

- `markets --demo` runs the bundled model and displays City–United head-to-head plus the market probabilities.
- `benchmark-markets --data-dir data --quarantine-invalid-shots` rebuilds the models and price research.
- The selected draw cover hit 76.8% on 3,520 historical fixtures. Historical priced simulations returned −1.6% on 1X2 and −2.7% on over/under 2.5 goals.
- Recorded yellow and red probabilities underperformed their simple historical baselines. The quote checker accepts goal and outcome markets, with card settlement matching still pending.
- Current Betway quotes are entered manually. The archived price study uses Bet365 1X2 and totals plus earlier archive-average totals for probability comparisons.
- Injury coverage remains zero in the market model.

## Preserved v0.3 outcome workflow


- Python package v0.3.0, command: python -m epl_predictor.
- 4,280 completed results, August 2015 through 8 November 2026.
- 12 source CSVs are acquired by the explicit cached download-data command; no raw provider data is bundled.
- 4,279 complete shot rows, all 4,280 discipline rows, zero possession rows, zero paired pre-match injury/suspension rows.
- Newcastle–West Ham, 15 August 2021: AS=8 and AST=9. The explicit quarantine option sets just that away shot pair missing, retaining the result and provenance.
- COVID season identity is based on source seasons, keeping July 2020 in 2019/20.
- 59 active prematch features: Elo, recent/longer form, absolute attack/defence, venue scoring, shots, cards, fouls and corners.
- Seven fixed candidates: frequency; logistic/Poisson/XGBoost; and three-year recency-weighted versions of those models.
- About 190 preceding matches calibrate each fit; previous dates train. Test parameters stay frozen per season.
- Select on pooled log loss over 2023/24, 2024/25 and 2025/26. The 2026/27 audit is excluded from selection but was already viewed in previous versions, so it is not pristine evidence.
- Selected recent_poisson: 53.65% over 3,420 historical development predictions; 54.21% on the 1,140-match selection window; 40% on the 100-match latest audit. The recorded bookmaker favourite benchmark is 48% on those 100 matches, with unaudited exact snapshot timing.
- The latest refit is not independently measured. Its coefficients fit through 4 March 2026; calibration/team state through 8 November 2026.
- Prior release artifacts are preserved. More history did not improve latest-season top-pick accuracy; do not label this an unqualified accuracy upgrade.
- 36 automated tests passed with XGBoost installed.

## Injury integration is the real remaining data blocker

availability.py accepts fixture-specific snapshots, not retrospective season totals. Columns: Date, Team, Injured, Suspended, KnownAt, Source; optional KeyMissing. Source and KnownAt must identify when the actual supplied counts were available. The conservative date-only pipeline uses snapshots before UK matchday midnight, further bounded by forecast creation time.

No actual historical injury dataset has been integrated. Candidate public retrospective datasets lack the required evidence of when injuries were known; the referenced API-Football injury endpoint documents data only from April 2021 and requires an account/key. Provider access or a verified export must be established before claiming this model responds to injuries. Do not paste keys into source; use local environment variables if a provider is later authorized.

For a new feed: establish rights and coverage; document source IDs, dates and timestamp semantics; map team names explicitly; audit unknown/missing versus healthy; test historical eligibility; retrain and run a paired feature-ablation comparison on identical fixtures. An injury-aware forecast call is deliberately rejected when the model has no historical availability training coverage.

## Commands

Use START_HERE.md for exact Windows/macOS commands. Main commands are demo, download-data, data-check, benchmark, predict, teams and score. After retraining into artifacts/my_run, pass --model artifacts/my_run/model.json to use that run. Training does not make hidden network calls.

## What not to do

Do not use current-match goals, shots, cards, later return dates or eventual season injury totals in prematch features. Do not infer a named player's suspension from team card totals. Do not claim a small high-confidence subset is the full-fixture hit rate. Do not add bookmaker odds as model features without a separate explicit design and forecast-time data audit. Do not claim profits, guaranteed percentages or independent live validation.

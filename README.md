# Premier League Markets — Python v0.4

The Python market research engine for SK uses Premier League history from **2015/16 onward**. Version 0.4 adds market probabilities, head-to-head results and a decimal odds comparison.

Run `python -m epl_predictor markets --demo` to see scorelines, total goals, double chance, recorded cards and the recent head-to-head. Pass `--price X2=1.60` to compare an entered decimal quote with the model. [Market guide and historical results](docs/MARKETS_V0_4.md)

The latest [market results check](docs/MARKET_RESULTS_CHECK.md) includes under 3.5, BTTS, filtered draw cover, chronological calibration comparisons and paired parlay tests. Run `python scripts/test_market_policies.py` to reproduce the saved-forecast experiment without downloading data. The project currently has 53 passing tests; no profitable betting system has been established.

## Previous outcome research in v0.3

**The requested 55–60% all-match accuracy has not been verified.** The selected model reaches 54.2% over the three latest completed development seasons but 40% on the current season's 100-match audit. Historical injury/suspension coverage is zero. The import/inference safeguards are built, but the supplied model is not injury-aware. These limitations are explicit in the reports and code.

Open **START_HERE.md** for terminal commands and **CODEX_HANDOFF.md** before continuing in Codex. Read **docs/V0_3_RESULTS.md** for all results and **docs/INJURY_DATA_AUDIT.md** for the unresolved data requirement.

## What is included

- 12 season sources, 4,280 completed matches, 8 August 2015 to 8 November 2026; raw provider files are downloaded explicitly rather than bundled.
- Earlier match outcomes, goals for/against, home/away venue performance, Elo strength, shots and shots on target.
- Five-game form plus 10/20-game points and goals, clean sheets, draw rates, trends, winless/loss streaks and absolute team strength.
- Earlier yellow/red cards, previous-game reds, corners and fouls. These are past discipline indicators, not known player suspensions.
- Logistic regression, Poisson goal regression and XGBoost, plus recent-weighted versions and a frequency benchmark.
- Timestamped possession and pre-match injury/suspension imports, with unknown versus observed-zero flags.
- A separate human probability/notes journal and conservative paired scoring.
- Preserved v0.1 and v0.2 artifacts, a current paper model, every historical prediction and model-selection metadata.
- 36 passing tests and an explicit cached downloader.

Pep joined City in July 2016. Starting at 2015/16 deliberately includes the preceding season rather than moving the user's requested start forward.

## First local run

Follow START_HERE.md to install the project and optional XGBoost extra. Then:

```sh
python -m epl_predictor demo
python -m epl_predictor teams
python -m unittest discover -s tests -v
```

The demo chooses an illustrative matchup and labels it clearly; it does not claim a scheduled fixture exists and never counts toward paper results. A pretrained current paper model is included, so the first demo needs no API key or raw CSVs.

## Download and validate the research history

```sh
python -m epl_predictor download-data --start-season 2015 --end-season 2026 --out data
python -m epl_predictor data-check --data-dir data --quarantine-invalid-shots
```

This downloader is an explicit user command. Training/prediction/tests never download data. It checks/reuses existing season files and records URLs and SHA-256 hashes in data/download_manifest.json. There is no scheduler or hidden automatic refresh. To update a cached current-season file, remove that one file deliberately and rerun the download command, respecting the provider's usage limits. Completed seasons do not need repeated downloads.

Use source data only for permitted private research until commercial rights are established. See [Football-Data's usage policy](https://football-data.co.uk/contact.php). No paid provider account was created.

The files have names such as 1516.csv, 1617.csv and 2627.csv. You can instead download manually from [Football-Data England](https://www.football-data.co.uk/englandm.php). Required columns are Date, HomeTeam, AwayTeam, FTHG and FTAG. FTR, when supplied, must agree with the goals. Div, when supplied, must be E0. Optional statistics and odds remain missing when unknown.

Source filename or explicit Season identifies each competition season, preserving July 2020 matches inside 2019/20. Generic combined CSVs should include Season as the starting year, e.g. 2019. Dates accept ISO or the provider's day-first format. Unknown team aliases are not automatically resolved.

### The explicit quality exception

Newcastle–West Ham on 15 August 2021 has source AS=8 and AST=9, an impossible pair. We do not guess a correction. With --quarantine-invalid-shots, the away shot pair is marked missing, the result stays, and the original values plus hash remain in provenance. Without that flag, inconsistent shots fail validation. The raw file is never edited. All 4,280 results remain; 4,279 rows have complete usable shot statistics. All 4,280 have discipline statistics.

## Reproduce the model comparison

```sh
python -m epl_predictor benchmark --data-dir data --quarantine-invalid-shots --out artifacts/my_run
```

Use --model artifacts/my_run/model.json to forecast with that new output. Use --skip-xgboost only when deliberately omitting both XGBoost candidates; no silent substitute is used.

The first two seasons warm up training. Each later season is predicted using only earlier dates. About 190 earlier matches, with same-day groups kept together, form the calibration window; earlier matches fit parameters. Every test-season parameter set stays frozen, while prior completed results update team state.

Selection uses the lowest pooled log loss over up to the three most recent completed development seasons. The current partial season is excluded from selection. Recent-weighted candidates retain the old data but apply a three-year half-life to fitting weights. Historical results in this project have already been inspected; this is exploratory research rather than an untouched external holdout.

The latest paper model is a separate refit using more recent fitting/calibration history. Do not assign it the frozen audit model's achieved score. It needs its own future paper trial.

| Output | Meaning |
|---|---|
| model.json | Latest refit for future paper forecasts; not independently tested |
| benchmark.json | Coverage, folds, candidates, selection, metrics and caveats |
| walk_forward_predictions.csv | Every historical prediction and actual result |
| audit_models/ | Frozen latest-audit parameter sets, with current team state |
| artifacts/v0_2/ | Preserved previous release artifacts |
| docs/EXPERIMENT_PLAN_V0_3.json | Candidate settings and evaluation policy |
| docs/DATA_COVERAGE_V0_3.json | Source hashes and the recorded shot-stat exception |

A historical pre-closing bookmaker comparator is also reported. It never enters model features or selection. Exact snapshot availability relative to a specific forecast is unaudited, so it is not a live pricing comparison or an ROI claim.

## Injury/suspension integration

**The included trained model has no verified historical injury data.** Supplying current injuries alone cannot teach it the missing relationship. It refuses an injury-aware forecast if it was not trained on historical availability.

Use examples/availability_template.csv for verified fixture-specific snapshots. Required columns are Date, Team, Injured, Suspended, KnownAt and Source. Optional KeyMissing is a consistently defined count of important absent players; it is not guessed from names or reputation. Counts must be nonnegative integers. Unknown data is not assumed to mean healthy or available.

KnownAt must be the timestamp when the exact supplied values were known, with an explicit timezone. It cannot be fabricated from the injury start date, a future return date or the time a retrospective dataset was scraped. Multiple revisions are allowed; the latest eligible snapshot is selected. The date-only pipeline conservatively requires snapshots before UK midnight starting matchday, and live forecasts also restrict knowledge to the moment the forecast is made. Same-day team news/lineups need a future verified-kickoff-time upgrade.

```sh
python -m epl_predictor benchmark --data-dir data --quarantine-invalid-shots --availability-csv inputs/availability.csv --out artifacts/with_availability
```

After training with genuine snapshots, a future forecast can use --availability-csv with fixture-specific current snapshots. The engine verifies that the model has historical availability coverage. The injury benefit must be compared against an otherwise identical model on the same fixtures, not assumed from feature presence.

Possession import remains optional through --possession-csv. See examples/possession_template.csv. Its KnownAt-equivalent field is AvailableAt; counts are never fabricated. Included possession coverage is also zero.

## Paper forecasts and your football knowledge

Use a verified future fixture and exact names from teams:

```sh
python -m epl_predictor predict --home "Man City" --away Arsenal --date YYYY-MM-DD --human 0.50,0.28,0.22 --note "Your actual independent reasoning"
```

Replace YYYY-MM-DD with the verified fixture date. H/D/A are home win, draw and away win. The three human values must sum to one. Write your own view before viewing the machine forecast to reduce anchoring. The journal preserves model and human probabilities separately; it applies no arbitrary emotional slider.

History must end before the requested date. Past dates or history more than 21 days behind the fixture require an explicit --demo. The engine does not verify fixture existence or confirm team news automatically.

After results settle:

```sh
python -m epl_predictor score --records predictions --data-dir data --quarantine-invalid-shots
```

Scoring compares human and model on identical paired fixtures. It excludes demos, unresolved results, later duplicates and same-day records. Until kickoff timestamps are added, create real records the day before. Local editable records are not independent verified performance evidence.

## Interpretation and limitations

Accuracy is the hit rate of the largest-probability pick. Log loss evaluates all three probabilities; Brier evaluates squared probability error. Lower loss is better, but it can improve while top-pick accuracy falls. That happened on the latest season here.

Confidence subsets have their own sample count, coverage and uncertainty intervals. They do not replace all-match accuracy. A tiny subset with a high hit rate is not proof of 55–60% performance or profitability.

No betting, staking, bankroll or automated wagering is implemented. No bookmaker edge is established. The independent Poisson model is deliberately simple, not Dixon-Coles. Team rest is league-only. Injuries, player bans, lineups, xG and possession are not secretly present when coverage is zero.

## Source modules

- data.py: strict match/stat ingestion, source seasons and explicit quality quarantine.
- features.py: features from earlier matches and eligible optional snapshots.
- availability.py: injury/suspension snapshot validation and as-of selection.
- research.py: seven candidate fits, seasonal testing, calibration and development-only selection.
- model.py: safe JSON inference and metrics; legacy single-holdout experiment.
- predict.py: live knowledge cutoff, independent human view and journal.
- downloads.py: explicit cached source acquisition.
- cli.py: terminal commands, including labelled demo.

The original train command remains available and writes to artifacts/legacy_holdout by default. It is not the expanded v0.3 experiment. Use benchmark for the 2015-onward research workflow.

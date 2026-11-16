# Saved market results check — 16 November 2026

The saved arithmetic reconciles. The historical price filter has not demonstrated an edge. Its selected probabilities deserve calibration work before stronger model claims.

Reproduce without downloading sources or changing the model:

```powershell
.\.venv\Scripts\python.exe scripts/check_market_results.py --out artifacts/markets/checks.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The checker independently reconstructs events from scores, verifies probability sums, double-chance identities, exact-score probabilities and settlements, reconciles headline and season report metrics, joins paper selections to forecasts, and checks their probabilities, settlements, payouts and filter eligibility. It adds fixed-bin calibration diagnostics, subset coverage and Wilson intervals to `checks.json`. It does not refit or calibrate the model.

## Hits on every evaluated fixture

| Selection rule | Historical hits / 3,520 | Hit rate (approximate 95% interval) | Latest 100 hits |
|---|---:|---:|---:|
| Highest-probability straight outcome | 1,877 | 53.3% (51.7–55.0%) | 42 |
| Higher probability between 1X and X2 | 2,703 | 76.8% (75.4–78.2%) | 69 |
| Higher probability over/under 2.5 goals | 1,980 | 56.3% (54.6–57.9%) | 51 |
| Modal exact score | 417 | 11.8% (10.8–13.0%) | 11 |

All rows cover 100% of their stated population. Latest straight outcomes hit 42% (32.8–51.8% interval), draw cover 69% (59.4–77.2%). Wilson intervals assume independent trials; common teams/dates may increase uncertainty. Draw cover includes two outcomes, so its hit rate is not directly comparable with straight-outcome accuracy.

The three recent completed development seasons, 2023/24–2025/26, contain 1,140 predictions: straight outcomes hit 611/1,140 (53.6%), draw cover 890/1,140 (78.1%). The latest partial 2026/27 season is excluded from that window. These seasons have already been inspected and are exploratory evidence.

At the existing 65% confidence threshold, straight outcomes hit 505/683 historically (73.9%; 19.4% fixture coverage; interval 70.5–77.1%). The latest subset hit 9/9 (100%; 9% coverage; interval 70.1–100%). Neither subset is the all-match result. No threshold search was performed.

## Quoted-price selections

| Archived price family | Fixtures with odds / 3,520 | Selections | Hits | Net units | Return per unit staked |
|---|---:|---:|---:|---:|---:|
| Bet365 1X2 | 3,520 (100%) | 594 | 151 (25.4%) | −9.49 | −1.60% |
| Bet365 over/under 2.5 | 2,759 (78.4%) | 1,229 | 526 (42.8%) | −33.50 | −2.73% |

Selections cover 16.9% and 34.9% of all fixtures respectively, or 16.9% and 44.5% of fixtures with that family's odds. One unit per selection; at most one selection per fixture per family. Hit-rate intervals are 22.1–29.1% and 40.1–45.6%. Prices vary, so hit rate alone does not determine return.

Latest samples have full quoted coverage of 100/100 fixtures: 1X2 selected 15, hit 4/15 (26.7%), returned +2.67 units (+17.8%); totals selected 49, hit 26/49 (53.1%), returned +11.20 units (+22.9%). These small historical samples do not overturn the longer losses or establish future profit.

## Calibration findings

The selected 1X2 probabilities averaged 30.0%, versus 25.4% realized hits. Selected totals probabilities averaged 50.4%, versus 42.8% hits. Estimated average returns were +18.2% and +14.4%, while realized returns were negative. This is an observed discrepancy, not proof of its cause. Price-based selection can expose weaknesses that pooled all-fixture calibration conceals.

Across all fixtures, over 2.5 averaged 53.5% versus 54.2% observed, Brier 0.2444; BTTS averaged 52.3% versus 52.9%, Brier 0.2468. Latest over 2.5 averaged 52.8% versus 47.0%, and BTTS 54.1% versus 49.0%. The small latest sample limits conclusions about a persistent change. `checks.json` contains bin counts and errors rather than only global averages.

Recorded yellows over 3.5 averaged 44.2% versus 47.1% observed. Over the recent completed development window the gap was larger: 48.1% versus 58.5%. All-history yellow Brier was 0.2495 versus the report's preceding-history baseline 0.2488; any-red Brier was 0.1005 versus baseline 0.0974. Lower is better. Both card models underperformed those simple baselines. Baseline values are retained from the original report because the exported forecasts omit their per-row inputs.

## Focused next experiments

1. **Test chronological calibration first.** The current market benchmark combines `fold['fit']` and `fold['cal']` to fit its heads; it does not fit a separate calibrator. Compare raw and calibrated candidates with identical earlier parameter-fitting data, a later date-grouped calibration window, and identical test fixtures. Keep outcome, double-chance and score probabilities coherent. Select only on the three recent completed development seasons; exclude the latest season and freeze settings before a genuinely future paper trial.
2. **Test the head-to-head contribution.** Compare the same model with and without the four head-to-head features on identical fixtures and timing. Report calibration, goal/score errors and the same fixed price filter. Presence of a football feature is not evidence of benefit.
3. **Require card models to beat their baselines.** Compare recent-history baseline and model on the same known-card fixtures before treating card probabilities as improved. Keep card price checks blocked until settlement equivalence is verified.

## Limits of this check

Raw `data/` CSVs are absent in this workspace. Original source hashes, timing, shot quarantine, complete odds coverage, normalized-odds benchmarks, omitted eligible selections and highest-return selection choice cannot be independently rechecked from the exported artifacts alone. Odds coverage above is reported coverage, not a new source audit. The checker verifies eligibility and payout of selections actually saved.

Archived quote timestamps remain unaudited. Historical double-chance, exact-score, BTTS and card prices are absent, so those markets have no measured paper returns here. Current Betway prices are entered manually. Injury/suspension training coverage remains zero. The latest paper refit remains distinct from the models that generated these historical predictions.

Sources: `artifacts/markets/report.json`, `forecasts.csv`, `paper_price_checks.csv`; evaluation implementation in `epl_predictor/markets.py`. No new data acquisition or future paper trial occurred in this check.

<!-- market-trial-update -->

## Goal markets and parlay experiment — 16 November 2026

### Executive Summary

Filtered draw cover is the strongest candidate for further paper testing in this experiment. A 75% minimum predicted probability produced 470/547 development hits (85.9%; 48.0% fixture coverage), and 27/31 latest-season hits (87.1%; 31% coverage). The latest interval is wide: 71.1–94.9%. This does not establish profitability: archived prices for these markets are absent.

Forced under 3.5 and BTTS picks remain unreliable. The latest 75%+ under subset hit 4/8, and the highest-probability BTTS side hit 48/100 when every fixture required a pick. Combining markets from the same game can narrow the winning scorelines sharply.

The original forecast probabilities beat the tested rate-scaling calibration on development Brier. No change was promoted to the production model. This is an exploratory saved-forecast experiment, not a raw-data refit or an untouched future test. [4]

### Baseline and calibration comparison

There are 3,140 comparable fixtures from 2018/19 through 8 November 2026. The first 380 saved forecasts warm up the prior-only comparisons and are excluded from all three candidates. Selection uses 2023/24–2025/26 only (1,140 fixtures); the latest 2026/27 sample (100) is excluded.

| Candidate | Development mean Brier | Latest mean Brier |
|---|---:|---:|
| raw | 0.2179 | 0.2233 |
| rate_scaled | 0.2191 | 0.2231 |
| prior_rate | 0.2382 | 0.2328 |

Lower Brier is better. This fixed metric averages under 3.5, BTTS yes, 1X and X2 probability errors. Raw probabilities were selected. Calibration multiplies home/away goal rates by observed-to-predicted goal ratios on about 190 prior out-of-season forecasts (whole boundary dates retained; scales capped at 0.5–2). The baseline uses event frequencies from those same prior dates. Parameters freeze before each test season. This evaluates a forecast postprocessor; it does not reserve a fresh calibration set inside the original goal-head fitting process.

| Event | Raw development Brier | Prior-rate development Brier | Raw latest Brier |
|---|---:|---:|---:|
| under_3.5 | 0.2295 | 0.2373 | 0.2162 |
| btts_yes | 0.2447 | 0.2472 | 0.2543 |
| 1X | 0.1876 | 0.2220 | 0.1968 |
| X2 | 0.2098 | 0.2462 | 0.2260 |

### When the rules hit and when they skip

All thresholds (65%, 75%, 80%) were fixed for comparison; none was optimized. The 75% rows below illustrate the common threshold, not a claim that it is optimal. Each draw-cover pick chooses between 1X and X2; BTTS chooses yes/no. Under 3.5 remains a one-sided selection.

| Rule / population | Hits / selections | Fixture coverage | Hit rate | Approximate 95% interval |
|---|---:|---:|---:|---:|
| Development: Under 3.5, every fixture | 736/1140 | 100.0% | 64.6% | 61.7%–67.3% |
| Development: BTTS higher-probability side, every fixture | 637/1140 | 100.0% | 55.9% | 53.0%–58.7% |
| Development: Draw cover, every fixture | 890/1140 | 100.0% | 78.1% | 75.6%–80.4% |
| Development: Under 3.5, 75%+ | 36/52 | 4.6% | 69.2% | 55.7%–80.1% |
| Development: Draw cover, 75%+ | 470/547 | 48.0% | 85.9% | 82.8%–88.6% |
| Latest: Under 3.5, every fixture | 69/100 | 100.0% | 69.0% | 59.4%–77.2% |
| Latest: BTTS higher-probability side, every fixture | 48/100 | 100.0% | 48.0% | 38.5%–57.7% |
| Latest: Draw cover, every fixture | 69/100 | 100.0% | 69.0% | 59.4%–77.2% |
| Latest: Under 3.5, 75%+ | 4/8 | 8.0% | 50.0% | 21.5%–78.5% |
| Latest: Draw cover, 75%+ | 27/31 | 31.0% | 87.1% | 71.1%–94.9% |

There were only four development BTTS selections at 65%+ (3 hits; 0.35% coverage; interval 30.1–95.4%) and none in the latest season. At 75%+ there were none in either period. A no-pick output is appropriate when the rule finds nothing; a large parlay is not filled with lower-confidence substitutes.

### What is associated with four or more goals?

These are descriptive associations, not established causes. Scores are used only to settle outcomes; every grouping uses earlier results or prematch expected rates. Overlapping segments must not be added together.

| Prematch grouping, all 3,140 comparable fixtures | 4+ goals / fixtures | Over 3.5 rate | Under 3.5 rate |
|---|---:|---:|---:|
| Expected total below 2.5 | 82/366 | 22.4% | 77.6% |
| Expected total 2.5 to below 3 | 629/2028 | 31.0% | 69.0% |
| Expected total at least 3 | 298/746 | 39.9% | 60.1% |
| Both teams' prior five games averaged at least 3 total goals | 224/664 | 33.7% | 66.3% |
| At least one prior-five average below 3 | 766/2407 | 31.8% | 68.2% |

Prior-five form is known for 3071/3,140 fixtures; incomplete histories stay missing. Simple recent scoring averages separated risk only weakly. The strongest separation in these predeclared groups came from expected total goals, but it did not reliably order the tiny latest-season groups: 3/7 low-expected-total games and 4/15 high-expected-total games went over. That limits using the historical association as a dependable rule.

Of 1009 actual 4+ goal games, 853 had both teams score and 156 had a clean sheet. In the latest sample it was 28 and 3 out of 31 respectively. These are descriptions after settlement: actual BTTS must never become a prematch input. A weak opponent can still lose 4–0; a presumed one-sided match does not guarantee under 3.5.

Recent defensive averages also provide no established universal explanation: in development, the combined prior-five concession rate at least 3 group went over 217/576 (37.7%), versus 185/549 (33.7%) below 3. The differences are exploratory and can reflect team strength and other confounding factors. Goal timings, red-card timings, lineups and verified historical availability are absent, so this experiment cannot attribute individual overs to them.

### Parlay size: compare the same dates

Each date supplies ranked selections with 75%+ model probability, using one draw-cover pick per fixture and no repeated teams. Only dates with four eligible selections enter this paired comparison; smaller slips use the top part of the same four. These are observed joint wins, not assumed-independent simulation outcomes.

| Slip size | Wins / 38 development dates | Hit rate (95% interval) | Wins / 108 historical dates |
|---|---:|---:|---:|
| 1 | 34/38 | 89.5% (75.9%–95.8%) | 94/108 |
| 2 | 29/38 | 76.3% (60.8%–87.0%) | 80/108 |
| 3 | 25/38 | 65.8% (49.9%–78.8%) | 65/108 |
| 4 | 18/38 | 47.4% (32.5%–62.7%) | 51/108 |

Those dates cover only 11.1% of development fixture dates and 11.1% of all comparable fixture dates. The latest season supplies only one such date, and its slips lost at every size; that sample is too small for useful ranking. Slip sizes are nested and dependent. More legs reduced the measured chance of all selections winning. This cannot identify the most profitable size without actual offered prices. All four markets have zero historical price coverage here; paper returns are unavailable. Other thresholds, mixed policies and all eligible-date results remain in the JSON rather than being presented as winners.

A useful next paper protocol is to compare eligible singles with eligible doubles, preserve no-pick days, record actual prices and settle every selection. Three- and four-leg slips are not justified as more reliable by these results. No real-money staking system or automated wager was added.

### Your Man City–Fulham example

The fixture is listed for 21 November 2026 by [Fulham's official ticket notice](https://www.fulhamfc.com/news/2026/october/13/man-city-away-tickets/). The current paper refit, with result history through 8 November, gives 65.1% under 3.5, 55.5% BTTS yes and 80.0% City-or-draw. These are unverified model estimates, not the historical filtered model's achieved hit rate. The first two fall below the illustrative 75% threshold; the draw cover passes that probability threshold, while a price check is still missing.

Combining all three in the same match permits only 1–1 or 2–1. Their model probabilities sum to 20.2%, so this is a narrow scoreline combination. Its probability must not be calculated by multiplying the three marginal probabilities. Betway's [Build a Bet explanation](https://www.betway.co.za/betway-pedia/documents/betway-features/page/how-does-build-a-bet-work) confirms that all selections must win. A same-game builder needs its own quoted combined price.

The user subsequently supplied Betway decimal prices. These are manually reported prices with an unknown provider observation timestamp; the local record preserves the time they were entered. The original forecast creation time and model hash are preserved.

| Market | Entered odds | Model probability | Break-even probability | Model-estimated return per unit |
|---|---:|---:|---:|---:|
| under_3.5 | 1.60 | 65.1% | 62.5% | +4.1% |
| btts_yes | 1.72 | 55.5% | 58.1% | -4.5% |
| 1X | 1.09 | 80.0% | 91.7% | -12.8% |

None passes the existing +5% minimum estimated-return filter. Under 3.5 has a small positive model estimate but also falls below the illustrative 75% probability floor. City-or-draw passes that probability floor but has a negative estimated return at the entered price. No selection was recorded and no realized return exists. The combined builder price is still missing; individual prices must not be multiplied to manufacture a same-game quote.

The separate `man_city_fulham_2026-11-21_price_check.json` retains the supplied quotes, filter results and unresolved status.

### Reproduce and inspect [4]

```powershell
.\.venv\Scripts\python.exe scripts/test_market_policies.py --out artifacts/market_trials_v2
.\.venv\Scripts\python.exe scripts/summarize_market_trial.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

[4] Inputs: `artifacts/markets/forecasts.csv` (SHA-256 recorded in the trial report). Outputs: `artifacts/market_trials_v2/report.json`, `forecasts.csv`, `slips.json`, and `man_city_fulham_2026-11-21.json`. The forecast and slip exports retain every evaluated candidate and ticket settlement. The full report preserves bin counts, paired-baseline Brier, fixed thresholds, fold dates, coverage and Wilson intervals. The six new checks cover chronology, whole-date calibration, missing form, future-label isolation, skipping and paired slip settlement; all 53 project tests passed.

No source downloads occurred. Raw CSVs remain absent. This does not complete a head-to-head feature ablation or the originally proposed raw-head calibration refit. All inspected history is exploratory; a future priced paper trial remains required. Wilson intervals assume independent trials and may understate uncertainty from repeated teams/dates. Do not sum overlapping report scopes or policies as distinct trials.

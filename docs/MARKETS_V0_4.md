# Match markets in v0.4

For an independent check of the saved calculations, calibration diagnostics and focused next experiments, see [the saved results check](MARKET_RESULTS_CHECK.md). Run `python scripts/check_market_results.py --out artifacts/markets/checks.json` without refitting or downloading data.

The market model uses 4,280 Premier League results from 2015/16 through 8 November 2026. It estimates home and away goal counts, exact score probabilities, total goals, both teams to score, straight outcomes and double chance. It also estimates total recorded yellows and the probability of at least one recorded red.

| Selection | Settles on |
|---|---|
| home / draw / away | The specified full-time outcome |
| 1X | Home win or draw |
| X2 | Draw or away win |
| 12 | Either team wins |
| over_2.5 | Three or more combined goals |
| under_2.5 | Two or fewer combined goals |
| btts_yes | Each team scores at least once |
| score_2_1 | Home scores two and away scores one |

Double chance uses the sum of its two outcome probabilities. Betway explains its regulation-time settlement in [Double Chance](https://www.betway.co.za/betway-pedia/documents/betway-sports/page/double-chance).

## Forecast and compare prices

```powershell
.\.venv\Scripts\python.exe -m epl_predictor markets --demo --price 1X=1.45 --price over_2.5=1.90 --price score_2_1=9.00
```

The demo uses an illustrative City–United matchup and the example prices above. For a future match, supply the verified fixture date with `--date YYYY-MM-DD` and its home and away teams with `--home` and `--away`. Enter the decimal prices shown for that fixture using one `--price` argument per quote. Add `--json` to capture the full forecast and time of entry.

The model displays fair odds as one divided by probability. It displays the entered quote's break-even probability as one divided by its odds. Its estimated return per unit is probability times the quote minus one. The paper filter uses a fixed +5% estimated return threshold and a 65% probability floor for straight home or away wins. The result focus defaults to the higher-probability draw cover below that straight-win floor. The focus describes the model probability. Entered prices determine the separate filter output.

A match's history includes head-to-head records across both venues since 2015/16. The output lists the latest five results and their archived 1X2 quotes where present. For fitting, the model uses the previous five meetings with shrinkage toward a league-like prior, alongside each team's form, venue and strength features.

## What the archive shows

Each test season was forecast with models fit on earlier seasons. Earlier completed dates supply form for subsequent fixtures. Home and away goals use independent Poisson distributions. Total recorded yellows use a separate Poisson regression, and any red uses a regularized binary classifier. Fitting uses a three-year recency half-life. Verified injury coverage remains zero.

| Metric | 3,520 walk-forward matches | Latest 100 matches |
|---|---:|---:|
| Top straight 1X2 outcome hit | 53.3% | 42.0% |
| Higher-probability option between 1X and X2 hit | 76.8% | 69.0% |
| Top exact score hit | 11.8% | 11.0% |
| Mean absolute total-goal error | 1.31 | 1.40 |

Over 3.5 recorded yellows and any recorded red both have worse probability error than their constant historical baselines across the full walk-forward run. Card results are therefore exploratory. Betway's rules distinguish card counts and booking points and specify which participants and dismissals count. Price comparisons for cards require verified settlement equivalence with the archive. [Betway football settlement rules](https://cdn.betway.co.za/terms-and-conditions)

| Historical price group | Fixtures with odds | Paper selections | Return per unit |
|---|---:|---:|---:|
| Bet365 1X2 | 3,520 | 594 | −1.6% |
| Bet365 over/under 2.5 | 2,759 | 1,229 | −2.7% |

For each price group and fixture, the simulation records at most one eligible selection, using the highest estimated return. Every paper selection has one unit of exposure. Earlier archive-average goal odds contribute to the probability benchmark only. The full results are in `artifacts/markets/report.json`, including the smaller latest-season samples, odds coverage and baseline errors. Archived quote timestamps are unaudited. Historical Betway, double chance, correct-score, both-teams-to-score and card prices are unavailable in these files. The positive latest-season results in the two priced groups are small historical samples, while the longer runs lost units. A future recorded Betway trial is needed to measure results at actual offered prices. [Football-Data column notes](https://www.football-data.co.uk/notes.txt)

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m epl_predictor download-data --start-season 2015 --end-season 2026 --out data
.\.venv\Scripts\python.exe -m epl_predictor benchmark-markets --data-dir data --quarantine-invalid-shots
```

The market model included with the ZIP is refit on all available results for future estimates. The historical metrics use the earlier-season fits. `forecasts.csv` holds the historical forecasts, and `paper_price_checks.csv` holds the selections and unit results. The legacy `predict`, `benchmark` and `score` commands retain their original outcome research workflow.

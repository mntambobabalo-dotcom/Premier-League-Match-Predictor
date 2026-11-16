# Data sources — v0.3

Checked 10 November 2026. The screenshot/video project is number one: match outcome prediction. The code is our implementation, not a claimed copy of the creator's training code.

## Match source

[Football-Data England](https://www.football-data.co.uk/englandm.php), with [column notes](https://www.football-data.co.uk/notes.txt) and [usage policy](https://football-data.co.uk/contact.php).

The twelve season start years are 2015 through 2026. The explicit downloader forms URLs such as https://www.football-data.co.uk/mmz4281/1516/E0.csv and caches four-digit season CSV filenames. Eleven full seasons have 380 results each, and 2026/27 has 100 through 8 November 2026: total 4,280. Source URLs, original byte hashes, coverage and the one shot-stat quarantine are retained in artifacts/benchmark.json and docs/DATA_COVERAGE_V0_3.json.

Results, goals, shots, shots on target, yellows, reds, corners and fouls are present. Possession and verified injury/suspension snapshots are absent. Bookmaker B365H/D/A fields are used only for a historical comparator, with unaudited exact snapshot timing; they are excluded from features and selection.

The source CSV bytes are unchanged and are not bundled. --quarantine-invalid-shots explicitly masks one impossible affected-side pair during loading, records the original values and keeps its result. Generic combined CSVs should provide Season so COVID-delayed July 2020 results are not relabelled by calendar year.

The free source is intended for permitted private research. This is not a commercial data licence. The engine has no hidden scraper or recurring downloads; its explicit acquisition command reuses cached files.

## Availability and alternative sources

See INJURY_DATA_AUDIT.md for the primary sources investigated and why none was accepted as verified point-in-time injury history. The original screenshots also mention Kaggle, DataHub and StatsBomb/Hudl open data; those were not integrated in this run and are not assumed to be complete current EPL feeds.

## Method references

- [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [scikit-learn chronological evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [SciPy Skellam probabilities](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skellam.html)
- [City's account of Pep's July 2016 arrival](https://live.mancity.com/news/mens/pep-guardiola-joe-mercer-longest-serving-manager-list-63741392)

Source documentation does not establish predictive quality. The code's recorded tests and future paper trials must supply that evidence.

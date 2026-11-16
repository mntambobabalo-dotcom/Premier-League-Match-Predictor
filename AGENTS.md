# Project instructions

Read README.md, CODEX_HANDOFF.md, docs/MARKETS_V0_4.md and docs/INJURY_DATA_AUDIT.md before modifying this project.

- Continue the Python engine first. Build an app only when the user asks for that next phase.
- Work with SK step by step; prefer focused edits and short explanations over full-file dumps.
- The current task centers on match markets, scorelines, goals and cards. Report probability calibration, odds coverage, hit rates and paper returns for the exact markets evaluated.
- Features must contain only information known at prediction time. Never use current-match scores, shots/cards, retrospective injury totals or future recovery dates as prematch inputs.
- Injury/suspension snapshots need real KnownAt and Source. Missing is not zero injuries. Current injury news alone is not training data.
- Do not infer player bans from team yellow/red totals. No medical diagnosis is part of this project.
- Preserve source Season identities, especially the July 2020 end of 2019/20. Generic combined CSVs should provide explicit Season.
- Source shot quarantine is explicit and recorded; never guess replacements or drop inconvenient results silently.
- Never randomly shuffle chronological evaluation. Keep whole dates together and keep calibration later than parameter fitting.
- Model selection uses the three most recent completed development seasons; latest season is excluded. Previously inspected history is exploratory, not an untouched holdout.
- Confidence-subset hit rates require counts, coverage and uncertainty; do not present them as all-match results.
- Preserve the latest paper refit's distinction from frozen historical evaluation models.
- Keep archived prices separate from forecasting inputs. Use quoted odds for probability benchmarks and paper price checks. Current Betway prices are entered manually.
- Keep models in safe JSON; do not introduce pickle loading. Do not embed credentials or API tokens in source or artifacts.
- data/ and local environments stay out of source control. Source acquisitions are explicit, cached and mindful of provider terms. No automatic recurring downloads.
- Use python -m unittest discover -s tests -v for validation. The boost extra enables XGBoost tests/comparison. Run meaningful checks for changed behavior, then stop retesting when checks pass.
- Add source/timing coverage, same-fixture ablations and a genuinely future paper trial before making stronger model claims.

# Injury/suspension data audit

Checked 10 November 2026. **No verified pre-match historical availability observations have been integrated.** The included model has zero injury/suspension training coverage. It cannot honestly be presented as injury-aware.

## Sources investigated

| Source | What is documented | Why it is not in the headline model |
|---|---|---|
| [Football-Data match CSVs](https://www.football-data.co.uk/notes.txt) | Results, goals, shots, cards, fouls, corners and odds | No player injury history or named suspension roster |
| [Public availability-data repository](https://github.com/withqwerty/availability-data) | README describes 2015/16 onward retrospective player matchday statuses and a club-season aggregate | Its documented scrape timestamp is not evidence of when each injury was known before a historical kickoff. Whole-season burden is future information for earlier fixtures. The documented sample raw-file path also did not resolve during inspection; nothing was silently substituted |
| [API-Football injuries](https://www.api-football.com/documentation-v3#tag/Injuries) | Missing/questionable fixture absences; documentation says availability begins April 2021 | Account/key access and exact historical point-in-time semantics were not established; this does not itself cover 2015 onward |
| [Sportmonks sidelined history](https://docs.sportmonks.com/v3/endpoints-and-entities/entities/team-player-squad-coach-and-referee) | Historical absence entities with start/end dates, player/team/season IDs and games missed | Access/coverage were not established. Injury start date is not necessarily its report time; eventual end dates and total missed games can disclose future outcomes |

This does not prove that no usable dataset exists. It records what was verified in this session. No paid account was created and no credential was requested or embedded.

## What counts as usable input

For each team and fixture, provide Injured and Suspended counts with a genuine KnownAt timestamp and Source identifying the information available then. KeyMissing is optional and must use a consistent, historically available definition. A source timestamp must describe the supplied values, not just today's collection time or a guessed injury start date.

The template is examples/availability_template.csv. The code validates timestamps, counts and match/team keys. Multiple revisions are allowed; only the latest eligible version is used. Unknown differs from known zero. Date-only evaluation conservatively excludes matchday snapshots; future work can add verified UTC kickoff times throughout training, forecasting and journal scoring.

Partial reliable coverage is still useful: for example, a verified 2021-onward subset could be trained/evaluated with older rows explicitly marked unknown. Full 2015 coverage is not a prerequisite for beginning a properly scoped injury experiment. What is missing is verified accessible data and its forecast-time semantics, not an arbitrary requirement that every year be complete.

## Implemented integration

```sh
python -m epl_predictor benchmark --data-dir data --quarantine-invalid-shots --availability-csv inputs/availability.csv --out artifacts/with_availability
```

Do not pass the empty template as if it were a populated injury dataset. Once real snapshots are integrated, check training/calibration/test coverage separately and compare with/without availability on identical fixtures. A future injury-aware forecast is rejected when the selected model has no historical availability training coverage.

## Discipline is a different input

Previous yellow/red cards are available and included in the main model. They describe earlier matches. Team-level cards do not identify which player is currently banned or the applicable ban length. No named suspension is inferred from those totals, and a card occurring during the match being predicted is never a prematch feature.

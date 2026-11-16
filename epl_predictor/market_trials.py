"""Chronological experiments on saved forecasts, without source acquisition.

These are research diagnostics, not live picks or a new production model.
"""

from hashlib import sha256
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .markets import goal_probabilities


EVENTS = ("under_3.5", "btts_yes", "1X", "X2")
THRESHOLDS = (0.65, 0.75, 0.80)
CANDIDATES = ("raw", "prior_rate", "rate_scaled")


def prior_form(rows):
    """Five earlier league games per team; an incomplete history stays unknown."""
    rows = rows.copy()
    history, records = defaultdict(list), []
    for _, fixtures in rows.groupby("Date", sort=False):
        for index, row in fixtures.iterrows():
            values = {"index": index}
            for side, team in [("home", row.HomeTeam), ("away", row.AwayTeam)]:
                recent = history[team][-5:]
                values[f"{side}_prior_games"] = len(recent)
                values[f"{side}_prior_total"] = float(np.mean([gf + ga for gf, ga in recent])) if len(recent) == 5 else np.nan
                values[f"{side}_prior_conceded"] = float(np.mean([ga for gf, ga in recent])) if len(recent) == 5 else np.nan
            records.append(values)
        for row in fixtures.itertuples(index=False):
            history[row.HomeTeam].append((row.FTHG, row.FTAG))
            history[row.AwayTeam].append((row.FTAG, row.FTHG))
    return rows.join(pd.DataFrame(records).set_index("index"))


def hit_metrics(won, population):
    won = np.asarray(won, dtype=bool)
    n = len(won)
    if not n:
        return dict(matches=0, hits=0, coverage=0.0, hit_rate=None, wilson_95=None)
    p, z = float(won.mean()), 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return dict(matches=n, hits=int(won.sum()), coverage=n / population, hit_rate=p,
                wilson_95=[float(center - margin), float(center + margin)])


def actual_events(rows):
    return pd.DataFrame({"under_3.5": rows.FTHG + rows.FTAG <= 3,
                         "btts_yes": (rows.FTHG > 0) & (rows.FTAG > 0),
                         "1X": rows.FTHG >= rows.FTAG, "X2": rows.FTHG <= rows.FTAG}, index=rows.index)


def probabilities(home, away, index):
    h, d, a = goal_probabilities(np.asarray(home), np.asarray(away))
    return pd.DataFrame({"under_3.5": poisson.cdf(3, np.asarray(home) + np.asarray(away)),
                         "btts_yes": -np.expm1(-np.asarray(home)) * -np.expm1(-np.asarray(away)),
                         "1X": h + d, "X2": d + a}, index=index)


def calibration_window(prior, requested=190):
    """Include the entire boundary date; never split a day's fixtures."""
    if len(prior) < requested:
        raise ValueError("Insufficient earlier forecasts for calibration")
    boundary = prior.iloc[-requested].Date
    return prior.loc[prior.Date >= boundary]


def seasonal_predictions(rows, requested=190):
    if not rows.Date.is_monotonic_increasing:
        raise ValueError("Saved forecasts must be chronological")
    if rows.duplicated(["Date", "HomeTeam", "AwayTeam"]).any():
        raise ValueError("Duplicate forecast fixture")
    core = rows[["expected_home_goals", "expected_away_goals", "FTHG", "FTAG"]].to_numpy()
    if not np.isfinite(core).all() or (core < 0).any():
        raise ValueError("Invalid goal counts or expected rates")
    if ((rows[["FTHG", "FTAG"]] % 1) != 0).any().any():
        raise ValueError("Actual goals must be integers")
    outputs, folds = [], []
    for season, test in rows.groupby("season_label", sort=False):
        prior = rows.loc[rows.Date < test.Date.min()]
        if len(prior) < requested:
            continue
        cal = calibration_window(prior, requested)
        if cal.Date.max() >= test.Date.min():
            raise ValueError("Calibration overlaps test dates")
        scales = [float(np.clip(cal[target].sum() / cal[expected].sum(), 0.5, 2.0))
                  if cal[expected].sum() else 1.0
                  for target, expected in [("FTHG", "expected_home_goals"), ("FTAG", "expected_away_goals")]]
        raw = probabilities(test.expected_home_goals, test.expected_away_goals, test.index)
        scaled = probabilities(test.expected_home_goals * scales[0], test.expected_away_goals * scales[1], test.index)
        prior_rates = actual_events(cal).mean()
        frame = test.copy()
        for name, probs in [("raw", raw), ("rate_scaled", scaled)]:
            for event in EVENTS:
                frame[f"{name}:{event}"] = probs[event]
        for event in EVENTS:
            frame[f"prior_rate:{event}"] = float(prior_rates[event])
        outputs.append(frame)
        folds.append(dict(season=season, calibration_matches=len(cal), calibration_start=cal.Date.min(),
                          calibration_end=cal.Date.max(), test_start=test.Date.min(), test_end=test.Date.max(),
                          home_scale=scales[0], away_scale=scales[1]))
    if len(outputs) < 2:
        raise ValueError("Need earlier forecast history, development and latest seasons")
    return pd.concat(outputs), folds


def event_metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = []
    for i in range(10):
        mask = np.minimum((p * 10).astype(int), 9) == i
        if mask.any():
            bins.append(dict(lower=i / 10, upper=(i + 1) / 10, matches=int(mask.sum()),
                             mean_probability=float(p[mask].mean()), actual_rate=float(y[mask].mean())))
    return dict(matches=len(y), mean_probability=float(p.mean()), actual_rate=float(y.mean()),
                brier=float(np.square(p - y).mean()), calibration_bins=bins)


def evaluate(rows, candidate):
    actual = actual_events(rows)
    metrics = {e: event_metrics(actual[e], rows[f"{candidate}:{e}"]) for e in EVENTS}
    return dict(matches=len(rows), mean_event_brier=float(np.mean([v["brier"] for v in metrics.values()])), events=metrics)


def choose_candidate(development_scores):
    # Audit rows cannot be passed to this function; tie order is fixed.
    return min(CANDIDATES, key=lambda c: (development_scores[c]["mean_event_brier"], CANDIDATES.index(c)))


def selections(rows, candidate, family, threshold):
    actual = actual_events(rows)
    names = {"under_3.5": ["under_3.5"], "btts": ["btts_yes", "btts_no"],
             "draw_cover": ["1X", "X2"],
             "mixed": ["under_3.5", "btts_yes", "btts_no", "1X", "X2"]}[family]
    p = {e: rows[f"{candidate}:{e}"] for e in EVENTS}
    p["btts_no"] = 1 - p["btts_yes"]
    actual["btts_no"] = ~actual.btts_yes
    matrix = np.column_stack([p[n] for n in names])
    choice = matrix.argmax(axis=1)
    selected_p = matrix[np.arange(len(rows)), choice]
    won = actual[names].to_numpy()[np.arange(len(rows)), choice]
    result = rows[["Date", "HomeTeam", "AwayTeam"]].copy()
    result["selection"] = np.asarray(names)[choice]
    result["probability"], result["won"] = selected_p, won
    return result.loc[selected_p >= threshold]


def slip_ladder(selected, population_dates):
    """Rank once per date; compare sizes both on available and identical dates."""
    records = {legs: [] for legs in (1, 2, 3, 4)}
    for date, group in selected.groupby("Date", sort=True):
        group = group.sort_values(["probability", "HomeTeam", "AwayTeam"], ascending=[False, True, True])
        chosen, teams = [], set()
        for row in group.itertuples(index=False):
            if row.HomeTeam in teams or row.AwayTeam in teams:
                continue
            chosen.append(row)
            teams.update([row.HomeTeam, row.AwayTeam])
            if len(chosen) == 4:
                break
        for legs in (1, 2, 3, 4):
            if len(chosen) < legs:
                continue
            pick = chosen[:legs]
            records[legs].append(dict(date=date, legs=legs, won=all(r.won for r in pick),
                                independent_probability=float(np.prod([r.probability for r in pick])),
                                fixtures=[dict(home=r.HomeTeam, away=r.AwayTeam, selection=r.selection,
                                               probability=float(r.probability), won=bool(r.won)) for r in pick]))
    def summarize(records):
        summary = hit_metrics([r["won"] for r in records], population_dates)
        summary["mean_independent_probability"] = float(np.mean([r["independent_probability"] for r in records])) if records else None
        summary["odds_coverage"], summary["paper_return"] = 0, None
        return summary
    paired_dates = {r["date"] for r in records[4]}
    return {legs: (summarize(values), values,
                   summarize([r for r in values if r["date"] in paired_dates]))
            for legs, values in records.items()}


def slips(selected, population_dates, legs):
    summary, records, _ = slip_ladder(selected, population_dates)[legs]
    return summary, records


def goal_risks(rows, candidate):
    """Prematch association checks; realized scores describe outcomes only."""
    total = rows.expected_home_goals + rows.expected_away_goals
    weaker = np.minimum(rows.expected_home_goals, rows.expected_away_goals)
    conditions = {
        "expected_total_below_2.5": total < 2.5,
        "expected_total_2.5_to_3": (total >= 2.5) & (total < 3),
        "expected_total_at_least_3": total >= 3,
        "both_expected_at_least_1": weaker >= 1,
        "one_expected_below_1": weaker < 1,
        "under_probability_at_least_75pct": rows[f"{candidate}:under_3.5"] >= 0.75,
    }
    form_known = rows.home_prior_total.notna() & rows.away_prior_total.notna()
    both_busy = (rows.home_prior_total >= 3) & (rows.away_prior_total >= 3)
    leaky = rows.home_prior_conceded + rows.away_prior_conceded >= 3
    conditions.update(both_prior_five_total_at_least_3=form_known & both_busy,
                      at_least_one_prior_five_total_below_3=form_known & ~both_busy,
                      prior_five_combined_conceded_at_least_3=form_known & leaky,
                      prior_five_combined_conceded_below_3=form_known & ~leaky)
    actual = actual_events(rows)
    segments = {}
    for name, mask in conditions.items():
        part = rows.loc[mask]
        segments[name] = {**hit_metrics(~actual.loc[mask, "under_3.5"], len(rows)),
                          "under_hit_rate": float(actual.loc[mask, "under_3.5"].mean()) if len(part) else None,
                          "mean_expected_total": float(total.loc[mask].mean()) if len(part) else None}
    over = rows.loc[~actual["under_3.5"]]
    btts = (over.FTHG > 0) & (over.FTAG > 0)
    return dict(segments=segments, prior_form_covered_fixtures=int(form_known.sum()),
                over_games=len(over), over_with_btts=int(btts.sum()),
                over_with_clean_sheet=int((~btts).sum()),
                most_common_over_scores=over.assign(score=over.FTHG.astype(str) + "-" + over.FTAG.astype(str))
                .score.value_counts().head(8).to_dict())


def run_trial(source, out):
    source, out = Path(source), Path(out)
    rows = pd.read_csv(source)
    rows, folds = seasonal_predictions(prior_form(rows))
    seasons = list(rows.season_label.unique())
    selection_seasons = seasons[:-1][-3:]
    development = rows.loc[rows.season_label.isin(selection_seasons)]
    scores = {c: evaluate(development, c) for c in CANDIDATES}
    selected = choose_candidate(scores)
    report = dict(selection_seasons=selection_seasons, latest_season=seasons[-1], selected_candidate=selected,
                  selection_metric="Mean Brier across under_3.5, BTTS yes, 1X and X2; latest season excluded",
                  development_scores=scores, folds=folds, scopes={},
                  source=dict(path=str(source), sha256=sha256(source.read_bytes()).hexdigest()),
                  policy=dict(candidates=list(CANDIDATES), thresholds=list(THRESHOLDS), slip_sizes=[1, 2, 3, 4],
                              requested_prior_calibration_matches=190, calibration_scale_bounds=[0.5, 2.0],
                              threshold_selection="none; all fixed thresholds reported"),
                  limits=["Saved forecast experiment; original goal-head parameters were not refit. Earlier out-of-season forecasts calibrate later seasons.",
                          "Raw CSVs absent. No injury, shot, red-card timing or lineup cause was inferred.",
                          "Previously viewed history is exploratory, including the latest season. Selected candidate is not the production paper refit.",
                          "No prices for these markets: odds coverage is zero and paper returns are unavailable, not zero.",
                          "Slip probability products assume independence across fixtures; observed outcomes are settled jointly. Same-game picks are excluded.",
                          "Wilson intervals assume independent observations; repeated teams and shared dates can increase uncertainty."])
    all_slips = []
    for scope, subset in [("all_evaluable", rows), ("recent_development", development),
                          ("latest_season", rows.loc[rows.season_label.eq(seasons[-1])])]:
        result = dict(models={c: evaluate(subset, c) for c in CANDIDATES}, filters={}, slips={}, paired_slips={},
                      over_goal_analysis=goal_risks(subset, selected))
        for family in ("under_3.5", "btts", "draw_cover", "mixed"):
            for threshold in THRESHOLDS:
                key = f"{family}@{threshold:.2f}"
                picks = selections(subset, selected, family, threshold)
                # Same selected fixtures, using the prior-only baseline probability of the same event.
                base_p, wins = [], []
                for index, pick in picks.iterrows():
                    event = "btts_yes" if pick.selection == "btts_no" else pick.selection
                    p = subset.loc[index, f"prior_rate:{event}"]
                    base_p.append(1 - p if pick.selection == "btts_no" else p)
                    wins.append(pick.won)
                result["filters"][key] = {**hit_metrics(wins, len(subset)),
                    "mean_probability": float(picks.probability.mean()) if len(picks) else None,
                    "model_brier_same_fixtures": float(np.mean((picks.probability.to_numpy() - wins) ** 2)) if len(picks) else None,
                    "prior_rate_brier_same_fixtures": float(np.mean((np.asarray(base_p) - wins) ** 2)) if len(picks) else None,
                    "odds_coverage": 0, "paper_return": None}
                for count, (metric, records, paired) in slip_ladder(picks, subset.Date.nunique()).items():
                    result["slips"][f"{key}/{count}"] = metric
                    result["paired_slips"][f"{key}/{count}"] = paired
                    all_slips.extend(dict(scope=scope, policy=key, **r) for r in records)
        report["scopes"][scope] = result
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "slips.json").write_text(json.dumps(all_slips, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows.to_csv(out / "forecasts.csv", index=False)
    return report

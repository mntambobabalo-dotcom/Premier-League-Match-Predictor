"""Audit saved market results without downloading data or refitting models.

Run from the project root with:
    python scripts/check_market_results.py --out artifacts/markets/checks.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(bool(np.allclose(actual, expected, rtol=1e-9, atol=1e-10)), message)


def hit_summary(won, population):
    """Wilson 95% interval; descriptive, assumes independent Bernoulli trials."""
    won = np.asarray(won, dtype=bool)
    n = len(won)
    if not n:
        return {"matches": 0, "coverage": 0.0, "hit_rate": None, "wilson_95": None}
    p, z = float(won.mean()), 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {"matches": n, "hits": int(won.sum()), "coverage": n / population,
            "hit_rate": p, "wilson_95": [float(center - margin), float(center + margin)]}


def probability_summary(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    known = np.isfinite(y) & np.isfinite(p)
    y, p = y[known], p[known]
    if not len(y):
        return {"matches": 0}
    bins = []
    # Fixed bins, including p=1 in the final bin; no fitted calibration here.
    membership = np.minimum((p * 10).astype(int), 9)
    for i in range(10):
        selected = membership == i
        if selected.any():
            bins.append({"probability_range": [i / 10, (i + 1) / 10],
                         "matches": int(selected.sum()),
                         "mean_probability": float(p[selected].mean()),
                         "actual_rate": float(y[selected].mean())})
    return {"matches": len(y), "mean_probability": float(p.mean()),
            "actual_rate": float(y.mean()), "probability_minus_actual": float((p - y).mean()),
            "brier": float(np.square(p - y).mean()),
            "ece_10_bins": sum(b["matches"] * abs(b["mean_probability"] - b["actual_rate"])
                               for b in bins) / len(y),
            "calibration_bins": bins}


def events(rows):
    h, a = rows.FTHG, rows.FTAG
    return {"home": h > a, "draw": h == a, "away": h < a,
            "1X": h >= a, "X2": h <= a, "12": h != a,
            "over_2.5": h + a > 2, "under_2.5": h + a <= 2,
            "btts": (h > 0) & (a > 0),
            "yellow_over_3.5": (rows.actual_yellow_cards > 3).where(rows.actual_yellow_cards.notna()),
            "red_any": rows.actual_red_any}


def summarize(rows):
    actual = events(rows)
    results = {name: probability_summary(y, rows["expected_red_any" if name == "red_any" else "p_" + name])
               for name, y in actual.items()}
    picks = {}
    for name, names in [("top_1x2", ["home", "draw", "away"]),
                        ("top_draw_cover", ["1X", "X2"]),
                        ("top_total_2_5", ["over_2.5", "under_2.5"])]:
        p = rows[["p_" + key for key in names]].to_numpy()
        y = np.column_stack([actual[key] for key in names])
        choice = p.argmax(axis=1)
        selected_p = p[np.arange(len(rows)), choice]
        won = y[np.arange(len(rows)), choice]
        picks[name] = {**hit_summary(won, len(rows)),
                       "mean_probability": float(selected_p.mean())}
        if name == "top_1x2":
            mask = selected_p >= 0.65
            picks["top_1x2_at_least_65pct"] = hit_summary(won[mask], len(rows))
    picks["top_exact_score"] = hit_summary(rows.modal_exact_hit, len(rows))
    return {"matches": len(rows), "events": results, "picks": picks,
            "total_goals_mae": float((rows.expected_home_goals + rows.expected_away_goals
                                      - rows.actual_total).abs().mean()),
            "scoreline_log_loss": float(-np.log(rows.p_exact_score.clip(1e-15, 1)).mean())}


def check_report(rows, reported, summary):
    require(len(rows) == reported["matches"], "Report fixture count mismatch")
    for key in ("total_goals_mae", "scoreline_log_loss"):
        close(summary[key], reported[key], "Report mismatch: " + key)
    close(rows.modal_exact_hit.mean(), reported["exact_score_top_pick_accuracy"], "Exact-score hit mismatch")
    for name, metric in summary["events"].items():
        if metric["matches"]:
            for key in ("matches", "actual_rate", "mean_probability", "brier"):
                close(metric[key], reported["event_metrics"][name][key], f"Report mismatch: {name}/{key}")
    for key in ("top_1x2", "top_draw_cover"):
        close(summary["picks"][key]["hit_rate"], reported["event_metrics"][key], "Pick hit mismatch: " + key)


def check_prices(prices, rows, reported):
    require(not prices.duplicated(["date", "home", "away", "group"]).any(), "Duplicate paper selection")
    merged = prices.merge(rows, left_on=["date", "home", "away"],
                          right_on=["Date", "HomeTeam", "AwayTeam"], validate="many_to_one", indicator=True)
    require(merged._merge.eq("both").all(), "Paper selection has no forecast")
    require(merged.group.isin(["bet365_1x2", "bet365_total_2_5"]).all(), "Unsupported priced group")
    require((np.isfinite(merged.decimal_odds) & merged.decimal_odds.gt(1)).all(), "Invalid paper odds")
    actual = events(merged)
    for selection, part in merged.groupby("selection"):
        require(selection in ("home", "draw", "away", "over_2.5", "under_2.5"), "Unsupported selection")
        expected_group = "bet365_1x2" if selection in ("home", "draw", "away") else "bet365_total_2_5"
        require(part.group.eq(expected_group).all(), "Selection/price group mismatch")
        close(part.probability, part["p_" + selection], "Paper probability differs from forecast")
        close(part.won.astype(int), actual[selection].loc[part.index].astype(int), "Paper settlement mismatch")
    close(merged.paper_net_units, merged.decimal_odds * merged.won - 1, "Paper payout mismatch")
    require((merged.probability * merged.decimal_odds - 1 >= 0.05 - 1e-12).all(), "Paper edge filter mismatch")
    win = merged.selection.isin(["home", "away"])
    require(merged.loc[win, "probability"].ge(0.65).all(), "Straight-win probability filter mismatch")
    summaries = {}
    for name, expected in reported.items():
        if not name.startswith("bet365"):
            continue
        group = merged.loc[merged.group.eq(name)]
        require(len(group) == expected["paper_selections"], "Paper selection count mismatch")
        close(group.paper_net_units.sum(), expected["paper_net_units"], "Paper net mismatch")
        if len(group):
            close(group.paper_net_units.mean(), expected["paper_return"], "Paper return mismatch")
            close(group.won.mean(), expected["paper_hit_rate"], "Paper hit mismatch")
        summaries[name] = {"priced_fixtures_reported": expected["priced_fixtures"],
                           "odds_coverage_reported": expected["priced_fixtures"] / len(rows),
                           **hit_summary(group.won, len(rows)),
                           "selection_coverage_of_priced_fixtures": len(group) / expected["priced_fixtures"],
                           "selection_calibration": probability_summary(group.won, group.probability),
                           "paper_net_units": float(group.paper_net_units.sum()),
                           "paper_return": float(group.paper_net_units.mean()) if len(group) else None,
                           "mean_estimated_return": float((group.probability * group.decimal_odds - 1).mean())
                           if len(group) else None}
    return summaries


def audit(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(directory / "forecasts.csv")
    prices = pd.read_csv(directory / "paper_price_checks.csv")
    require(len(rows) > 0, "No saved forecasts")
    require(pd.to_datetime(rows.Date).is_monotonic_increasing, "Forecasts are not chronological")
    require(not rows.duplicated(["Date", "HomeTeam", "AwayTeam"]).any(), "Duplicate forecast")
    for col in [c for c in rows if c.startswith("p_")] + ["expected_red_any"]:
        values = rows[col].dropna()
        require((np.isfinite(values) & values.between(0, 1)).all(), "Invalid probability: " + col)
    core = ["p_home", "p_draw", "p_away", "p_over_2.5", "p_under_2.5", "p_btts", "p_exact_score"]
    require(rows[core].notna().all().all(), "Missing goal/outcome probabilities")
    close(rows.p_home + rows.p_draw + rows.p_away, 1, "1X2 probabilities do not sum to one")
    for name, first, second in [("1X", "home", "draw"), ("X2", "draw", "away"), ("12", "home", "away")]:
        close(rows["p_" + name], rows["p_" + first] + rows["p_" + second], "Double-chance mismatch")
    close(rows["p_over_2.5"] + rows["p_under_2.5"], 1, "Totals do not sum to one")
    close(rows.actual_total, rows.FTHG + rows.FTAG, "Actual total mismatch")
    close(rows.p_exact_score, poisson.pmf(rows.FTHG, rows.expected_home_goals)
          * poisson.pmf(rows.FTAG, rows.expected_away_goals), "Exact-score probability mismatch")
    close(rows.modal_exact_hit, (rows.FTHG == np.floor(rows.expected_home_goals))
          & (rows.FTAG == np.floor(rows.expected_away_goals)), "Modal score settlement mismatch")
    summaries = {}
    latest_season = rows.iloc[-1].season_label
    for name, subset, expected, expected_prices in [
        ("all_walk_forward", rows, report["all_walk_forward"], report["historical_prices"]),
        ("latest_season", rows.loc[rows.season_label.eq(latest_season)], report["latest_season"], report["latest_season_prices"]),
    ]:
        summaries[name] = summarize(subset)
        check_report(subset, expected, summaries[name])
        selected = prices if name == "all_walk_forward" else prices.loc[prices.date.isin(subset.Date)]
        summaries[name]["paper_prices"] = check_prices(selected, subset, expected_prices)
    for season, subset in rows.groupby("season_label", sort=False):
        check_report(subset, report["season_metrics"][season], summarize(subset))
    development = list(rows.season_label.unique())[:-1][-3:]
    summaries["recent_completed_development"] = {
        "seasons": development, **summarize(rows.loc[rows.season_label.isin(development)])}
    return {"status": "saved_calculations_verified", "latest_season": latest_season,
            "limits": ["Raw source files are absent here: source integrity, full odds coverage, odds benchmarks and completeness of the paper filter cannot be independently rechecked.",
                       "No model refit, calibration fit, threshold search or new holdout evaluation was performed.",
                       "Calibration bins are descriptive. Wilson intervals assume independent trials; shared teams/dates may increase uncertainty.",
                       "Archived quote timing and card settlement equivalence remain unaudited. No historical prices for double chance, exact score, BTTS or cards.",
                       "Historical fits differ from the latest paper refit. Injury training coverage remains zero."],
            **{"results": summaries}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="artifacts/markets")
    parser.add_argument("--out", help="Optional JSON output; original artifacts are preserved")
    args = parser.parse_args()
    result = audit(args.directory)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        Path(args.out).write_text(encoded + "\n", encoding="utf-8")
        print(f"Saved verified market calculations: {args.out}")
    else:
        print(encoded)

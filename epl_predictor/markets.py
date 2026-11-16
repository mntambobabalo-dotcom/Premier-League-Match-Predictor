"""Pre-match score, goal and discipline distributions with historical price checks."""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import poisson, skellam
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler

from .features import ADVANCED_FEATURES, build_features, fixture_features
from .research import seasonal_folds


H2H_COLUMNS = ["meetings", "home_goals", "away_goals", "draw_share"]
HEADS = ("home_goals", "away_goals", "yellow_cards", "red_any")
EDGE_FLOOR = 0.05
WIN_FLOOR = 0.65
PRICE_GROUPS = {
    "bet365_1x2": (("home", "draw", "away"), ("B365H", "B365D", "B365A")),
    "bet365_total_2_5": (("over_2.5", "under_2.5"), ("B365>2.5", "B365<2.5")),
    "archive_average_total_2_5": (("over_2.5", "under_2.5"), ("BbAv>2.5", "BbAv<2.5")),
}


def pair_key(home, away):
    return json.dumps(sorted((home, away)), ensure_ascii=False)


def meetings_features(meetings, home):
    meetings = meetings[-5:]
    own = [r["hg"] if r["home"] == home else r["ag"] for r in meetings]
    against = [r["ag"] if r["home"] == home else r["hg"] for r in meetings]
    n = len(meetings)
    # Five prior-equivalent matches shrink a small sample toward league-like values.
    return [n, (sum(own) + 5 * 1.4) / (n + 5), (sum(against) + 5 * 1.4) / (n + 5),
            (sum(a == b for a, b in zip(own, against)) + 5 * 0.25) / (n + 5)]


def market_features(matches):
    base, teams = build_features(matches, ADVANCED_FEATURES)
    history = defaultdict(list)
    rows = []
    for _, group in matches.groupby("Date", sort=False):
        fixtures = list(group.itertuples(index=False))
        for row in fixtures:
            rows.append(meetings_features(history[pair_key(row.HomeTeam, row.AwayTeam)], row.HomeTeam))
        for row in fixtures:
            odds = [getattr(row, column, np.nan) for column in ("B365H", "B365D", "B365A")]
            history[pair_key(row.HomeTeam, row.AwayTeam)].append(
                {"date": row.Date.date().isoformat(), "home": row.HomeTeam, "away": row.AwayTeam,
                 "hg": int(row.FTHG), "ag": int(row.FTAG),
                 "archived_bet365_1x2": [float(value) for value in odds] if np.isfinite(odds).all() else None}
            )
    h2h = pd.DataFrame(rows, columns=H2H_COLUMNS, index=base.index)
    return pd.concat([base, h2h], axis=1), teams, dict(history)


def targets(matches):
    red_count = matches["HR"] + matches["AR"]
    return pd.DataFrame({
        "home_goals": matches["FTHG"], "away_goals": matches["FTAG"],
        "yellow_cards": matches["HY"] + matches["AY"],
        "red_any": (red_count > 0).where(red_count.notna()).astype(float),
    })


def fit_heads(features, outcomes, dates, fit):
    scaler = StandardScaler().fit(features.loc[fit])
    x = scaler.transform(features)
    age_days = (dates.loc[fit].max() - dates).dt.days.to_numpy()
    weights = np.exp2(-age_days[fit] / 1095.75)
    weights /= weights.mean()
    weights_by_row = pd.Series(weights, index=dates.index[fit])
    heads = {}
    for name in HEADS:
        known = fit & outcomes[name].notna()
        y = outcomes.loc[known, name].to_numpy()
        if not len(y):
            heads[name] = None
            continue
        w = weights_by_row.loc[outcomes.index[known]].to_numpy()
        baseline = float(np.average(y, weights=w))
        if name == "red_any" and len(np.unique(y)) == 1:
            heads[name] = {"constant": float((y.sum() + 0.5) / (len(y) + 1)),
                           "baseline": baseline, "training_rows": len(y)}
            continue
        if name != "red_any" and not y.any():
            heads[name] = {"constant": 0.0, "baseline": baseline, "training_rows": len(y)}
            continue
        estimator = (LogisticRegression(C=0.2, max_iter=500) if name == "red_any"
                     else PoissonRegressor(alpha=1.0, max_iter=500))
        estimator.fit(x[known], y, sample_weight=w)
        heads[name] = {"coef": estimator.coef_.ravel().tolist(),
                       "intercept": float(np.asarray(estimator.intercept_).ravel()[0]),
                       "baseline": baseline, "training_rows": len(y)}
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "heads": heads}


def estimate(model, features):
    z = (np.atleast_2d(np.asarray(features)) - model["mean"]) / model["scale"]
    values = {}
    for name, head in model["heads"].items():
        if head is None:
            continue
        if "constant" in head:
            values[name] = np.full(len(z), head["constant"])
            continue
        score = z @ head["coef"] + head["intercept"]
        if name == "red_any":
            values[name] = expit(score)
        else:
            cap = 20 if name == "yellow_cards" else 6
            values[name] = np.exp(np.clip(score, np.log(0.01), np.log(cap)))
    return values


def goal_probabilities(home, away):
    # A zero fitted rate is possible with a zero-only training sample.
    home = np.maximum(home, 1e-12)
    away = np.maximum(away, 1e-12)
    draw = skellam.pmf(0, home, away)
    away_win = skellam.cdf(-1, home, away)
    return 1 - draw - away_win, draw, away_win


def market_probabilities(home, away, yellow=None, red=None):
    h, d, a = map(float, goal_probabilities(home, away))
    p = {"home": h, "draw": d, "away": a, "1X": h + d, "X2": d + a, "12": h + a}
    for line in (1.5, 2.5, 3.5):
        p[f"over_{line}"] = float(poisson.sf(int(line), home + away))
        p[f"under_{line}"] = 1 - p[f"over_{line}"]
    p["btts_yes"] = float(-np.expm1(-home) * -np.expm1(-away))
    p["btts_no"] = 1 - p["btts_yes"]
    if yellow is not None:
        for line in (2.5, 3.5, 4.5, 5.5):
            p[f"yellow_over_{line}"] = float(poisson.sf(int(line), yellow))
            p[f"yellow_under_{line}"] = 1 - p[f"yellow_over_{line}"]
    if red is not None:
        p.update(red_any=red, red_none=1 - red)
    return p


def eligible_price(selection, probability, decimal_price):
    if selection in ("home", "away") and probability < WIN_FLOOR:
        return False
    return probability * decimal_price - 1 >= EDGE_FLOOR


def price_checks(raw_quotes, probabilities, home, away):
    checks = {}
    for quote in raw_quotes:
        selection, separator, raw_price = quote.partition("=")
        selection = selection.strip()
        if not separator or selection in checks:
            raise ValueError("Use one --price selection=decimal_odds per selection")
        price = float(raw_price)
        if not np.isfinite(price) or price <= 1:
            raise ValueError("Decimal odds must be finite and greater than 1")
        if selection.startswith(("yellow_", "red_")):
            raise ValueError("Card prices require a verified match between the archive counts and the offered settlement rules")
        probability = probabilities.get(selection)
        if probability is None:
            score = re.fullmatch(r"score_(\d{1,2})_(\d{1,2})", selection)
            if score is None or max(int(score[1]), int(score[2])) > 19:
                raise ValueError(f"Unsupported market selection: {selection}")
            probability = float(poisson.pmf(int(score[1]), home) * poisson.pmf(int(score[2]), away))
        checks[selection] = {"probability": probability, "entered_odds": price,
                             "fair_odds": 1 / probability if probability else None,
                             "break_even_probability": 1 / price,
                             "estimated_return_per_unit": probability * price - 1,
                             "passes_paper_filter": eligible_price(selection, probability, price)}
    return checks


def forecast(artifact, home, away, when, demo=False, quotes=()):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", when):
        raise ValueError("Use an ISO calendar date (YYYY-MM-DD)")
    date = pd.Timestamp(when)
    cutoff = pd.Timestamp(artifact["history_through"])
    now = datetime.now(timezone.utc)
    if date <= cutoff:
        raise ValueError("Fixture date must follow the saved result history")
    if not demo and (date.date() < now.date() or (date - cutoff).days > 21):
        raise ValueError("Refresh result history for a current fixture or use --demo for an illustration")
    if home == away or not {home, away}.issubset(artifact["teams"]):
        raise ValueError("Provide two different teams using their archived names")
    h2h = artifact["meetings"].get(pair_key(home, away), [])
    own_goals = [r["hg"] if r["home"] == home else r["ag"] for r in h2h]
    opp_goals = [r["ag"] if r["home"] == home else r["hg"] for r in h2h]
    head_to_head = {"meetings": len(h2h), "home_team_wins": sum(h > a for h, a in zip(own_goals, opp_goals)),
                    "draws": sum(h == a for h, a in zip(own_goals, opp_goals)),
                    "away_team_wins": sum(h < a for h, a in zip(own_goals, opp_goals)),
                    "mean_total_goals": (sum(own_goals) + sum(opp_goals)) / len(h2h) if h2h else None}
    x = fixture_features(artifact["teams"], home, away, when, ADVANCED_FEATURES)
    rates = {name: float(values[0]) for name, values in
             estimate(artifact, np.r_[x, meetings_features(h2h, home)]).items()}
    hg, ag = rates["home_goals"], rates["away_goals"]
    probabilities = market_probabilities(hg, ag, rates.get("yellow_cards"), rates.get("red_any"))
    grid = np.outer(poisson.pmf(np.arange(20), hg), poisson.pmf(np.arange(20), ag))
    top = np.argsort(grid.ravel())[-5:][::-1]
    top_scores = [{"score": f"{i // 20}-{i % 20}", "probability": float(grid.flat[i])} for i in top]
    win = max(("home", "away"), key=probabilities.get)
    result_focus = win if probabilities[win] >= WIN_FLOOR else max(("1X", "X2"), key=probabilities.get)
    return {"home": home, "away": away, "date": date.date().isoformat(), "demo": demo,
            "created_utc": now.isoformat(), "history_through": artifact["history_through"],
            "result_focus": result_focus, "expected_home_goals": hg, "expected_away_goals": ag,
            "expected_total_goals": hg + ag, "expected_yellows": rates.get("yellow_cards"),
            "top_scores": top_scores, "head_to_head": head_to_head, "last_five_meetings": h2h[-5:],
            "probabilities": probabilities, "price_checks": price_checks(quotes, probabilities, hg, ag)}


def historical_rows(matches, outcomes, model, features, test, season):
    forecast = estimate(model, features.loc[test])
    rows = matches.loc[test].copy()
    for name, predicted in forecast.items():
        rows["expected_" + name] = predicted
        rows["baseline_" + name] = model["heads"][name]["baseline"]
    rows["season_label"] = season
    hg, ag = forecast["home_goals"], forecast["away_goals"]
    rows["p_home"], rows["p_draw"], rows["p_away"] = goal_probabilities(hg, ag)
    rows["p_1X"] = rows["p_home"] + rows["p_draw"]
    rows["p_X2"] = rows["p_draw"] + rows["p_away"]
    rows["p_12"] = rows["p_home"] + rows["p_away"]
    rows["p_over_2.5"] = poisson.sf(2, hg + ag)
    rows["p_under_2.5"] = 1 - rows["p_over_2.5"]
    rows["p_btts"] = -np.expm1(-hg) * -np.expm1(-ag)
    for name in ("yellow_cards", "red_any"):
        rows["actual_" + name] = outcomes.loc[test, name]
    rows["p_yellow_over_3.5"] = poisson.sf(3, rows.get("expected_yellow_cards", np.nan))
    rows["actual_total"] = rows.FTHG + rows.FTAG
    rows["p_exact_score"] = poisson.pmf(rows.FTHG, hg) * poisson.pmf(rows.FTAG, ag)
    rows["modal_exact_hit"] = (rows.FTHG == np.floor(hg)) & (rows.FTAG == np.floor(ag))
    return rows


def realized_events(rows):
    return pd.DataFrame({
        "home": rows.FTR.eq("H"), "draw": rows.FTR.eq("D"), "away": rows.FTR.eq("A"),
        "1X": rows.FTR.ne("A"), "X2": rows.FTR.ne("H"), "12": rows.FTR.ne("D"),
        "over_2.5": rows.actual_total.gt(2), "under_2.5": rows.actual_total.le(2),
        "btts": rows.FTHG.gt(0) & rows.FTAG.gt(0),
        "yellow_over_3.5": rows.actual_yellow_cards.gt(3).where(rows.actual_yellow_cards.notna()),
        "red_any": rows.actual_red_any,
    })


def summarize(rows):
    actual = realized_events(rows)
    scores = {}
    for event in actual:
        col = "expected_red_any" if event == "red_any" else "p_" + event
        if col not in rows:
            continue
        known = actual[event].notna() & rows[col].notna()
        y, p = actual.loc[known, event].astype(float), rows.loc[known, col]
        if len(y):
            scores[event] = {"matches": len(y), "actual_rate": float(y.mean()),
                             "mean_probability": float(p.mean()), "brier": float(((p - y) ** 2).mean())}
    for event in ("yellow_over_3.5", "red_any"):
        if event in scores:
            base = rows["baseline_red_any"] if event == "red_any" else poisson.sf(3, rows["baseline_yellow_cards"])
            forecast_col = "expected_red_any" if event == "red_any" else "p_" + event
            known = actual[event].notna() & rows[forecast_col].notna()
            scores[event]["historical_rate_baseline_brier"] = float(
                ((actual[event].astype(float) - base).loc[known] ** 2).mean())
    for cols, key in [(("p_home", "p_draw", "p_away"), "top_1x2"),
                      (("p_1X", "p_X2"), "top_draw_cover")]:
        names = [c[2:] for c in cols]
        choice = rows[list(cols)].to_numpy().argmax(axis=1)
        scores[key] = float(actual[names].to_numpy()[np.arange(len(rows)), choice].mean())
    return {"matches": len(rows), "exact_score_top_pick_accuracy": float(rows.modal_exact_hit.mean()),
            "scoreline_log_loss": float(-np.log(np.clip(rows.p_exact_score, 1e-15, 1)).mean()),
            "total_goals_mae": float((rows.expected_home_goals + rows.expected_away_goals - rows.actual_total).abs().mean()),
            "yellow_cards_mae": float((rows.expected_yellow_cards - rows.actual_yellow_cards).abs().mean())
            if "expected_yellow_cards" in rows and rows.actual_yellow_cards.notna().any() else None,
            "event_metrics": scores}


def historical_prices(rows):
    outcomes = realized_events(rows)
    summaries, signals = {}, []
    for group, (names, columns) in PRICE_GROUPS.items():
        quoted = rows.dropna(subset=list(columns))
        if quoted.empty:
            continue
        prices = quoted[list(columns)].to_numpy()
        probabilities = quoted[["p_" + n for n in names]].to_numpy()
        actual = outcomes.loc[quoted.index, list(names)].to_numpy(dtype=float)
        implied = 1 / prices
        implied /= implied.sum(axis=1, keepdims=True)
        summary = {"priced_fixtures": len(quoted),
                   "model_brier_sum": float(np.square(probabilities - actual).sum(axis=1).mean()),
                   "normalized_odds_brier_sum": float(np.square(implied - actual).sum(axis=1).mean())}
        if group.startswith("bet365"):
            results = []
            for i, (_, row) in enumerate(quoted.iterrows()):
                eligible = [j for j, name in enumerate(names)
                            if eligible_price(name, probabilities[i, j], prices[i, j])]
                if not eligible:
                    continue
                j = max(eligible, key=lambda j: probabilities[i, j] * prices[i, j])
                net = prices[i, j] * actual[i, j] - 1
                results.append(net)
                signals.append({"date": row.Date.date().isoformat(), "home": row.HomeTeam,
                                "away": row.AwayTeam, "group": group, "selection": names[j],
                                "decimal_odds": prices[i, j], "probability": probabilities[i, j],
                                "won": bool(actual[i, j]), "paper_net_units": net})
            summary.update(paper_selections=len(results), paper_net_units=float(sum(results)),
                           paper_return=float(np.mean(results)) if results else None,
                           paper_hit_rate=float(np.mean(np.asarray(results) > 0)) if results else None,
                           priced_fixture_coverage=len(results) / len(quoted))
        summaries[group] = summary
    return summaries, signals


def benchmark_markets(matches, provenance, out):
    features, teams, meetings = market_features(matches)
    outcomes = targets(matches)
    folds = seasonal_folds(matches)
    frames = []
    for fold in folds:
        training = fold["fit"] | fold["cal"]
        if matches.loc[training, "Date"].max() >= matches.loc[fold["test"], "Date"].min():
            raise ValueError("Training seasons overlap the test period")
        model = fit_heads(features, outcomes, matches.Date, training)
        frames.append(historical_rows(matches, outcomes, model, features, fold["test"], fold["season"]))
    rows = pd.concat(frames)
    latest = frames[-1]
    prices, signals = historical_prices(rows)
    latest_prices, _ = historical_prices(latest)
    report = {"source_matches": len(matches), "from": matches.Date.min().date().isoformat(),
              "through": matches.Date.max().date().isoformat(),
              "all_walk_forward": summarize(rows), "latest_season": summarize(latest),
              "season_metrics": {frame.season_label.iloc[0]: summarize(frame) for frame in frames},
              "historical_prices": prices, "latest_season_prices": latest_prices,
              "paper_filter": {"minimum_estimated_return": EDGE_FLOOR, "minimum_win_probability": WIN_FLOOR,
                               "one_selection_per_fixture_per_priced_market_family": True},
              "price_coverage": {"double_chance": 0, "correct_score": 0, "btts": 0, "cards": 0},
              "evaluation": "Fixed models refit before each season using preceding dates. Prior completed fixtures update features within each test season. Dates and some results were inspected in earlier work.",
              "price_limits": "Archived Bet365 quotes and earlier archive averages. Quote timestamps are unaudited. Only Bet365 quotes enter unit simulations. Betway quotes enter through manual input.",
              "card_limits": "Targets are archived yellow counts and any recorded red. Betway card settlement equivalence and historical prices remain unverified."}
    model = fit_heads(features, outcomes, matches.Date, pd.Series(True, index=matches.index))
    model.update(schema="epl_markets_v1", history_through=report["through"], teams=teams, meetings=meetings,
                 created_utc=datetime.now(timezone.utc).isoformat(), data_fingerprints=provenance)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [("model.json", model), ("report.json", report)]:
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    export = ["Date", "HomeTeam", "AwayTeam", "season_label", "FTHG", "FTAG", "actual_total",
              "actual_yellow_cards", "actual_red_any", "p_exact_score", "modal_exact_hit"]
    export += [c for c in rows if c.startswith(("p_", "expected_")) and c not in export]
    rows[export].to_csv(out / "forecasts.csv", index=False)
    columns = ["date", "home", "away", "group", "selection", "decimal_odds", "probability", "won", "paper_net_units"]
    pd.DataFrame(signals, columns=columns).to_csv(out / "paper_price_checks.csv", index=False)
    return report

"""Features use strictly earlier dates. All same-day rows are frozen first."""
from copy import deepcopy
from datetime import date
import numpy as np
import pandas as pd
from .availability import AVAILABILITY_FEATURES, availability_values

FEATURES = ["elo_difference", "recent_points_difference", "recent_goals_for_difference",
            "recent_goals_against_difference", "league_rest_days_difference",
            "home_recent_games", "away_recent_games", "home_venue_points", "away_venue_points"]
SHOT_FEATURES = ["recent_shots_difference", "recent_target_shots_difference",
                 "home_shots_observed", "away_shots_observed",
                 "home_target_shots_observed", "away_target_shots_observed"]
POSSESSION_FEATURES = ["recent_possession_difference", "home_possession_observed", "away_possession_observed"]
RESEARCH_FEATURES = FEATURES + SHOT_FEATURES
FORM_METRICS = ["goals_for_10", "goals_against_10", "points_10", "points_20", "draw_rate_10",
                "clean_sheet_rate_10", "shots_10", "target_shots_10", "opponent_target_shots_10",
                "yellows_5", "reds_5", "previous_reds", "corners_5", "fouls_5", "discipline_observed_5",
                "venue_goals_for_10", "venue_goals_against_10", "venue_points_10", "points_trend_3_vs_10",
                "winless_streak_20", "loss_streak_20", "absolute_elo"]
ADVANCED_FEATURES = RESEARCH_FEATURES + [f"{side}_{metric}" for side in ["home", "away"] for metric in FORM_METRICS]


def supported_feature_sets():
    return [FEATURES, RESEARCH_FEATURES, RESEARCH_FEATURES + POSSESSION_FEATURES] + [
        ADVANCED_FEATURES + poss + avail for poss in [[], POSSESSION_FEATURES] for avail in [[], AVAILABILITY_FEATURES]]


def empty_team():
    return {"elo": 1500.0, "recent": [], "home": [], "away": [], "last_date": None,
            "shots": [], "possession": [], "long_history": []}


def _view(state, team, when):
    item = deepcopy(state.get(team, empty_team()))
    item.setdefault("shots", [])
    item.setdefault("possession", [])
    item.setdefault("long_history", [])
    if item["last_date"]:
        gap = (date.fromisoformat(str(when)[:10]) - date.fromisoformat(item["last_date"])).days
        if gap <= 0:
            raise ValueError("Feature history must end before the fixture date")
        if gap > 90:
            # A fixed, deliberately simple offseason/inactivity rule.
            item["elo"] = 1500 + 0.75 * (item["elo"] - 1500)
            item["recent"], item["home"], item["away"] = [], [], []
            item["shots"], item["possession"] = [], []
        if gap > 365:
            item["long_history"] = []
    else:
        gap = 14
    return item, min(gap, 14)


def _mean(rows, index, default):
    return float(np.mean([r[index] for r in rows])) if rows else default


def fixture_features(state, home, away, when, feature_names=None, known_at=None, availability=None):
    h, hr = _view(state, home, when)
    a, ar = _view(state, away, when)
    values = [
        (h["elo"] - a["elo"]) / 100,
        _mean(h["recent"], 0, 1.3) - _mean(a["recent"], 0, 1.3),
        _mean(h["recent"], 1, 1.3) - _mean(a["recent"], 1, 1.3),
        _mean(h["recent"], 2, 1.3) - _mean(a["recent"], 2, 1.3),
        hr - ar, len(h["recent"]), len(a["recent"]),
        _mean(h["home"], 0, 1.5), _mean(a["away"], 0, 1.1),
    ]
    names = feature_names or FEATURES
    if names not in supported_feature_sets():
        raise ValueError("Unknown feature set")
    if names != FEATURES:
        def observed(item, index):
            return [r[index] for r in item["shots"] if r[index] is not None]
        hs, ats = observed(h, 0), observed(a, 0)
        ht, at = observed(h, 1), observed(a, 1)
        mean = lambda v, default=0: float(np.mean(v)) if v else default
        values += [mean(hs) - mean(ats), mean(ht) - mean(at), len(hs), len(ats), len(ht), len(at)]
        if names[:len(ADVANCED_FEATURES)] == ADVANCED_FEATURES:
            def form(item, venue):
                rows = item["long_history"]
                ten, five, three = rows[-10:], rows[-5:], rows[-3:]
                venue_rows = [r for r in rows if r["venue"] == venue][-10:]
                def avg(data, key, default):
                    valid = [r[key] for r in data if r.get(key) is not None]
                    return float(np.mean(valid)) if valid else default
                def streak(condition):
                    n = 0
                    for r in reversed(rows):
                        if not condition(r): break
                        n += 1
                    return n
                return [avg(ten, "gf", 1.35), avg(ten, "ga", 1.35), avg(ten, "points", 1.3), avg(rows, "points", 1.3),
                        float(np.mean([r["points"] == 1 for r in ten])) if ten else 0.25,
                        float(np.mean([r["ga"] == 0 for r in ten])) if ten else 0.25,
                        avg(ten, "shots", 12), avg(ten, "target_shots", 4), avg(ten, "opponent_target_shots", 4),
                        avg(five, "yellow", 2), avg(five, "red", 0.08),
                        float(rows[-1]["red"]) if rows and rows[-1].get("red") is not None else 0,
                        avg(five, "corners", 5), avg(five, "fouls", 11),
                        sum(r.get("red") is not None and r.get("yellow") is not None for r in five),
                        avg(venue_rows, "gf", 1.35), avg(venue_rows, "ga", 1.35), avg(venue_rows, "points", 1.3),
                        avg(three, "points", 1.3) - avg(ten, "points", 1.3),
                        streak(lambda r: r["points"] < 3), streak(lambda r: r["points"] == 0), item["elo"] / 100]
            values += form(h, "home") + form(a, "away")
        if "recent_possession_difference" in names:
            cutoff = pd.Timestamp(str(when)[:10]).tz_localize("Europe/London").tz_convert("UTC")
            if known_at is not None:
                cutoff = min(cutoff, pd.Timestamp(known_at))
            def known(item):
                return [r[0] for r in item["possession"] if pd.Timestamp(r[1]) < cutoff][-5:]
            hp, ap = known(h), known(a)
            values += [mean(hp, 50) - mean(ap, 50), len(hp), len(ap)]
        if "home_injured" in names:
            values += availability_values(availability or {}, when, known_at)
    return np.array(values, dtype=float)


def update(state, row):
    when = row.Date.strftime("%Y-%m-%d")
    h, _ = _view(state, row.HomeTeam, when)
    a, _ = _view(state, row.AwayTeam, when)
    expected = 1 / (1 + 10 ** ((a["elo"] - h["elo"] - 60) / 400))
    score = 1.0 if row.FTR == "H" else 0.5 if row.FTR == "D" else 0.0
    change = 20 * (score - expected)
    h["elo"] += change
    a["elo"] -= change
    hp, ap = (3, 0) if row.FTR == "H" else (1, 1) if row.FTR == "D" else (0, 3)
    for team, pts, gf, ga, venue in [(h, hp, row.FTHG, row.FTAG, "home"),
                                    (a, ap, row.FTAG, row.FTHG, "away")]:
        record = [int(pts), int(gf), int(ga)]
        team["recent"] = (team["recent"] + [record])[-5:]
        team[venue] = (team[venue] + [record])[-5:]
        team["last_date"] = when
    for team, total, target in [(h, "HS", "HST"), (a, "AS", "AST")]:
        def optional(col):
            value = getattr(row, col, None)
            return float(value) if value is not None and pd.notna(value) else None
        team["shots"] = (team["shots"] + [[optional(total), optional(target)]])[-5:]
    for team, prefix, other, points, gf, ga, venue in [
            (h, "H", "A", hp, row.FTHG, row.FTAG, "home"),
            (a, "A", "H", ap, row.FTAG, row.FTHG, "away")]:
        record = {"points": int(points), "gf": int(gf), "ga": int(ga), "venue": venue,
                  "shots": optional(prefix + "S"), "target_shots": optional(prefix + "ST"),
                  "opponent_target_shots": optional(other + "ST"), "yellow": optional(prefix + "Y"),
                  "red": optional(prefix + "R"), "corners": optional(prefix + "C"), "fouls": optional(prefix + "F")}
        team["long_history"] = (team["long_history"] + [record])[-20:]
    available = getattr(row, "AvailableAt", None)
    if available is not None and pd.notna(available):
        for team, col in [(h, "HomePossession"), (a, "AwayPossession")]:
            value = getattr(row, col, None)
            if value is not None and pd.notna(value):
                team["possession"].append([float(value), str(available)])
    state[row.HomeTeam], state[row.AwayTeam] = h, a


def build_features(matches, feature_names=None):
    if not matches.Date.is_monotonic_increasing:
        raise ValueError("Sort match history by date before building features")
    state, rows = {}, []
    names = feature_names or FEATURES
    for _, day in matches.groupby("Date", sort=True):
        batch = list(day.itertuples(index=False))
        rows.extend(fixture_features(state, r.HomeTeam, r.AwayTeam, r.Date.strftime("%Y-%m-%d"), names,
                                     availability=r._asdict())
                    for r in batch)
        for row in batch:
            update(state, row)
    return pd.DataFrame(rows, columns=names, index=matches.index), state

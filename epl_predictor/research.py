"""Fixed candidate comparison with seasonal walk-forward testing.

Model choice uses development seasons only. The final season is a separate
audit of that choice. The latest paper model is a labelled refit, not a test.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler

from .features import ADVANCED_FEATURES, POSSESSION_FEATURES, build_features
from .availability import AVAILABILITY_FEATURES
from .model import LABELS, metrics, probabilities, reliability

XGB_CONFIG = {"n_estimators": 180, "max_depth": 2, "learning_rate": 0.035,
              "min_child_weight": 8, "subsample": 0.9, "colsample_bytree": 1.0,
              "reg_lambda": 5.0, "objective": "multi:softprob", "eval_metric": "mlogloss",
              "random_state": 42, "n_jobs": 2, "tree_method": "hist", "device": "cpu"}
CANDIDATES = ["training_frequency", "multinomial_logistic_regression", "independent_poisson", "xgboost",
              "recent_logistic", "recent_poisson", "recent_xgboost"]
RECENT_BASE = {"recent_logistic": "multinomial_logistic_regression", "recent_poisson": "independent_poisson", "recent_xgboost": "xgboost"}
CALIBRATION_MATCHES = 190


def season_labels(matches):
    if "Season" in matches:
        return matches.Season.astype(int)
    return matches.Date.dt.year - matches.Date.dt.month.lt(7).astype(int)


def seasonal_folds(matches, min_train_seasons=2):
    labels = season_labels(matches)
    seasons = sorted(labels.unique())
    if len(seasons) <= min_train_seasons:
        raise ValueError("Need at least three seasons for development and four seasons for a separate final audit")
    folds = []
    for season in seasons[min_train_seasons:]:
        prior = matches.loc[labels < season]
        split = prior.Date.iloc[max(0, len(prior) - CALIBRATION_MATCHES)]
        fit = (labels < season) & (matches.Date < split)
        cal = (labels < season) & (matches.Date >= split)
        test = labels == season
        if fit.sum() < 400 or cal.sum() < 50:
            raise ValueError("A seasonal fold has insufficient training or calibration data")
        if set(matches.loc[fit, "FTR"]) != set(LABELS) or set(matches.loc[cal, "FTR"]) != set(LABELS):
            raise ValueError("Each training/calibration fold needs all three outcomes")
        folds.append({"season": f"{season}/{str(season + 1)[-2:]}", "fit": fit, "cal": cal, "test": test})
    return folds


def fit_candidate(kind, matches, x, fit, cal, provenance):
    if not fit.any() or not cal.any() or matches.loc[fit, "Date"].max() >= matches.loc[cal, "Date"].min():
        raise ValueError("Training must end strictly before calibration")
    if set(matches.loc[fit, "FTR"]) != set(LABELS):
        raise ValueError("Training requires H, D and A examples")
    candidate_name = kind
    weighted = kind in RECENT_BASE
    kind = RECENT_BASE.get(kind, kind)
    artifact = {"schema_version": 3, "model_type": kind, "candidate_name": candidate_name, "temperature": 1.0,
                "features": list(x.columns), "classes": LABELS,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "parameters_fit_through": matches.loc[fit, "Date"].max().strftime("%Y-%m-%d"),
                "calibrated_through": matches.loc[cal, "Date"].max().strftime("%Y-%m-%d"),
                "data_fingerprints": provenance}
    y = matches.loc[fit, "FTR"]
    weights = None
    if weighted:
        age = (matches.loc[fit, "Date"].max() - matches.loc[fit, "Date"]).dt.days.to_numpy()
        weights = np.exp2(-age / 1095.75)
        weights /= weights.mean()
    artifact["recency_half_life_days"] = 1095.75 if weighted else None
    artifact["availability_training_rows"] = int(matches.loc[fit, ["HomeKnownAt", "AwayKnownAt"]].notna().all(axis=1).sum()) if "HomeKnownAt" in matches else 0
    if kind == "training_frequency":
        artifact["prior"] = [(int(y.eq(k).sum()) + 1) / (len(y) + 3) for k in LABELS]
        return artifact
    if kind == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ValueError('Install the comparison extra: python -m pip install -e ".[boost]"') from exc
        learner = xgb.XGBClassifier(**XGB_CONFIG)
        encoded = np.array([LABELS.index(k) for k in y])
        learner.fit(x.loc[fit].to_numpy(), encoded, sample_weight=weights)
        artifact["booster"] = json.loads(learner.get_booster().save_raw(raw_format="json"))
        artifact["config"] = XGB_CONFIG
    else:
        scaler = StandardScaler().fit(x.loc[fit])
        artifact.update(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist())
        train_x = scaler.transform(x.loc[fit])
        if kind == "multinomial_logistic_regression":
            learner = LogisticRegression(C=1.0, max_iter=2000).fit(train_x, y, sample_weight=weights)
            artifact.update(coefficients=learner.coef_.tolist(), intercept=learner.intercept_.tolist(),
                            classes=learner.classes_.tolist(), config={"C": 1.0, "max_iter": 2000})
        elif kind == "independent_poisson":
            models = [PoissonRegressor(alpha=1.0, max_iter=2000).fit(train_x, matches.loc[fit, target], sample_weight=weights)
                      for target in ["FTHG", "FTAG"]]
            artifact.update(goal_coefficients=[m.coef_.tolist() for m in models],
                            goal_intercepts=[float(m.intercept_) for m in models],
                            config={"alpha": 1.0, "rate_floor": 0.05, "rate_cap": 6.0})
        else:
            raise ValueError(f"Unknown model: {kind}")
    raw = probabilities(artifact, x.loc[cal])
    log_p = np.log(np.clip(raw, 1e-12, 1))
    def loss(t):
        return metrics(matches.loc[cal, "FTR"], softmax(log_p / t, axis=1))["log_loss"]
    opt = minimize_scalar(loss, bounds=(0.5, 3.0), method="bounded")
    artifact["temperature"] = float(opt.x) if opt.success and loss(opt.x) < loss(1.0) else 1.0
    return artifact


def choose_candidate(development):
    # Deliberately accepts no audit/final-season argument.
    return min(development, key=lambda kind: (development[kind]["log_loss"], kind))


def benchmark(matches, provenance, out_dir, include_xgboost=True):
    kinds = CANDIDATES if include_xgboost else [k for k in CANDIDATES if "xgboost" not in k]
    folds = seasonal_folds(matches)
    if len(folds) < 2:
        raise ValueError("Use at least four seasons: two initial training seasons, a development season and an audit season")
    names = ADVANCED_FEATURES + (POSSESSION_FEATURES if "HomePossession" in matches and matches.HomePossession.notna().any() else [])
    if "HomeKnownAt" in matches and matches[["HomeKnownAt", "AwayKnownAt"]].notna().any().any():
        names += AVAILABILITY_FEATURES
    x, state = build_features(matches, names)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fold_reports, prediction_rows, last_artifacts = [], [], {}
    for i, fold in enumerate(folds):
        fit, cal, test = fold["fit"], fold["cal"], fold["test"]
        info = {"season": fold["season"], "role": "audit" if i == len(folds) - 1 else "development",
                "train_matches": int(fit.sum()), "calibration_matches": int(cal.sum()), "test_matches": int(test.sum()),
                "train_end": matches.loc[fit, "Date"].max().strftime("%Y-%m-%d"),
                "calibration_start": matches.loc[cal, "Date"].min().strftime("%Y-%m-%d"),
                "calibration_end": matches.loc[cal, "Date"].max().strftime("%Y-%m-%d"),
                "test_start": matches.loc[test, "Date"].min().strftime("%Y-%m-%d"),
                "test_end": matches.loc[test, "Date"].max().strftime("%Y-%m-%d"), "models": {}}
        for kind in kinds:
            artifact = fit_candidate(kind, matches, x, fit, cal, provenance)
            p = probabilities(artifact, x.loc[test])
            info["models"][kind] = metrics(matches.loc[test, "FTR"], p)
            frame = matches.loc[test, ["Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
            frame[["p_home", "p_draw", "p_away"]] = p
            frame["prediction"] = np.array(LABELS)[p.argmax(axis=1)]
            frame["model"], frame["season"], frame["role"] = kind, info["season"], info["role"]
            prediction_rows.append(frame)
            if info["role"] == "audit":
                artifact.update(teams=state, history_through=info["test_end"], purpose="frozen_audit_parameters_with_current_team_state")
                last_artifacts[kind] = artifact
        fold_reports.append(info)
        print(f"Evaluated {info['season']} ({info['role']}): {info['test_matches']} matches", flush=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    dev, audit, selection = {}, {}, {}
    selection_seasons = [f["season"] for f in fold_reports if f["role"] == "development"][-3:]
    for kind in kinds:
        for role, collection in [("development", dev), ("audit", audit)]:
            rows = predictions.loc[predictions.model.eq(kind) & predictions.role.eq(role)]
            collection[kind] = metrics(rows.FTR, rows[["p_home", "p_draw", "p_away"]].to_numpy())
        recent = predictions.loc[predictions.model.eq(kind) & predictions.season.isin(selection_seasons)]
        selection[kind] = metrics(recent.FTR, recent[["p_home", "p_draw", "p_away"]].to_numpy())
    selected = choose_candidate(selection)
    # A latest refit is distinct from all the frozen historical evaluation models.
    split = matches.Date.iloc[max(0, len(matches) - CALIBRATION_MATCHES)]
    latest = fit_candidate(selected, matches, x, matches.Date < split, matches.Date >= split, provenance)
    latest.update(teams=state, history_through=matches.Date.max().strftime("%Y-%m-%d"),
                  purpose="latest_refit_for_future_paper_trial_not_out_of_sample_tested",
                  selection_rule="lowest pooled log loss on up to three most recent development seasons; audit excluded")
    report = {"version": "0.3.0", "created_utc": datetime.now(timezone.utc).isoformat(),
              "matches_total": len(matches), "data_start": matches.Date.min().strftime("%Y-%m-%d"),
              "data_end": latest["history_through"], "features": names,
              "coverage": {"shots_rows": int(matches[["HS", "AS", "HST", "AST"]].notna().all(axis=1).sum()) if set(["HS", "AS", "HST", "AST"]).issubset(matches) else 0,
                           "possession_rows": int(matches.HomePossession.notna().sum()) if "HomePossession" in matches else 0,
                           "discipline_rows": int(matches[["HY", "AY", "HR", "AR"]].notna().all(axis=1).sum()) if set(["HY", "AY", "HR", "AR"]).issubset(matches) else 0,
                           "availability_paired_rows": int(matches[["HomeKnownAt", "AwayKnownAt"]].notna().all(axis=1).sum()) if "HomeKnownAt" in matches else 0},
              "folds": fold_reports, "development": dev, "audit": audit,
              "selected_model": selected, "selection_rule": latest["selection_rule"],
              "selection_seasons": selection_seasons, "selection_metrics": selection,
              "calibration_matches_requested": CALIBRATION_MATCHES,
              "latest_paper_model": {k: latest[k] for k in ["purpose", "parameters_fit_through", "calibrated_through", "history_through"]},
              "data_fingerprints": provenance,
              "data_quality": {"quarantined_stat_pairs": sum(len(p.get("quarantined_stat_pairs", [])) for p in provenance),
                               "results_discarded_by_quarantine": 0},
              "limitations": ["Historical exploratory research; portions of this data were already viewed in v0.1, so the final audit is not a pristine external holdout.",
                              "Latest season can be partial. Source season identity preserves the July 2020 conclusion of 2019/20; generic files need explicit Season for exceptions.",
                              "Test parameters are frozen per season; team state updates after each earlier match date.",
                              "Fixed candidate configurations; no tuning against audit results.",
                              "Possession is absent unless explicitly supplied with availability timestamps; it is never fabricated.",
                              "Shots are previous-match stats, assumed available by the next match date. Publication/revision timing is not audited.",
                              "Injuries/suspensions are included only when pre-match historical snapshots are supplied; missing availability is never called healthy.",
                              "Prior yellow/red cards describe past discipline, not known future suspensions. No player-level ban inference is made.",
                              "No live odds or profitability evaluation. No bookmaker edge established.",
                              "Latest paper model is refit on newer history and has no independent measured performance yet."]}
    audit_kind = predictions.loc[predictions.model.eq(selected) & predictions.role.eq("audit")]
    report["selected_audit_reliability"] = reliability(audit_kind.FTR, audit_kind[["p_home", "p_draw", "p_away"]].to_numpy())
    report["confidence_subsets"] = confidence_subsets(audit_kind)
    report["target_accuracy"] = {"goals": [0.55, 0.60], "audit_accuracy": audit[selected]["accuracy"],
                                 "at_least_55_percent_on_all_audit_matches": audit[selected]["accuracy"] >= 0.55,
                                 "at_least_60_percent_on_all_audit_matches": audit[selected]["accuracy"] >= 0.60}
    report["historical_bookmaker_benchmark"] = bookmaker_benchmark(matches, folds[-1]["test"])
    (out / "model.json").write_text(json.dumps(latest, indent=2, allow_nan=False), encoding="utf-8")
    (out / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    predictions.to_csv(out / "walk_forward_predictions.csv", index=False)
    audit_dir = out / "audit_models"
    audit_dir.mkdir(exist_ok=True)
    for kind, artifact in last_artifacts.items():
        (audit_dir / f"{kind}.json").write_text(json.dumps(artifact, allow_nan=False), encoding="utf-8")
    return report


def confidence_subsets(rows):
    p = rows[["p_home", "p_draw", "p_away"]].to_numpy()
    result = []
    for threshold in [0.5, 0.6, 0.7]:
        mask = p.max(axis=1) >= threshold
        n = int(mask.sum())
        summary = {"threshold": threshold, "matches": n, "total_matches": len(rows), "coverage": n / len(rows)}
        if n:
            summary["metrics"] = metrics(rows.loc[mask, "FTR"], p[mask])
            rate = summary["metrics"]["accuracy"]
            z = 1.96
            centre = (rate + z*z/(2*n)) / (1 + z*z/n)
            radius = z * np.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1 + z*z/n)
            summary["accuracy_wilson_95_interval"] = [float(centre-radius), float(centre+radius)]
        result.append(summary)
    return result


def bookmaker_benchmark(matches, test):
    cols = ["B365H", "B365D", "B365A"]
    if not set(cols).issubset(matches):
        return {"matches": 0, "reason": "No stored pre-closing bookmaker columns"}
    subset = matches.loc[test].dropna(subset=cols)
    if subset.empty:
        return {"matches": 0, "reason": "No complete pre-closing odds rows"}
    implied = 1 / subset[cols].to_numpy()
    normalized = implied / implied.sum(axis=1, keepdims=True)
    return {"source": "Historical B365H/B365D/B365A", "metrics": metrics(subset.FTR, normalized),
            "limitation": "Recorded pre-closing odds, with unaudited exact forecast-time availability. Comparator only; excluded from model inputs and selection. No ROI claim."}

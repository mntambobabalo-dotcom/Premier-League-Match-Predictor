"""Chronological training, temperature calibration, safe JSON inference."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, RESEARCH_FEATURES, POSSESSION_FEATURES, build_features, supported_feature_sets

LABELS = ["H", "D", "A"]


def probabilities(artifact, features):
    x = np.atleast_2d(np.asarray(features, dtype=float))
    kind = artifact["model_type"]
    if kind == "training_frequency":
        return np.tile(artifact["prior"], (len(x), 1))
    if kind == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ValueError('XGBoost model requires: python -m pip install -e ".[boost]"') from exc
        booster = xgb.Booster(params={"nthread": 2})
        booster.load_model(bytearray(json.dumps(artifact["booster"]).encode("utf-8")))
        logits = np.asarray(booster.predict(xgb.DMatrix(x), output_margin=True), dtype=float)
        return softmax(logits / artifact["temperature"], axis=1)
    z = (x - np.array(artifact["mean"])) / np.array(artifact["scale"])
    if kind == "independent_poisson":
        from scipy.stats import skellam
        rates = np.exp(np.clip(z @ np.array(artifact["goal_coefficients"]).T +
                               np.array(artifact["goal_intercepts"]), np.log(0.05), np.log(6.0)))
        home, away = rates[:, 0], rates[:, 1]
        p_away = skellam.cdf(-1, home, away)
        p_draw = skellam.pmf(0, home, away)
        p_home = 1 - skellam.cdf(0, home, away)
        p = np.column_stack([p_home, p_draw, p_away])
        return softmax(np.log(np.clip(p, 1e-12, 1)) / artifact["temperature"], axis=1)
    logits = z @ np.array(artifact["coefficients"]).T + np.array(artifact["intercept"])
    p = softmax(logits / artifact["temperature"], axis=1)
    return p[:, [artifact["classes"].index(k) for k in LABELS]]


def metrics(y, p):
    p = np.asarray(p, dtype=float)
    if p.shape != (len(y), 3) or not len(y) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Expected a nonempty H,D,A probability matrix")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Each probability row must sum to one")
    p = p / p.sum(axis=1, keepdims=True)
    target = np.array([[float(v == k) for k in LABELS] for v in y])
    return {"matches": len(y), "accuracy": float(accuracy_score(y, np.array(LABELS)[p.argmax(axis=1)])),
            "log_loss": float(log_loss(y, p[:, [2, 1, 0]], labels=["A", "D", "H"])),
            "multiclass_brier_sum": float(np.mean(np.sum((p - target) ** 2, axis=1))),
            "confusion_matrix_actual_rows_predicted_columns_H_D_A": confusion_matrix(
                y, np.array(LABELS)[p.argmax(axis=1)], labels=LABELS).tolist()}


def reliability(y, p):
    rows = []
    for i, label in enumerate(LABELS):
        for low in np.arange(0, 1, 0.1):
            mask = (p[:, i] >= low) & (p[:, i] < low + 0.1 if low < 0.9 else p[:, i] <= 1)
            if mask.any():
                rows.append({"outcome": label, "lower": float(low), "count": int(mask.sum()),
                             "mean_probability": float(p[mask, i].mean()),
                             "observed_frequency": float(np.mean(np.asarray(y)[mask] == label))})
    return rows


def train(matches, provenance, out_dir):
    dates = matches.Date.drop_duplicates().sort_values().tolist()
    if len(matches) < 400 or len(dates) < 100:
        raise ValueError("Use at least 400 matches and 100 distinct match dates; several seasons preferred")
    train_end, calibration_end = dates[int(len(dates) * 0.6)], dates[int(len(dates) * 0.8)]
    fit = matches.Date < train_end
    cal = (matches.Date >= train_end) & (matches.Date < calibration_end)
    test = matches.Date >= calibration_end
    if set(matches.loc[fit, "FTR"]) != set(LABELS) or set(matches.loc[cal, "FTR"]) != set(LABELS):
        raise ValueError("Training and calibration periods must each contain H, D and A")
    x, state = build_features(matches)
    scaler = StandardScaler().fit(x.loc[fit])
    model = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(x.loc[fit]), matches.loc[fit, "FTR"])
    artifact = {
        "schema_version": 1, "model_type": "multinomial_logistic_regression",
        "created_utc": datetime.now(timezone.utc).isoformat(), "features": FEATURES, "classes": model.classes_.tolist(),
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coefficients": model.coef_.tolist(), "intercept": model.intercept_.tolist(), "temperature": 1.0,
        "parameters_fit_through": matches.loc[fit, "Date"].max().strftime("%Y-%m-%d"),
        "calibrated_through": matches.loc[cal, "Date"].max().strftime("%Y-%m-%d"),
        "history_through": matches.Date.max().strftime("%Y-%m-%d"), "teams": state,
        "data_fingerprints": provenance,
    }
    cal_logits = model.decision_function(scaler.transform(x.loc[cal]))
    order = [model.classes_.tolist().index(k) for k in ["A", "D", "H"]]
    def loss(t):
        return log_loss(matches.loc[cal, "FTR"], softmax(cal_logits / t, axis=1)[:, order], labels=["A", "D", "H"])
    opt = minimize_scalar(loss, bounds=(0.5, 3.0), method="bounded")
    artifact["temperature"] = float(opt.x) if opt.success and loss(opt.x) < loss(1.0) else 1.0
    p = probabilities(artifact, x.loc[test])
    prior = np.array([(matches.loc[fit, "FTR"].eq(k).sum() + 1) / (fit.sum() + 3) for k in LABELS])
    baseline = np.tile(prior, (int(test.sum()), 1))
    report = {
        "version": "0.1.0", "matches_total": len(matches),
        "data_start": matches.Date.min().strftime("%Y-%m-%d"), "data_end": artifact["history_through"],
        "split": {"method": "60/20/20 distinct dates; chronological, same-day groups kept together",
                  "train_matches": int(fit.sum()), "calibration_matches": int(cal.sum()), "test_matches": int(test.sum()),
                  "train_end": artifact["parameters_fit_through"], "calibration_end": artifact["calibrated_through"],
                  "test_start": calibration_end.strftime("%Y-%m-%d"), "test_end": artifact["history_through"]},
        "model": metrics(matches.loc[test, "FTR"], p),
        "training_frequency_baseline": metrics(matches.loc[test, "FTR"], baseline),
        "always_home_accuracy": float(matches.loc[test, "FTR"].eq("H").mean()),
        "temperature": artifact["temperature"], "reliability": reliability(matches.loc[test, "FTR"], p),
        "interpretation": "Lower log loss and Brier are better. Accuracy alone does not establish betting value.",
        "limitations": ["One chronological holdout; not an independent live trial or a profitability backtest.",
                        "Earlier test results update team state for later test dates, never model parameters.",
                        "No injuries, lineups, xG, odds, other competitions or news included.",
                        "Rest means days since last league game, capped at 14; not actual all-competition rest.",
                        "Calibration estimates are uncertain; class bins can be small.",
                        "The saved model retains the evaluated coefficients; it is not refit on the test set.",
                        "New teams cannot be predicted until supported in the model history."]}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    results = matches.loc[test, ["Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
    results[["p_home", "p_draw", "p_away"]] = p
    results["prediction"] = np.array(LABELS)[p.argmax(axis=1)]
    results.to_csv(out / "backtest.csv", index=False)
    return report


def read_model(path):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    valid = supported_feature_sets()
    if artifact.get("schema_version") not in [1, 2, 3] or artifact.get("features") not in valid:
        raise ValueError("Unsupported model schema or feature order")
    if artifact.get("model_type") not in ["training_frequency", "multinomial_logistic_regression", "xgboost", "independent_poisson"]:
        raise ValueError("Unsupported model type")
    return artifact

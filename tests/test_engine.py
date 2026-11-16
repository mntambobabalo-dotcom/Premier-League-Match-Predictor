"""Meaningful guards against leakage and misleading evaluation. No test extras needed."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from epl_predictor.data import load_matches, parse_dates
from epl_predictor.features import build_features, FEATURES
from epl_predictor.model import LABELS, metrics, probabilities, train
from epl_predictor.predict import predict, save_record, score_records


def history(n=450):
    rng = np.random.default_rng(7)
    start = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=n + 1)
    rows = []
    for i in range(n):
        h, a = ("Alpha", "Beta") if i % 2 else ("Beta", "Alpha")
        hg, ag = int(rng.poisson(1.7)), int(rng.poisson(1.2))
        result = "H" if hg > ag else "D" if hg == ag else "A"
        rows.append([start + pd.Timedelta(days=i), h, a, hg, ag, result])
    return pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])


class LeakageTests(unittest.TestCase):
    def test_current_and_future_goals_do_not_affect_current_features(self):
        df = history(80)
        before, _ = build_features(df)
        altered = df.copy()
        altered.loc[40:, ["FTHG", "FTAG", "FTR"]] = [9, 0, "H"]
        after, _ = build_features(altered)
        np.testing.assert_allclose(before.iloc[:41], after.iloc[:41])
        self.assertFalse(np.allclose(before.iloc[41:], after.iloc[41:]))

    def test_same_date_result_cannot_change_other_same_date_features(self):
        df = history(10)
        extra = df.iloc[[-1]].copy()
        extra[["HomeTeam", "AwayTeam"]] = ["Gamma", "Delta"]
        df = pd.concat([df, extra], ignore_index=True)
        before, _ = build_features(df)
        df.loc[9, ["FTHG", "FTAG", "FTR"]] = [8, 0, "H"]
        after, _ = build_features(df)
        np.testing.assert_allclose(before, after)

    def test_iso_and_day_first_dates(self):
        dates = parse_dates(pd.Series(["12/11/2025", "2025-11-12", "12/11/25"]))
        self.assertTrue(dates.eq(pd.Timestamp("2025-11-12")).all())

    def test_reject_duplicate_and_mislabeled_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matches.csv"
            df = history(10)
            pd.concat([df, df.iloc[[0]]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_matches([path])
            df.loc[0, "FTR"] = "A" if df.loc[0, "FTR"] != "A" else "H"
            df.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_matches([path])

    def test_metrics_class_order_and_brier(self):
        p = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        m = metrics(["H", "D", "A"], p)
        self.assertAlmostEqual(m["log_loss"], -np.log(0.8))
        self.assertAlmostEqual(m["multiclass_brier_sum"], 0.06)
        self.assertEqual(m["accuracy"], 1)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.df = history()
        cls.report = train(cls.df, [], cls.temp.name)
        cls.art = json.loads((Path(cls.temp.name) / "model.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_frozen_json_inference_matches_sklearn(self):
        x, _ = build_features(self.df)
        fit = self.df.Date <= pd.Timestamp(self.art["parameters_fit_through"])
        scaler = StandardScaler().fit(x.loc[fit])
        model = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(x.loc[fit]), self.df.loc[fit, "FTR"])
        a = deepcopy(self.art)
        a["temperature"] = 1.0
        expected = model.predict_proba(scaler.transform(x.iloc[-10:]))[:, [model.classes_.tolist().index(k) for k in LABELS]]
        np.testing.assert_allclose(probabilities(a, x.iloc[-10:]), expected, atol=1e-12)

    def test_test_labels_do_not_change_fitted_parameters_or_calibration(self):
        changed = self.df.copy()
        cutoff = pd.Timestamp(self.report["split"]["test_start"])
        changed.loc[changed.Date >= cutoff, ["FTHG", "FTAG", "FTR"]] = [7, 0, "H"]
        with tempfile.TemporaryDirectory() as tmp:
            train(changed, [], tmp)
            artifact = json.loads((Path(tmp) / "model.json").read_text())
        for field in ["mean", "scale", "coefficients", "intercept", "temperature"]:
            np.testing.assert_allclose(artifact[field], self.art[field])

    def test_probability_sum_and_separate_human_view(self):
        day = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
        r = predict(self.art, "Alpha", "Beta", day, [0.4, 0.3, 0.3], "Illustrative independent view")
        self.assertAlmostEqual(sum(r["model_probabilities"].values()), 1)
        self.assertEqual(r["human_probabilities"], {"H": 0.4, "D": 0.3, "A": 0.3})
        self.assertEqual(r["mode"], "paper_forecast")

    def test_reject_bad_human_inputs_and_unknown_teams(self):
        day = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
        with self.assertRaisesRegex(ValueError, "Human probabilities"):
            predict(self.art, "Alpha", "Beta", day, [0.8, 0.4, 0.1], "Reason")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            predict(self.art, "Typo FC", "Beta", day)
        with self.assertRaisesRegex(ValueError, "note"):
            predict(self.art, "Alpha", "Beta", day, [0.4, 0.3, 0.3])

    def test_reject_retroactive_history_and_stale_predictions(self):
        with self.assertRaisesRegex(ValueError, "history cutoff"):
            predict(self.art, "Alpha", "Beta", self.art["history_through"])
        day = (datetime.now(timezone.utc) + timedelta(days=60)).date().isoformat()
        with self.assertRaisesRegex(ValueError, "21 days"):
            predict(self.art, "Alpha", "Beta", day)
        r = predict(self.art, "Alpha", "Beta", day, demo=True)
        self.assertIn("demo", r["mode"])

    def test_demo_excluded_from_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            day = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
            save_record(predict(self.art, "Alpha", "Beta", day, demo=True), tmp)
            score = score_records(tmp, self.df)
            self.assertEqual(score["scored"], 0)
            self.assertEqual(score["skipped"]["demo"], 1)

    def test_journal_paired_comparison_and_duplicate_guard(self):
        when = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
        r = predict(self.art, "Alpha", "Beta", when, [0.4, 0.3, 0.3], "Independent view")
        r["created_utc"] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        settled = pd.DataFrame([[pd.Timestamp(when), "Alpha", "Beta", 2, 1, "H"]], columns=self.df.columns)
        with tempfile.TemporaryDirectory() as tmp:
            save_record(r, tmp)
            second = deepcopy(r)
            second["record_id"] = "second-record"
            save_record(second, tmp)
            score = score_records(tmp, settled)
        self.assertEqual(score["scored"], 1)
        self.assertEqual(score["paired_matches"], 1)
        self.assertEqual(score["skipped"]["duplicate_fixture"], 1)
        self.assertAlmostEqual(score["human_paired"]["log_loss"], -np.log(0.4))


if __name__ == "__main__":
    unittest.main()

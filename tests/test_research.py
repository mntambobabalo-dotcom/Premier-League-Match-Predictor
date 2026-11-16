from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import skellam

from epl_predictor.data import attach_possession, load_matches
from epl_predictor.features import build_features, fixture_features, RESEARCH_FEATURES, POSSESSION_FEATURES
from epl_predictor.model import metrics, probabilities, read_model
from epl_predictor.predict import predict
from epl_predictor.research import benchmark, choose_candidate, fit_candidate, seasonal_folds
from test_engine import history


class HistoricalStatsTests(unittest.TestCase):
    def test_current_shots_cannot_enter_current_features(self):
        df = history(60)
        df[["HS", "AS", "HST", "AST"]] = [10, 8, 4, 3]
        a, _ = build_features(df, RESEARCH_FEATURES)
        changed = df.copy()
        changed.loc[30:, ["HS", "AS", "HST", "AST"]] = [99, 1, 99, 1]
        b, _ = build_features(changed, RESEARCH_FEATURES)
        np.testing.assert_allclose(a.iloc[:31], b.iloc[:31])
        self.assertFalse(np.allclose(a.iloc[31:], b.iloc[31:]))

    def test_missing_stats_are_marked_and_not_observations(self):
        x, _ = build_features(history(20), RESEARCH_FEATURES)
        self.assertTrue(x[["home_shots_observed", "away_shots_observed"]].eq(0).all().all())

    def test_timestamped_possession_waits_until_available(self):
        df = pd.DataFrame([[pd.Timestamp(d), "Alpha", "Beta", 1, 0, "H"]
                           for d in ["2025-01-01", "2025-01-03", "2025-01-10"]],
                          columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
        observed = pd.DataFrame([["2025-01-01", "Alpha", "Beta", 60, 40, "2025-01-05T12:00:00Z"]],
                                columns=["Date", "HomeTeam", "AwayTeam", "HomePossession", "AwayPossession", "AvailableAt"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "possession.csv"
            observed.to_csv(path, index=False)
            merged, _ = attach_possession(df, path)
        x, _ = build_features(merged, RESEARCH_FEATURES + POSSESSION_FEATURES)
        self.assertEqual(x.loc[0, "home_possession_observed"], 0)
        self.assertEqual(x.loc[1, "home_possession_observed"], 0)
        self.assertEqual(x.loc[2, "home_possession_observed"], 1)
        self.assertEqual(x.loc[2, "recent_possession_difference"], 20)
        # A forecast produced earlier than the fixture must not see a later release.
        _, state = build_features(merged.iloc[:1], RESEARCH_FEATURES + POSSESSION_FEATURES)
        early = fixture_features(state, "Alpha", "Beta", "2025-01-10", RESEARCH_FEATURES + POSSESSION_FEATURES,
                                 known_at=pd.Timestamp("2025-01-03T12:00:00Z"))
        self.assertEqual(early[-2], 0)

    def test_reject_unmatched_or_invalid_possession(self):
        df = history(10)
        row = df.iloc[[0]][["Date", "HomeTeam", "AwayTeam"]].copy()
        row["HomePossession"], row["AwayPossession"] = 60, 40
        row["AvailableAt"] = row.Date.map(lambda d: d.strftime("%Y-%m-%dT22:00:00Z"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "poss.csv"
            changed = row.copy()
            changed.HomeTeam = "Unknown FC"
            changed.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "unknown matches"):
                attach_possession(df, path)
            changed = row.copy()
            changed.AwayPossession = 70
            changed.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "sum"):
                attach_possession(df, path)
            row.AvailableAt = "2026-01-01T20:00:00"
            row.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "offset"):
                attach_possession(df, path)

    def test_reject_impossible_shot_statistics(self):
        df = history(10)
        df[["HS", "AS", "HST", "AST"]] = [10, 8, 20, 3]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matches.csv"
            df.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exceed"):
                load_matches([path])


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = history(450)
        cls.x, cls.state = build_features(cls.df, RESEARCH_FEATURES)
        cls.fit = cls.df.index < 360
        cls.cal = ~cls.fit

    def test_poisson_uses_exact_goal_difference_probabilities(self):
        artifact = {"model_type": "independent_poisson", "mean": [0], "scale": [1],
                    "goal_coefficients": [[0], [0]], "goal_intercepts": [np.log(1.8), np.log(1.1)], "temperature": 1}
        p = probabilities(artifact, [[0]])[0]
        expected = [1 - skellam.cdf(0, 1.8, 1.1), skellam.pmf(0, 1.8, 1.1), skellam.cdf(-1, 1.8, 1.1)]
        np.testing.assert_allclose(p, expected, atol=1e-12)

    def test_json_roundtrip_and_inference_for_all_candidates(self):
        for kind in ["training_frequency", "multinomial_logistic_regression", "independent_poisson", "xgboost"]:
            with self.subTest(kind=kind):
                try:
                    artifact = fit_candidate(kind, self.df, self.x, self.fit, self.cal, [])
                except ValueError as e:
                    if kind == "xgboost" and "Install" in str(e):
                        continue
                    raise
                artifact.update(teams=self.state, history_through=str(self.df.Date.max().date()))
                p = probabilities(artifact, self.x.iloc[-8:])
                self.assertTrue(np.isfinite(p).all())
                np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "model.json"
                    path.write_text(json.dumps(artifact, allow_nan=False))
                    loaded = read_model(path)
                    np.testing.assert_allclose(probabilities(loaded, self.x.iloc[-8:]), p, atol=1e-12)
                    future = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    forecast = predict(loaded, "Alpha", "Beta", future)
                    self.assertEqual(forecast["model_type"], kind)

    def test_same_day_training_calibration_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly"):
            fit_candidate("independent_poisson", self.df, self.x, self.fit, self.fit, [])

    def test_float32_probabilities_are_normalized_for_metrics(self):
        p = np.array([[0.4, 0.3, 0.3]], dtype=np.float32)
        self.assertTrue(np.isfinite(metrics(["H"], p)["log_loss"]))
        with self.assertRaisesRegex(ValueError, "sum"):
            metrics(["H"], [[0.4, 0.4, 0.4]])

    def test_selection_accepts_development_scores_only(self):
        self.assertEqual(choose_candidate({"a": {"log_loss": 1.05}, "b": {"log_loss": 1.01}}), "b")


class WalkForwardTests(unittest.TestCase):
    def test_seasonal_boundaries_and_audit_cannot_change_development_choice(self):
        df = history(1500)
        folds = seasonal_folds(df)
        for f in folds:
            self.assertLess(df.loc[f["fit"], "Date"].max(), df.loc[f["cal"], "Date"].min())
            self.assertLess(df.loc[f["cal"], "Date"].max(), df.loc[f["test"], "Date"].min())
        altered = df.copy()
        altered.loc[folds[-1]["test"], ["FTHG", "FTAG", "FTR"]] = [8, 0, "H"]
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            before = benchmark(df, [], first, include_xgboost=False)
            after = benchmark(altered, [], second, include_xgboost=False)
            self.assertEqual(before["selected_model"], after["selected_model"])
            self.assertEqual(before["development"], after["development"])
            self.assertEqual(before["coverage"]["possession_rows"], 0)
            self.assertIn("not_out_of_sample_tested", before["latest_paper_model"]["purpose"])


if __name__ == "__main__":
    unittest.main()

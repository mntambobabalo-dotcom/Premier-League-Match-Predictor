from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from epl_predictor.availability import attach_availability, availability_values, read_snapshots, snapshot_context, AVAILABILITY_FEATURES
from epl_predictor.data import load_matches
from epl_predictor.downloads import download_seasons
from epl_predictor.features import ADVANCED_FEATURES, build_features, fixture_features
from epl_predictor.model import probabilities, read_model
from epl_predictor.predict import predict, save_record
from epl_predictor.research import bookmaker_benchmark, confidence_subsets, fit_candidate, season_labels
from test_engine import history


class ExtendedHistoryTests(unittest.TestCase):
    def test_cards_from_current_or_future_match_do_not_enter_features(self):
        df = history(80)
        df[["HY", "AY", "HR", "AR", "HC", "AC", "HF", "AF"]] = [1, 2, 0, 0, 5, 4, 9, 10]
        x, _ = build_features(df, ADVANCED_FEATURES)
        changed = df.copy()
        changed.loc[40:, ["HY", "HR"]] = [8, 2]
        other, _ = build_features(changed, ADVANCED_FEATURES)
        np.testing.assert_allclose(x.iloc[:41], other.iloc[:41])
        self.assertFalse(np.allclose(x.iloc[41:], other.iloc[41:]))

    def test_bad_form_and_prior_red_cards_are_visible(self):
        df = pd.DataFrame([[pd.Timestamp("2025-01-01") + pd.Timedelta(days=i), "Alpha", "Beta", 0, 2, "A", 1, 2, 0, 0]
                           for i in range(15)], columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HY", "AY", "HR", "AR"])
        df.loc[9, "HR"] = 1
        x, _ = build_features(df, ADVANCED_FEATURES)
        self.assertEqual(x.loc[10, "home_loss_streak_20"], 10)
        self.assertEqual(x.loc[10, "home_winless_streak_20"], 10)
        self.assertEqual(x.loc[10, "home_goals_for_10"], 0)
        self.assertEqual(x.loc[10, "home_previous_reds"], 1)
        self.assertEqual(x.loc[10, "away_points_10"], 3)

    def test_covid_season_uses_source_identity(self):
        df = pd.DataFrame([["26/07/2020", "Alpha", "Beta", 2, 1]], columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1920.csv"
            df.to_csv(path, index=False)
            loaded, source = load_matches([path])
        self.assertEqual(season_labels(loaded).iloc[0], 2019)
        self.assertEqual(source[0]["season_source"], "season filename")

    def test_invalid_shots_quarantine_retains_result_and_records_source(self):
        df = history(12)
        df[["HS", "AS", "HST", "AST"]] = [10, 8, 3, 9]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matches.csv"
            df.to_csv(path, index=False)
            before = path.read_bytes()
            loaded, source = load_matches([path], quarantine_invalid_shots=True)
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(loaded), 12)
        self.assertTrue(loaded.AS.isna().all() and loaded.AST.isna().all())
        self.assertTrue(loaded.HS.notna().all())
        self.assertEqual(len(source[0]["quarantined_stat_pairs"]), 12)
        self.assertEqual(source[0]["quarantined_stat_pairs"][0]["source_values"], [8, 9])

    def test_cached_download_makes_no_network_request(self):
        df = history(10)
        df["Date"] = pd.date_range("2025-08-01", periods=10)
        with tempfile.TemporaryDirectory() as tmp:
            df.to_csv(Path(tmp) / "2526.csv", index=False)
            with patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network")):
                manifest = download_seasons(2025, 2025, tmp)
        self.assertTrue(manifest["files"][0]["cached"])

    def test_recent_candidates_export_and_predict(self):
        df = history(450)
        x, _ = build_features(df, ADVANCED_FEATURES)
        fit = df.index < 360
        for kind in ["recent_logistic", "recent_poisson", "recent_xgboost"]:
            with self.subTest(kind=kind):
                try:
                    artifact = fit_candidate(kind, df, x, fit, ~fit, [])
                except ValueError as e:
                    if kind == "recent_xgboost" and "Install" in str(e): continue
                    raise
                self.assertEqual(artifact["candidate_name"], kind)
                self.assertEqual(artifact["recency_half_life_days"], 1095.75)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "model.json"
                    path.write_text(json.dumps(artifact, allow_nan=False))
                    loaded = read_model(path)
                p = probabilities(loaded, x.iloc[-3:])
                np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)


class AvailabilityTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([[pd.Timestamp("2025-01-03"), "Alpha", "Beta", 2, 1, "H"]],
                            columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

    def snapshots(self):
        return pd.DataFrame([["2025-01-03", "Alpha", 2, 1, 1, "2025-01-02T12:00:00Z", "verified example source"],
                             ["2025-01-03", "Beta", 1, 0, 0, "2025-01-02T13:00:00Z", "verified example source"],
                             ["2025-01-03", "Alpha", 8, 5, 5, "2025-01-03T09:00:00Z", "late example source"]],
                            columns=["Date", "Team", "Injured", "Suspended", "KeyMissing", "KnownAt", "Source"])

    def test_matchday_and_future_known_information_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "availability.csv"
            self.snapshots().to_csv(path, index=False)
            loaded, source = attach_availability(self.fixture(), path)
            snapshots = read_snapshots(path)
        self.assertEqual(source["eligible_paired_matches"], 1)
        x, _ = build_features(loaded, ADVANCED_FEATURES + AVAILABILITY_FEATURES)
        self.assertEqual(x.loc[0, "home_injured"], 2)
        self.assertEqual(x.loc[0, "away_injured"], 1)
        self.assertEqual(x.loc[0, "home_suspended"], 1)
        early = snapshot_context(snapshots, "Alpha", "Beta", "2025-01-03", pd.Timestamp("2025-01-01T00:00:00Z"))
        self.assertFalse(early)

    def test_unknown_is_distinct_from_known_healthy(self):
        unknown = availability_values({}, "2025-01-03")
        healthy = availability_values({"HomeInjured": 0, "HomeSuspended": 0, "HomeKnownAt": "2025-01-02T12:00:00Z"}, "2025-01-03")
        self.assertEqual(unknown[6], 0)
        self.assertEqual(healthy[6], 1)

    def test_retrospective_season_aggregate_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aggregate.csv"
            pd.DataFrame([{"club": "Alpha", "season": 2025, "injury_burden": 0.2}]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "KnownAt"):
                read_snapshots(path)

    def test_untrained_model_cannot_advertise_injury_aware_forecast(self):
        df = history(450)
        x, state = build_features(df, ADVANCED_FEATURES)
        fit = df.index < 360
        artifact = fit_candidate("multinomial_logistic_regression", df, x, fit, ~fit, [])
        artifact.update(teams=state, history_through=str(df.Date.max().date()))
        day = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
        with self.assertRaisesRegex(ValueError, "not trained"):
            predict(artifact, "Alpha", "Beta", day, availability={"HomeInjured": 3})

    def test_trained_availability_model_accepts_and_saves_real_snapshot_types(self):
        # Synthetic integration data checks mechanics only, not sports accuracy.
        df = history(450)
        rng = np.random.default_rng(11)
        for side in ["Home", "Away"]:
            df[side + "KnownAt"] = df.Date.map(lambda d: (d - pd.Timedelta(days=1)).strftime("%Y-%m-%dT12:00:00Z"))
            df[side + "Injured"] = rng.integers(0, 6, len(df))
            df[side + "Suspended"] = rng.integers(0, 3, len(df))
        x, state = build_features(df, ADVANCED_FEATURES + AVAILABILITY_FEATURES)
        fit = df.index < 360
        artifact = fit_candidate("recent_logistic", df, x, fit, ~fit, [])
        artifact.update(teams=state, history_through=str(df.Date.max().date()))
        day = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
        known = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        rows = pd.DataFrame([[day, "Alpha", 4, 1, known, "synthetic integration test"],
                             [day, "Beta", 1, 0, known, "synthetic integration test"]],
                            columns=["Date", "Team", "Injured", "Suspended", "KnownAt", "Source"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "availability.csv"
            rows.to_csv(path, index=False)
            context = snapshot_context(read_snapshots(path), "Alpha", "Beta", day, datetime.now(timezone.utc))
            record = predict(artifact, "Alpha", "Beta", day, availability=context)
            saved = save_record(record, tmp)
            parsed = json.loads(saved.read_text())
        self.assertEqual(artifact["availability_training_rows"], 360)
        self.assertEqual(parsed["availability"]["HomeInjured"], 4)
        self.assertAlmostEqual(sum(parsed["model_probabilities"].values()), 1)


class ReportingTests(unittest.TestCase):
    def test_confidence_subset_reports_coverage(self):
        rows = pd.DataFrame({"FTR": ["H", "A", "H"], "p_home": [0.8, 0.1, 0.4], "p_draw": [0.1, 0.6, 0.35], "p_away": [0.1, 0.3, 0.25]})
        subsets = confidence_subsets(rows)
        self.assertEqual(subsets[1]["matches"], 2)
        self.assertAlmostEqual(subsets[1]["coverage"], 2/3)
        self.assertEqual(subsets[1]["metrics"]["accuracy"], 0.5)
        self.assertEqual(subsets[2]["matches"], 1)

    def test_odds_are_a_comparator_not_features(self):
        df = pd.DataFrame({"FTR": ["A"], "B365H": [5.0], "B365D": [4.0], "B365A": [2.0]})
        result = bookmaker_benchmark(df, [True])
        self.assertEqual(result["metrics"]["accuracy"], 1)
        self.assertFalse(any("B365" in name for name in ADVANCED_FEATURES))


if __name__ == "__main__":
    unittest.main()

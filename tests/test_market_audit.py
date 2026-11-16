"""The saved-result check must detect corruption rather than just print metrics."""

from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.check_market_results import audit, hit_summary, probability_summary


class MarketAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(__file__).resolve().parents[1] / "artifacts" / "markets"
        cls.rows = pd.read_csv(cls.directory / "forecasts.csv")
        cls.prices = pd.read_csv(cls.directory / "paper_price_checks.csv")

    def run_with(self, rows=None, prices=None):
        with patch("scripts.check_market_results.pd.read_csv", side_effect=[
            self.rows if rows is None else rows, self.prices if prices is None else prices,
        ]):
            return audit(self.directory)

    def test_bundled_results_reconcile(self):
        result = self.run_with()
        self.assertEqual(result["status"], "saved_calculations_verified")
        self.assertEqual(result["results"]["all_walk_forward"]["matches"], len(self.rows))
        self.assertNotIn(result["latest_season"],
                         result["results"]["recent_completed_development"]["seasons"])

    def test_changed_payout_is_rejected(self):
        prices = self.prices.copy()
        prices.loc[0, "paper_net_units"] += 1
        with self.assertRaisesRegex(ValueError, "payout mismatch"):
            self.run_with(prices=prices)

    def test_duplicate_fixture_is_rejected(self):
        rows = pd.concat([self.rows.iloc[[0]], self.rows], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate forecast"):
            self.run_with(rows=rows)

    def test_missing_outcome_probability_is_rejected(self):
        rows = self.rows.copy()
        rows.loc[0, "p_home"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Missing goal/outcome probabilities"):
            self.run_with(rows=rows)

    def test_small_subset_has_coverage_and_uncertainty(self):
        result = hit_summary([True] * 9, 100)
        self.assertEqual(result["coverage"], 0.09)
        self.assertAlmostEqual(result["wilson_95"][0], 0.7008549515804559)
        self.assertIsNone(hit_summary([], 100)["hit_rate"])

    def test_calibration_bins_preserve_unknowns_and_probability_one(self):
        result = probability_summary([0, 1, float("nan")], [0, 1, 0.7])
        self.assertEqual(result["matches"], 2)
        self.assertEqual(result["brier"], 0)
        self.assertEqual(sum(b["matches"] for b in result["calibration_bins"]), 2)

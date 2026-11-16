import unittest

import numpy as np
import pandas as pd
from scipy.stats import poisson

from epl_predictor.markets import market_features, market_probabilities, price_checks, targets


class MarketTests(unittest.TestCase):
    def test_markets_match_the_score_distribution(self):
        home, away = 1.8, 1.1
        p = market_probabilities(home, away, 4.0, 0.15)
        i, j = np.indices((30, 30))
        score = poisson.pmf(i, home) * poisson.pmf(j, away)
        self.assertAlmostEqual(p["home"], score[i > j].sum())
        self.assertAlmostEqual(p["1X"], score[i >= j].sum())
        self.assertAlmostEqual(p["X2"], score[i <= j].sum())
        self.assertAlmostEqual(p["12"], score[i != j].sum())
        self.assertAlmostEqual(p["over_2.5"], score[i + j >= 3].sum())
        self.assertAlmostEqual(p["btts_yes"], score[(i > 0) & (j > 0)].sum())
        self.assertAlmostEqual(p["yellow_over_3.5"] + p["yellow_under_3.5"], 1)
        self.assertAlmostEqual(p["red_any"] + p["red_none"], 1)

    def test_quotes_distinguish_hit_probability_from_price(self):
        checks = price_checks(["1X=1.20", "home=2.00", "score_1_1=7.00"],
                              {"1X": 0.8, "home": 0.6}, 1.6, 1.2)
        self.assertAlmostEqual(checks["1X"]["estimated_return_per_unit"], -0.04)
        self.assertFalse(checks["1X"]["passes_paper_filter"])
        self.assertFalse(checks["home"]["passes_paper_filter"])
        self.assertAlmostEqual(checks["score_1_1"]["probability"],
                               poisson.pmf(1, 1.6) * poisson.pmf(1, 1.2))

    def test_unverified_card_settlement_and_invalid_odds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "settlement"):
            price_checks(["red_any=4.0"], {"red_any": 0.2}, 1.6, 1.2)
        with self.assertRaisesRegex(ValueError, "finite"):
            price_checks(["1X=inf"], {"1X": 0.8}, 1.6, 1.2)

    def test_head_to_head_uses_earlier_meetings_in_current_team_order(self):
        matches = pd.DataFrame([
            {"Date": pd.Timestamp("2015-08-08"), "HomeTeam": "A", "AwayTeam": "B", "FTHG": 3, "FTAG": 1, "FTR": "H"},
            {"Date": pd.Timestamp("2016-02-02"), "HomeTeam": "B", "AwayTeam": "A", "FTHG": 2, "FTAG": 0, "FTR": "H"},
        ])
        x, _, _ = market_features(matches)
        self.assertEqual(x.meetings.tolist(), [0, 1])
        self.assertAlmostEqual(x.home_goals.iloc[1], (1 + 7) / 6)
        self.assertAlmostEqual(x.away_goals.iloc[1], (3 + 7) / 6)
        matches.loc[1, "FTHG"] = 5
        changed, _, _ = market_features(matches)
        np.testing.assert_allclose(x, changed)

    def test_missing_card_counts_remain_unknown(self):
        actual = targets(pd.DataFrame({
            "FTHG": [1, 2], "FTAG": [0, 2], "HY": [1, np.nan], "AY": [3, 1],
            "HR": [0, np.nan], "AR": [0, 0],
        }))
        self.assertEqual(actual.red_any.iloc[0], 0)
        self.assertTrue(pd.isna(actual.red_any.iloc[1]))
        self.assertTrue(pd.isna(actual.yellow_cards.iloc[1]))

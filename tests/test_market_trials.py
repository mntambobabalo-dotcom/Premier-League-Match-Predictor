from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from epl_predictor.market_trials import (calibration_window, choose_candidate, prior_form,
                                        seasonal_predictions, selections, slip_ladder, slips)


def sample():
    rows = []
    for season in range(4):
        for day in range(6):
            for home, away in [("A", "B"), ("C", "D")]:
                rows.append(dict(Date=f"{2020 + season}-08-{day + 1:02d}", HomeTeam=home, AwayTeam=away,
                                 season_label=str(2020 + season), FTHG=day % 3, FTAG=day % 2,
                                 expected_home_goals=1.5, expected_away_goals=1.0))
    return pd.DataFrame(rows)


class MarketTrialTests(unittest.TestCase):
    def test_calibration_window_keeps_boundary_date(self):
        rows = sample().iloc[:12]
        window = calibration_window(rows, 3)
        self.assertEqual(len(window), 4)
        self.assertEqual(window.Date.nunique(), 2)

    def test_future_labels_cannot_change_any_forecast_or_scaling(self):
        rows = sample()
        first, folds = seasonal_predictions(rows, requested=4)
        changed = rows.copy()
        changed.loc[changed.season_label.eq("2023"), ["FTHG", "FTAG"]] = [8, 7]
        second, new_folds = seasonal_predictions(changed, requested=4)
        columns = [c for c in first if ":" in c]
        np.testing.assert_allclose(first[columns], second[columns])
        self.assertEqual(folds, new_folds)
        for fold in folds:
            self.assertLess(fold["calibration_end"], fold["test_start"])

    def test_prior_form_excludes_same_date_and_future_results(self):
        rows = sample()
        first = prior_form(rows)
        changed = rows.copy()
        changed.loc[changed.Date.ge("2021-08-03"), ["FTHG", "FTAG"]] = [9, 8]
        second = prior_form(changed)
        cols = [c for c in first if "prior" in c]
        np.testing.assert_allclose(first.loc[first.Date.le("2021-08-03"), cols],
                                   second.loc[second.Date.le("2021-08-03"), cols], equal_nan=True)
        self.assertTrue(first.home_prior_total.iloc[:10].isna().all())

    def test_selection_is_development_only(self):
        scores = {"raw": {"mean_event_brier": .2}, "rate_scaled": {"mean_event_brier": .19},
                  "prior_rate": {"mean_event_brier": .25}}
        self.assertEqual(choose_candidate(scores), "rate_scaled")
        copy = deepcopy(scores)
        self.assertEqual(scores, copy)

    def test_filters_skip_and_slips_have_distinct_teams_and_real_settlement(self):
        rows, _ = seasonal_predictions(sample(), requested=4)
        self.assertTrue(selections(rows, "raw", "mixed", 1.0).empty)
        selected = pd.DataFrame([
            dict(Date="2025-01-01", HomeTeam="A", AwayTeam="B", selection="1X", probability=.9, won=True),
            dict(Date="2025-01-01", HomeTeam="A", AwayTeam="C", selection="1X", probability=.85, won=True),
            dict(Date="2025-01-01", HomeTeam="D", AwayTeam="E", selection="X2", probability=.8, won=False),
        ])
        summary, records = slips(selected, 2, 2)
        self.assertEqual(summary["matches"], 1)
        self.assertEqual(summary["hit_rate"], 0)
        self.assertEqual(summary["coverage"], .5)
        self.assertAlmostEqual(records[0]["independent_probability"], .72)
        self.assertEqual(records[0]["fixtures"][1]["home"], "D")
        self.assertIsNone(summary["paper_return"])

    def test_paired_slips_compare_identical_dates(self):
        picks = pd.DataFrame([
            dict(Date="2025-01-01", HomeTeam=f"H{i}", AwayTeam=f"A{i}", selection="1X",
                 probability=.9 - i * .02, won=i < 2) for i in range(4)
        ] + [dict(Date="2025-01-02", HomeTeam="X", AwayTeam="Y", selection="1X", probability=.9, won=True)])
        ladder = slip_ladder(picks, 2)
        self.assertEqual(ladder[1][0]["matches"], 2)
        self.assertEqual([ladder[k][2]["matches"] for k in range(1, 5)], [1, 1, 1, 1])
        self.assertEqual([ladder[k][2]["hit_rate"] for k in range(1, 5)], [1, 1, 0, 0])

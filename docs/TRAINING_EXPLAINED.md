# How this version trains

The video proposes learning from historical Premier League games, possession and goals. Our current implementation starts at 2015/16 and uses the statistics actually available: goals, form, Elo, home/away performance, shots, cards, fouls and corners. Possession and injuries remain explicit optional imports, not hidden data.

Each match is a training example. Its inputs describe earlier games and eligible pre-match news. Its target is the final H/D/A result. Current-match goals, shots and red cards only become history for later dates. The model cannot know a future red card before kickoff.

We test each season using earlier dates. About the last 190 prior matches calibrate probability sharpness; earlier rows fit the coefficients/trees. Test-season parameters remain frozen, while team form updates after completed earlier dates. Source season labels preserve the July 2020 end of 2019/20.

Seven candidates are compared: a frequency baseline, logistic regression, an independent Poisson goal model, XGBoost, and recent-weighted versions of the three learned models. Recent weighting gives older fitting rows less influence but keeps the 2015 history. The half-life is three years.

The Poisson model estimates home and away expected goals, then turns their difference into H/D/A probabilities. Logistic regression directly learns outcome-class relationships. XGBoost learns small tree rules. Complexity does not guarantee better predictions.

Choose the lowest probability log loss across the latest three completed development seasons. The partial current season does not choose the winner. The newest refit is then prepared for future paper forecasts; its live performance has not been established.

Accuracy counts the largest-probability pick. Log loss and Brier evaluate the probability estimates. A model can improve loss and still pick fewer winners correctly, which happened on the latest season here. Read V0_3_RESULTS.md rather than assuming more data equals improvement.

Your football judgment is a separate H/D/A forecast with notes, recorded before viewing the machine output. The journal later compares both on the same fixtures. A confident opinion or an injury narrative is not automatically an edge; it needs recorded future evidence.

The target remains 55–60%, but the selected model scored 54.2% on the recent completed development seasons and 40% on the latest 100-match audit. No bookmaker edge or profits have been demonstrated. The next meaningful work is reliable availability data, paired feature tests and an untouched future paper trial.

"""Refresh the trial section of the existing Markdown research note."""

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd

from epl_predictor.market_trials import hit_metrics, selections


def pct(value):
    return "unavailable" if value is None else f"{value:.1%}"


def interval(metric):
    bounds = metric["wilson_95"]
    return "unavailable" if bounds is None else f"{bounds[0]:.1%}–{bounds[1]:.1%}"


def render(directory):
    directory = Path(directory)
    report_bytes = (directory / "report.json").read_bytes()
    # The numerical narrative below describes this reviewed snapshot. Refuse to
    # carry it into a changed experiment; new results require a new interpretation.
    reviewed_hash = "d31b8588806a3f4836a6d9327f24bec72e1b3d3e65551cb781e6db6c1c75e0e6"
    if sha256(report_bytes).hexdigest() != reviewed_hash:
        raise ValueError("Trial results changed: review and update the narrative before rendering")
    report = json.loads(report_bytes)
    rows = pd.read_csv(directory / "forecasts.csv")
    selected = report["selected_candidate"]
    scopes = report["scopes"]
    lines = ["## Goal markets and parlay experiment — 16 November 2026", "",
        "### Executive Summary", "",
        "Filtered draw cover is the strongest candidate for further paper testing in this experiment. A 75% minimum predicted probability produced 470/547 development hits (85.9%; 48.0% fixture coverage), and 27/31 latest-season hits (87.1%; 31% coverage). The latest interval is wide: 71.1–94.9%. This does not establish profitability: archived prices for these markets are absent.", "",
        "Forced under 3.5 and BTTS picks remain unreliable. The latest 75%+ under subset hit 4/8, and the highest-probability BTTS side hit 48/100 when every fixture required a pick. Combining markets from the same game can narrow the winning scorelines sharply.", "",
        "The original forecast probabilities beat the tested rate-scaling calibration on development Brier. No change was promoted to the production model. This is an exploratory saved-forecast experiment, not a raw-data refit or an untouched future test. [4]", "",
        "### Baseline and calibration comparison", "",
        "There are 3,140 comparable fixtures from 2018/19 through 8 November 2026. The first 380 saved forecasts warm up the prior-only comparisons and are excluded from all three candidates. Selection uses 2023/24–2025/26 only (1,140 fixtures); the latest 2026/27 sample (100) is excluded.", "",
        "| Candidate | Development mean Brier | Latest mean Brier |", "|---|---:|---:|"]
    for c in ("raw", "rate_scaled", "prior_rate"):
        lines.append(f"| {c} | {report['development_scores'][c]['mean_event_brier']:.4f} | {scopes['latest_season']['models'][c]['mean_event_brier']:.4f} |")
    lines += ["", "Lower Brier is better. This fixed metric averages under 3.5, BTTS yes, 1X and X2 probability errors. Raw probabilities were selected. Calibration multiplies home/away goal rates by observed-to-predicted goal ratios on about 190 prior out-of-season forecasts (whole boundary dates retained; scales capped at 0.5–2). The baseline uses event frequencies from those same prior dates. Parameters freeze before each test season. This evaluates a forecast postprocessor; it does not reserve a fresh calibration set inside the original goal-head fitting process.", "",
              "| Event | Raw development Brier | Prior-rate development Brier | Raw latest Brier |", "|---|---:|---:|---:|"]
    for event in ("under_3.5", "btts_yes", "1X", "X2"):
        dev = report["development_scores"]
        lines.append(f"| {event} | {dev['raw']['events'][event]['brier']:.4f} | {dev['prior_rate']['events'][event]['brier']:.4f} | {scopes['latest_season']['models']['raw']['events'][event]['brier']:.4f} |")
    lines += ["", "### When the rules hit and when they skip", "",
              "All thresholds (65%, 75%, 80%) were fixed for comparison; none was optimized. The 75% rows below illustrate the common threshold, not a claim that it is optimal. Each draw-cover pick chooses between 1X and X2; BTTS chooses yes/no. Under 3.5 remains a one-sided selection.", "",
              "| Rule / population | Hits / selections | Fixture coverage | Hit rate | Approximate 95% interval |", "|---|---:|---:|---:|---:|"]
    for scope, label in [("recent_development", "Development"), ("latest_season", "Latest")]:
        subset = rows.loc[rows.season_label.isin(report["selection_seasons"])] if scope == "recent_development" else rows.loc[rows.season_label.eq(report["latest_season"])]
        for family, title in [("under_3.5", "Under 3.5, every fixture"), ("btts", "BTTS higher-probability side, every fixture"), ("draw_cover", "Draw cover, every fixture")]:
            picks = selections(subset, selected, family, 0)
            m = hit_metrics(picks.won, len(subset))
            lines.append(f"| {label}: {title} | {m['hits']}/{m['matches']} | {pct(m['coverage'])} | {pct(m['hit_rate'])} | {interval(m)} |")
        for key, title in [("under_3.5@0.75", "Under 3.5, 75%+"), ("draw_cover@0.75", "Draw cover, 75%+")]:
            m = scopes[scope]["filters"][key]
            lines.append(f"| {label}: {title} | {m['hits']}/{m['matches']} | {pct(m['coverage'])} | {pct(m['hit_rate'])} | {interval(m)} |")
    lines += ["", "There were only four development BTTS selections at 65%+ (3 hits; 0.35% coverage; interval 30.1–95.4%) and none in the latest season. At 75%+ there were none in either period. A no-pick output is appropriate when the rule finds nothing; a large parlay is not filled with lower-confidence substitutes.", "",
              "### What is associated with four or more goals?", "",
              "These are descriptive associations, not established causes. Scores are used only to settle outcomes; every grouping uses earlier results or prematch expected rates. Overlapping segments must not be added together.", "",
              "| Prematch grouping, all 3,140 comparable fixtures | 4+ goals / fixtures | Over 3.5 rate | Under 3.5 rate |", "|---|---:|---:|---:|"]
    risks = scopes["all_evaluable"]["over_goal_analysis"]
    for key, title in [("expected_total_below_2.5", "Expected total below 2.5"), ("expected_total_2.5_to_3", "Expected total 2.5 to below 3"), ("expected_total_at_least_3", "Expected total at least 3"),
                       ("both_prior_five_total_at_least_3", "Both teams' prior five games averaged at least 3 total goals"), ("at_least_one_prior_five_total_below_3", "At least one prior-five average below 3")]:
        m = risks["segments"][key]
        lines.append(f"| {title} | {m['hits']}/{m['matches']} | {pct(m['hit_rate'])} | {pct(m['under_hit_rate'])} |")
    lines += ["", f"Prior-five form is known for {risks['prior_form_covered_fixtures']}/3,140 fixtures; incomplete histories stay missing. Simple recent scoring averages separated risk only weakly. The strongest separation in these predeclared groups came from expected total goals, but it did not reliably order the tiny latest-season groups: 3/7 low-expected-total games and 4/15 high-expected-total games went over. That limits using the historical association as a dependable rule.", "",
              f"Of {risks['over_games']} actual 4+ goal games, {risks['over_with_btts']} had both teams score and {risks['over_with_clean_sheet']} had a clean sheet. In the latest sample it was 28 and 3 out of 31 respectively. These are descriptions after settlement: actual BTTS must never become a prematch input. A weak opponent can still lose 4–0; a presumed one-sided match does not guarantee under 3.5.", "",
              "Recent defensive averages also provide no established universal explanation: in development, the combined prior-five concession rate at least 3 group went over 217/576 (37.7%), versus 185/549 (33.7%) below 3. The differences are exploratory and can reflect team strength and other confounding factors. Goal timings, red-card timings, lineups and verified historical availability are absent, so this experiment cannot attribute individual overs to them.", "",
              "### Parlay size: compare the same dates", "",
              "Each date supplies ranked selections with 75%+ model probability, using one draw-cover pick per fixture and no repeated teams. Only dates with four eligible selections enter this paired comparison; smaller slips use the top part of the same four. These are observed joint wins, not assumed-independent simulation outcomes.", "",
              "| Slip size | Wins / 38 development dates | Hit rate (95% interval) | Wins / 108 historical dates |", "|---|---:|---:|---:|"]
    for k in range(1, 5):
        dev = scopes["recent_development"]["paired_slips"][f"draw_cover@0.75/{k}"]
        hist = scopes["all_evaluable"]["paired_slips"][f"draw_cover@0.75/{k}"]
        lines.append(f"| {k} | {dev['hits']}/{dev['matches']} | {pct(dev['hit_rate'])} ({interval(dev)}) | {hist['hits']}/{hist['matches']} |")
    lines += ["", "Those dates cover only 11.1% of development fixture dates and 11.1% of all comparable fixture dates. The latest season supplies only one such date, and its slips lost at every size; that sample is too small for useful ranking. Slip sizes are nested and dependent. More legs reduced the measured chance of all selections winning. This cannot identify the most profitable size without actual offered prices. All four markets have zero historical price coverage here; paper returns are unavailable. Other thresholds, mixed policies and all eligible-date results remain in the JSON rather than being presented as winners.", "",
              "A useful next paper protocol is to compare eligible singles with eligible doubles, preserve no-pick days, record actual prices and settle every selection. Three- and four-leg slips are not justified as more reliable by these results. No real-money staking system or automated wager was added.", "",
              "### Your Man City–Fulham example", "",
              "The fixture is listed for 21 November 2026 by [Fulham's official ticket notice](https://www.fulhamfc.com/news/2026/october/13/man-city-away-tickets/). The current paper refit, with result history through 8 November, gives 65.1% under 3.5, 55.5% BTTS yes and 80.0% City-or-draw. These are unverified model estimates, not the historical filtered model's achieved hit rate. The first two fall below the illustrative 75% threshold; the draw cover passes that probability threshold, while a price check is still missing.", "",
              "Combining all three in the same match permits only 1–1 or 2–1. Their model probabilities sum to 20.2%, so this is a narrow scoreline combination. Its probability must not be calculated by multiplying the three marginal probabilities. Betway's [Build a Bet explanation](https://www.betway.co.za/betway-pedia/documents/betway-features/page/how-does-build-a-bet-work) confirms that all selections must win. A same-game builder needs its own quoted combined price.", "",
              "The pending forecast is saved with creation time, model hash, fixture source and no invented prices. It is awaiting actual decimal odds for each market and, if desired, the combined builder quote. No priced paper selection or return is recorded yet.", "",
              "### Reproduce and inspect [4]", "", "```powershell", ".\\.venv\\Scripts\\python.exe scripts/test_market_policies.py --out artifacts/market_trials_v2", ".\\.venv\\Scripts\\python.exe scripts/summarize_market_trial.py", ".\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v", "```", "",
              "[4] Inputs: `artifacts/markets/forecasts.csv` (SHA-256 recorded in the trial report). Outputs: `artifacts/market_trials_v2/report.json`, `forecasts.csv`, `slips.json`, and `man_city_fulham_2026-11-21.json`. The forecast and slip exports retain every evaluated candidate and ticket settlement. The full report preserves bin counts, paired-baseline Brier, fixed thresholds, fold dates, coverage and Wilson intervals. The six new checks cover chronology, whole-date calibration, missing form, future-label isolation, skipping and paired slip settlement; all 53 project tests passed.", "",
              "No source downloads occurred. Raw CSVs remain absent. This does not complete a head-to-head feature ablation or the originally proposed raw-head calibration refit. All inspected history is exploratory; a future priced paper trial remains required. Wilson intervals assume independent trials and may understate uncertainty from repeated teams/dates. Do not sum overlapping report scopes or policies as distinct trials.", ""]
    quote_path = directory / "man_city_fulham_2026-11-21_price_check.json"
    if quote_path.exists():
        quote = json.loads(quote_path.read_text(encoding="utf-8"))
        note = ("The user subsequently supplied Betway decimal prices. These are manually reported prices with an unknown provider observation timestamp; the local record preserves the time they were entered. The original forecast creation time and model hash are preserved.")
        table = [note, "", "| Market | Entered odds | Model probability | Break-even probability | Model-estimated return per unit |", "|---|---:|---:|---:|---:|"]
        for name, check in quote["price_checks"].items():
            table.append(f"| {name} | {check['entered_odds']:.2f} | {pct(check['probability'])} | {pct(check['break_even_probability'])} | {check['estimated_return_per_unit']:+.1%} |")
        table += ["", "None passes the existing +5% minimum estimated-return filter. Under 3.5 has a small positive model estimate but also falls below the illustrative 75% probability floor. City-or-draw passes that probability floor but has a negative estimated return at the entered price. No selection was recorded and no realized return exists. The combined builder price is still missing; individual prices must not be multiplied to manufacture a same-game quote.", "", "The separate `man_city_fulham_2026-11-21_price_check.json` retains the supplied quotes, filter results and unresolved status."]
        old = "The pending forecast is saved with creation time, model hash, fixture source and no invented prices. It is awaiting actual decimal odds for each market and, if desired, the combined builder quote. No priced paper selection or return is recorded yet."
        lines[lines.index(old)] = "\n".join(table)
    return "\n".join(lines)


if __name__ == "__main__":
    path = Path("docs/MARKET_RESULTS_CHECK.md")
    original = path.read_text(encoding="utf-8")
    marker = "<!-- market-trial-update -->"
    original = original.split(marker)[0].rstrip()
    path.write_text(original + "\n\n" + marker + "\n\n" + render("artifacts/market_trials_v2"), encoding="utf-8")
    print(f"Updated {path}")

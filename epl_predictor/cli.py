import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

from .data import load_matches, attach_possession
from .model import read_model, train
from .predict import predict, save_record, score_records
from .availability import attach_availability, read_snapshots, snapshot_context


def input_paths(args):
    if args.csv:
        return args.csv
    paths = sorted(Path(args.data_dir).glob("[0-9][0-9][0-9][0-9].csv"))
    if not paths:
        raise ValueError("No season CSVs found in --data-dir; use download-data first")
    return paths


def inputs(command):
    group = command.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", nargs="+")
    group.add_argument("--data-dir", help="Directory of four-digit season CSV names such as 1516.csv")
    command.add_argument("--quarantine-invalid-shots", action="store_true", help="Keep outcomes but explicitly mark inconsistent shot pairs missing, with provenance")


def main():
    parser = argparse.ArgumentParser(description="Premier League probability research engine")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Original v0.1 single 60/20/20 holdout experiment")
    inputs(t)
    t.add_argument("--out", default="artifacts/legacy_holdout")
    b = sub.add_parser("benchmark", help="Compare fixed candidates on successive seasons, then save a latest paper model")
    inputs(b)
    b.add_argument("--possession-csv", help="Optional timestamped historical possession observations")
    b.add_argument("--availability-csv", help="Optional pre-match injury/suspension snapshots with KnownAt")
    b.add_argument("--out", default="artifacts")
    b.add_argument("--skip-xgboost", action="store_true", help="Explicitly run the comparison without the optional XGBoost dependency")
    check = sub.add_parser("data-check", help="Validate data and show which inputs are actually available")
    inputs(check)
    check.add_argument("--possession-csv")
    check.add_argument("--availability-csv")
    download = sub.add_parser("download-data", help="Explicit one-time cached personal-research season download")
    download.add_argument("--start-season", type=int, default=2015)
    download.add_argument("--end-season", type=int, default=2026)
    download.add_argument("--out", default="data")
    p = sub.add_parser("predict", help="Create a paper forecast or explicitly labelled demo")
    p.add_argument("--model", default="artifacts/model.json")
    p.add_argument("--home", required=True)
    p.add_argument("--away", required=True)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--human", help="Your independent H,D,A probabilities, e.g. 0.50,0.28,0.22")
    p.add_argument("--note", default="")
    p.add_argument("--records", default="predictions")
    p.add_argument("--demo", action="store_true", help="Allow stale/past illustrative matchups; excluded from journal scores")
    p.add_argument("--availability-csv", help="Verified snapshots for this fixture; requires a model trained with availability")
    names = sub.add_parser("teams", help="List team names known to this model")
    names.add_argument("--model", default="artifacts/model.json")
    demo = sub.add_parser("demo", help="Run a labelled illustrative matchup with the included model")
    demo.add_argument("--model", default="artifacts/model.json")
    demo.add_argument("--records", default="predictions")
    markets = sub.add_parser("markets", help="Show score, goals, double chance, cards and manual price checks")
    markets.add_argument("--model", default="artifacts/markets/model.json")
    markets.add_argument("--home")
    markets.add_argument("--away")
    markets.add_argument("--date", help="YYYY-MM-DD")
    markets.add_argument("--demo", action="store_true", help="Label the matchup as an illustration")
    markets.add_argument("--price", action="append", default=[], help="Repeat for each quote, e.g. X2=1.60")
    markets.add_argument("--json", action="store_true", help="Emit the complete market report as JSON")
    market_test = sub.add_parser("benchmark-markets", help="Refit and evaluate the market models on past seasons")
    inputs(market_test)
    market_test.add_argument("--out", default="artifacts/markets")
    s = sub.add_parser("score", help="Compare recorded human and model forecasts on the same settled matches")
    s.add_argument("--records", default="predictions")
    inputs(s)
    args = parser.parse_args()
    try:
        if args.command in ["benchmark", "data-check"]:
            matches, provenance = load_matches(input_paths(args), args.quarantine_invalid_shots)
            if args.possession_csv:
                matches, pos_source = attach_possession(matches, args.possession_csv)
                provenance.append(pos_source)
            if args.availability_csv:
                matches, availability_source = attach_availability(matches, args.availability_csv)
                provenance.append(availability_source)
            if args.command == "data-check":
                print(json.dumps({"matches": len(matches), "from": str(matches.Date.min().date()),
                                  "through": str(matches.Date.max().date()),
                                  "complete_shot_rows": int(matches[["HS", "AS", "HST", "AST"]].notna().all(axis=1).sum()),
                                  "possession_rows": int(matches.HomePossession.notna().sum()) if "HomePossession" in matches else 0,
                                  "discipline_rows": int(matches[["HY", "AY", "HR", "AR"]].notna().all(axis=1).sum()),
                                  "availability_paired_rows": int(matches[["HomeKnownAt", "AwayKnownAt"]].notna().all(axis=1).sum()) if "HomeKnownAt" in matches else 0,
                                  "data_fingerprints": provenance}, indent=2))
            else:
                from .research import benchmark
                report = benchmark(matches, provenance, args.out, not args.skip_xgboost)
                print(json.dumps({k: report[k] for k in ["coverage", "development", "selection_metrics", "audit", "target_accuracy", "selected_model", "latest_paper_model"]}, indent=2))
        elif args.command == "download-data":
            from .downloads import download_seasons
            print(json.dumps(download_seasons(args.start_season, args.end_season, args.out), indent=2))
        elif args.command == "train":
            matches, provenance = load_matches(input_paths(args), args.quarantine_invalid_shots)
            report = train(matches, provenance, args.out)
            print(json.dumps({k: v for k, v in report.items() if k != "reliability"}, indent=2))
        elif args.command == "predict":
            human = [float(x) for x in args.human.split(",")] if args.human else None
            availability = None
            if args.availability_csv:
                availability = snapshot_context(read_snapshots(args.availability_csv), args.home, args.away, args.date,
                                                datetime.now(timezone.utc))
            r = predict(read_model(args.model), args.home, args.away, args.date, human, args.note, args.demo, availability)
            path = save_record(r, args.records)
            print(json.dumps(r, indent=2))
            print(f"Saved forecast: {path}")
        elif args.command == "benchmark-markets":
            from .markets import benchmark_markets
            matches, provenance = load_matches(input_paths(args), args.quarantine_invalid_shots, include_totals_odds=True)
            report = benchmark_markets(matches, provenance, args.out)
            print(json.dumps({key: report[key] for key in ("source_matches", "through", "latest_season",
                                                           "historical_prices", "latest_season_prices")}, indent=2))
        elif args.command == "markets":
            import pandas as pd
            from .markets import forecast
            artifact = json.loads(Path(args.model).read_text(encoding="utf-8"))
            if artifact.get("schema") != "epl_markets_v1":
                raise ValueError("Use the market model from benchmark-markets")
            if args.demo and args.home is None and args.away is None:
                args.home, args.away = "Man City", "Man United"
            if args.demo and args.date is None:
                args.date = (pd.Timestamp(artifact["history_through"]) + pd.Timedelta(days=1)).date().isoformat()
            if not all((args.home, args.away, args.date)):
                raise ValueError("Specify --home, --away and --date, or run markets --demo")
            result = forecast(artifact, args.home, args.away, args.date, args.demo, args.price)
            if args.json:
                print(json.dumps(result, indent=2, allow_nan=False))
            else:
                print("Illustrative matchup" if args.demo else "Pre-match model estimates")
                print(f"{args.home} vs {args.away} | {result['date']} | history through {result['history_through']}")
                print(f"Expected goals: {result['expected_home_goals']:.2f} home, "
                      f"{result['expected_away_goals']:.2f} away, {result['expected_total_goals']:.2f} total")
                print("Most likely scorelines: " + ", ".join(
                    f"{entry['score']} ({entry['probability']:.1%})" for entry in result["top_scores"]))
                print(f"Result focus: {result['result_focus']} (probability only)")
                for key, probability in result["probabilities"].items():
                    print(f"{key:20} {probability:6.1%}  fair odds {1 / probability:.2f}" if probability
                          else f"{key:20}   0.0%  fair odds unavailable")
                if result["expected_yellows"] is not None:
                    print(f"Expected recorded yellows: {result['expected_yellows']:.2f}")
                    print("Card probabilities use archive counts. Check the offered settlement rules separately.")
                meetings = result["head_to_head"]
                if meetings["meetings"]:
                    print(f"Head-to-head across both venues ({meetings['meetings']} meetings): "
                          f"{args.home} {meetings['home_team_wins']} wins, {meetings['draws']} draws, "
                          f"{args.away} {meetings['away_team_wins']} wins, "
                          f"{meetings['mean_total_goals']:.2f} goals per match")
                for game in reversed(result["last_five_meetings"]):
                    match = f"{game['date']}  {game['home']} {game['hg']}-{game['ag']} {game['away']}"
                    if game.get("archived_bet365_1x2"):
                        odds = " / ".join(f"{value:.2f}" for value in game["archived_bet365_1x2"])
                        match += f" | archived Bet365 H/D/A {odds}"
                    print(match)
                for selection, check in result["price_checks"].items():
                    status = "passes paper filter" if check["passes_paper_filter"] else "below paper filter"
                    print(f"{selection} @ {check['entered_odds']:.2f} | model {check['probability']:.1%} | "
                          f"break-even {check['break_even_probability']:.1%} | "
                          f"estimated return {check['estimated_return_per_unit']:+.1%} | {status}")
        elif args.command == "demo":
            import pandas as pd
            artifact = read_model(args.model)
            teams = sorted(artifact["teams"])
            home, away = ("Man City", "Arsenal") if {"Man City", "Arsenal"}.issubset(teams) else teams[:2]
            day = (pd.Timestamp(artifact["history_through"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            record = predict(artifact, home, away, day, note="Illustrative demo only; not a verified fixture", demo=True)
            path = save_record(record, args.records)
            print("DEMO ONLY: illustrative matchup, not a verified fixture")
            print(f"{home} vs {away} | example date {day}")
            for label, name in [("H", "Home win"), ("D", "Draw"), ("A", "Away win")]:
                print(f"{name}: {record['model_probabilities'][label]:.1%}")
            print(f"Candidate: {record['model_candidate']}. Demo excluded from paper scores.")
            print(f"Saved record: {path}")
        elif args.command == "teams":
            print("\n".join(sorted(read_model(args.model)["teams"])))
        else:
            matches, _ = load_matches(input_paths(args), args.quarantine_invalid_shots)
            print(json.dumps(score_records(args.records, matches), indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

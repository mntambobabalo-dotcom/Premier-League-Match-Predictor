"""Run the forecast-only market experiment from the project root."""

import argparse

from epl_predictor.market_trials import run_trial


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="artifacts/markets/forecasts.csv")
    parser.add_argument("--out", default="artifacts/market_trials_v2")
    args = parser.parse_args()
    report = run_trial(args.source, args.out)
    print(f"Saved market experiment: {args.out}")
    print(f"Development-selected candidate: {report['selected_candidate']}")

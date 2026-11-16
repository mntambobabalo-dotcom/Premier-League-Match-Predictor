"""Record independent model and human views before results are known."""
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .features import fixture_features
from .model import LABELS, metrics, probabilities


def predict(artifact, home, away, when, human=None, note="", demo=False, availability=None):
    fixture_date = pd.Timestamp(when)
    if fixture_date.tzinfo is not None:
        raise ValueError("Use an ISO calendar date: YYYY-MM-DD")
    fixture_date = fixture_date.normalize()
    last = pd.Timestamp(artifact["history_through"])
    if fixture_date <= last:
        raise ValueError("Fixture must be after the model's history cutoff; retrain with an earlier cutoff to backtest")
    now = datetime.now(timezone.utc)
    if not demo:
        if fixture_date.date() < now.date():
            raise ValueError("Past matchups require --demo and do not count as prematch records")
        if (fixture_date - last).days > 21:
            raise ValueError("History is over 21 days behind this fixture. Refresh results and retrain, or use --demo")
    if home == away:
        raise ValueError("Choose two different teams")
    unknown = set([home, away]) - set(artifact["teams"])
    if unknown:
        raise ValueError(f"Unknown team(s): {sorted(unknown)}. Use the teams command for exact names")
    if availability is not None:
        if "home_injured" not in artifact["features"] or artifact.get("availability_training_rows", 0) == 0:
            raise ValueError("This model was not trained on verified historical availability. Retrain with --availability-csv before requesting injury-aware predictions")
        if not availability:
            raise ValueError("No timely availability snapshots match this fixture")
    if human is not None:
        human = np.asarray(human, dtype=float)
        if human.shape != (3,) or not np.isfinite(human).all() or (human < 0).any() or (human > 1).any() or not np.isclose(human.sum(), 1, atol=1e-6):
            raise ValueError("Human probabilities must be three finite values H,D,A between 0 and 1, summing to 1")
        if not note.strip():
            raise ValueError("Add a note explaining your human forecast")
    p = probabilities(artifact, fixture_features(artifact["teams"], home, away,
                      fixture_date.date().isoformat(), artifact["features"], known_at=now, availability=availability))[0]
    return {"record_id": str(uuid4()), "created_utc": now.isoformat(),
            "mode": "historical_demo_not_prematch_evidence" if demo else "paper_forecast",
            "date": fixture_date.date().isoformat(), "home": home, "away": away,
            "model_probabilities": dict(zip(LABELS, p.tolist())),
            "human_probabilities": None if human is None else dict(zip(LABELS, human.tolist())),
            "note": note, "model_created_utc": artifact["created_utc"],
            "history_through": artifact["history_through"],
            "parameters_fit_through": artifact["parameters_fit_through"],
            "calibrated_through": artifact["calibrated_through"],
            "data_fingerprints": artifact["data_fingerprints"],
            "model_type": artifact["model_type"], "model_purpose": artifact.get("purpose", "legacy_holdout_model"),
            "model_candidate": artifact.get("candidate_name", artifact["model_type"]),
            "availability": None if availability is None else {k: v if pd.notna(v) else None for k, v in availability.items()},
            "warning": "Research estimate. No betting or profitability claim. Date-only records need a timing audit."}


def save_record(record, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['record_id']}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    return path


def score_records(directory, matches):
    lookup = {(r.Date.strftime("%Y-%m-%d"), r.HomeTeam, r.AwayTeam): r.FTR
              for r in matches.itertuples(index=False)}
    selected, seen = [], set()
    skipped = {"demo": 0, "unsettled": 0, "same_day_or_late_record": 0, "duplicate_fixture": 0}
    records = [json.loads(path.read_text(encoding="utf-8")) for path in Path(directory).glob("*.json")]
    records.sort(key=lambda r: (r["created_utc"], r["record_id"]))
    for r in records:
        if r["mode"] != "paper_forecast":
            skipped["demo"] += 1
            continue
        key = (r["date"], r["home"], r["away"])
        # No kickoff timestamps yet: accept only forecasts made before the match's
        # calendar day in both UTC and UK time. This conservatively drops matchday calls.
        made = pd.Timestamp(r["created_utc"])
        if max(made.date(), made.tz_convert("Europe/London").date()) >= pd.Timestamp(r["date"]).date():
            skipped["same_day_or_late_record"] += 1
            continue
        if key not in lookup:
            skipped["unsettled"] += 1
            continue
        if key in seen:
            skipped["duplicate_fixture"] += 1
            continue
        seen.add(key)
        selected.append((r, lookup[key]))
    if not selected:
        return {"scored": 0, "skipped": skipped, "message": "No eligible settled prematch records yet"}
    def evaluate(rows, field):
        return metrics([y for _, y in rows], np.array([[r[field][k] for k in LABELS] for r, _ in rows]))
    paired = [(r, y) for r, y in selected if r["human_probabilities"] is not None]
    return {"scored": len(selected), "skipped": skipped, "model_all": evaluate(selected, "model_probabilities"),
            "paired_matches": len(paired),
            "model_paired": evaluate(paired, "model_probabilities") if paired else None,
            "human_paired": evaluate(paired, "human_probabilities") if paired else None,
            "limitations": "Local editable research journal; not independent audited evidence. No odds or ROI measured."}

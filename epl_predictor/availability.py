"""Point-in-time injury/suspension snapshots. Unknown never means healthy.

No retrospective season totals or guessed suspension bans are accepted here.
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from .data import parse_dates

AVAILABILITY_FEATURES = ["home_injured", "away_injured", "home_suspended", "away_suspended",
                         "injury_difference", "suspension_difference", "home_availability_observed",
                         "away_availability_observed", "home_key_missing", "away_key_missing",
                         "home_key_missing_observed", "away_key_missing_observed"]


def read_snapshots(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["Date", "Team", "Injured", "Suspended", "KnownAt", "Source"]
    if not set(required).issubset(df) or df[required].isna().any().any():
        raise ValueError("Availability CSV requires complete Date,Team,Injured,Suspended,KnownAt,Source")
    df = df.copy()
    df.Date = parse_dates(df.Date)
    if df.Date.isna().any():
        raise ValueError("Invalid availability fixture date")
    for col in ["Team", "Source"]:
        df[col] = df[col].astype(str).str.strip()
        if df[col].eq("").any():
            raise ValueError("Availability team and source cannot be blank")
    if "KeyMissing" not in df:
        df["KeyMissing"] = np.nan
    for col in ["Injured", "Suspended", "KeyMissing"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        v = df[col].dropna()
        if (~np.isfinite(v) | (v < 0) | (v % 1 != 0)).any():
            raise ValueError("Availability counts must be finite nonnegative integers")
    if not df.KnownAt.astype(str).str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True).all():
        raise ValueError("Availability KnownAt requires an explicit timezone offset or Z")
    df.KnownAt = pd.to_datetime(df.KnownAt, utc=True, format="mixed", errors="raise")
    if df.KnownAt.isna().any() or (df.KnownAt > pd.Timestamp.now(tz="UTC")).any():
        raise ValueError("Availability KnownAt cannot be missing or in the future")
    if df.duplicated(["Date", "Team", "KnownAt"]).any():
        raise ValueError("Duplicate availability snapshots")
    return df


def snapshot_context(df, home, away, when, known_at=None):
    cutoff = pd.Timestamp(str(when)[:10]).tz_localize("Europe/London").tz_convert("UTC")
    if known_at is not None:
        cutoff = min(cutoff, pd.Timestamp(known_at))
    context = {}
    for side, team in [("Home", home), ("Away", away)]:
        rows = df.loc[df.Date.eq(pd.Timestamp(str(when)[:10])) & df.Team.eq(team) & df.KnownAt.lt(cutoff)]
        if len(rows):
            latest = rows.sort_values("KnownAt").iloc[-1]
            for col in ["Injured", "Suspended", "KeyMissing", "Source"]:
                value = latest[col]
                context[side + col] = str(value) if col == "Source" else int(value) if pd.notna(value) else None
            context[side + "KnownAt"] = latest.KnownAt.isoformat()
    return context


def attach_availability(matches, path):
    snapshots = read_snapshots(path)
    known = pd.concat([matches[["Date", "HomeTeam"]].rename(columns={"HomeTeam": "Team"}),
                       matches[["Date", "AwayTeam"]].rename(columns={"AwayTeam": "Team"})]).drop_duplicates()
    check = snapshots.merge(known, on=["Date", "Team"], how="left", indicator=True)
    if not check["_merge"].eq("both").all():
        raise ValueError("Availability contains unknown result-date/team keys")
    contexts = [snapshot_context(snapshots, row.HomeTeam, row.AwayTeam, row.Date.strftime("%Y-%m-%d"))
                for row in matches.itertuples(index=False)]
    added = pd.DataFrame(contexts, index=matches.index)
    for side in ["Home", "Away"]:
        for field in ["Injured", "Suspended", "KeyMissing", "Source", "KnownAt"]:
            col = side + field
            if col not in added:
                added[col] = np.nan
    out = pd.concat([matches, added], axis=1)
    eligible = out[["HomeKnownAt", "AwayKnownAt"]].notna().all(axis=1)
    source = {"file": Path(path).name, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              "rows": len(snapshots), "kind": "prematch_availability_snapshots", "eligible_paired_matches": int(eligible.sum())}
    return out, source


def availability_values(context, when, known_at=None):
    cutoff = pd.Timestamp(str(when)[:10]).tz_localize("Europe/London").tz_convert("UTC")
    if known_at is not None:
        cutoff = min(cutoff, pd.Timestamp(known_at))
    values = []
    for side in ["Home", "Away"]:
        raw = context.get(side + "KnownAt")
        observed = raw is not None and pd.notna(raw) and pd.Timestamp(raw) < cutoff
        injured = context.get(side + "Injured") if observed else None
        suspended = context.get(side + "Suspended") if observed else None
        observed = bool(observed and pd.notna(injured) and pd.notna(suspended))
        key = context.get(side + "KeyMissing") if observed else None
        key_observed = key is not None and pd.notna(key)
        values.append([float(injured) if observed else 0.0, float(suspended) if observed else 0.0,
                       float(observed), float(key) if key_observed else 0.0, float(key_observed)])
    h, a = values
    return [h[0], a[0], h[1], a[1], h[0] - a[0], h[1] - a[1], h[2], a[2], h[3], a[3], h[4], a[4]]

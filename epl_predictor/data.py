"""Strict local CSV ingestion. No credentials, scraping, or hidden downloads."""
from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd

REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
SHOT_COLUMNS = ["HS", "AS", "HST", "AST"]
DISCIPLINE_COLUMNS = ["HY", "AY", "HR", "AR", "HC", "AC", "HF", "AF"]
ODDS_COLUMNS = ["B365H", "B365D", "B365A"]
TOTALS_ODDS_COLUMNS = ["B365>2.5", "B365<2.5", "BbAv>2.5", "BbAv<2.5"]


def parse_dates(values):
    # Football-Data dates are day/month/year; ISO dates are accepted too.
    return pd.to_datetime(values, format="mixed", dayfirst=True, errors="raise").dt.normalize()


def load_matches(paths, quarantine_invalid_shots=False, include_totals_odds=False):
    odds_columns = ODDS_COLUMNS + TOTALS_ODDS_COLUMNS if include_totals_odds else ODDS_COLUMNS
    frames, provenance = [], []
    for raw in paths:
        path = Path(raw)
        df = pd.read_csv(path, encoding="utf-8-sig").dropna(how="all")
        explicit_season = "Season" in df
        missing = set(REQUIRED) - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        if "Div" in df and not df["Div"].eq("E0").all():
            raise ValueError(f"{path.name}: expected Premier League Div=E0 only")
        if df[REQUIRED].isna().any().any():
            raise ValueError(f"{path.name}: incomplete matches; use completed results only")
        df = df.copy()
        df["Date"] = parse_dates(df["Date"])
        if df["Date"].isna().any():
            raise ValueError(f"{path.name}: missing or invalid match dates")
        for col in ["HomeTeam", "AwayTeam"]:
            df[col] = df[col].astype(str).str.strip()
            if df[col].eq("").any():
                raise ValueError("Team names cannot be blank")
        for col in ["FTHG", "FTAG"]:
            df[col] = pd.to_numeric(df[col], errors="raise")
            if ((df[col] < 0) | (df[col] % 1 != 0) | ~np.isfinite(df[col])).any():
                raise ValueError("Goals must be finite nonnegative integers")
            df[col] = df[col].astype(int)
        result = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG == df.FTAG, "D", "A"))
        if "FTR" in df and not df["FTR"].eq(result).all():
            raise ValueError(f"{path.name}: FTR disagrees with the goals")
        df["FTR"] = result
        if df.HomeTeam.eq(df.AwayTeam).any():
            raise ValueError("A team cannot play itself")
        match = re.search(r"(\d{2})(\d{2})(?:\.download)?\.csv$", path.name)
        source_season = None
        if match and int(match[2]) == (int(match[1]) + 1) % 100:
            source_season = (1900 if int(match[1]) >= 90 else 2000) + int(match[1])
        if "Season" in df:
            df["Season"] = pd.to_numeric(df.Season.astype(str).str.slice(0, 4), errors="raise").astype(int)
        elif source_season is not None:
            df["Season"] = source_season
        else:
            df["Season"] = df.Date.dt.year - df.Date.dt.month.lt(7).astype(int)
        for season, group in df.groupby("Season"):
            # The COVID-interrupted 2019/20 season ended in July 2020.
            end = pd.Timestamp(year=int(season)+1, month=8 if season == 2019 else 7, day=1)
            start = pd.Timestamp(year=int(season), month=7, day=1)
            if (group.Date.lt(start) | group.Date.ge(end)).any():
                raise ValueError(f"Match dates disagree with the season in {path.name}; generic combined files need an explicit Season column")
        for col in SHOT_COLUMNS + DISCIPLINE_COLUMNS:
            if col not in df:
                df[col] = np.nan
            values = pd.to_numeric(df[col], errors="raise")
            known = values.dropna()
            if ((known < 0) | (known % 1 != 0) | ~np.isfinite(known)).any():
                raise ValueError("Shot/card/corner/foul statistics must be finite nonnegative integers when supplied")
            df[col] = values
        quarantined = []
        for total, target in [("HS", "HST"), ("AS", "AST")]:
            invalid = df[target] > df[total]
            if invalid.any():
                if not quarantine_invalid_shots:
                    raise ValueError("Shots on target cannot exceed total shots; use --quarantine-invalid-shots to retain results and explicitly mark the invalid pair missing")
                for _, row in df.loc[invalid].iterrows():
                    quarantined.append({"date": row.Date.strftime("%Y-%m-%d"), "home": row.HomeTeam, "away": row.AwayTeam,
                                        "columns": [total, target], "source_values": [float(row[total]), float(row[target])],
                                        "action": "both affected-side shot values marked missing; result retained"})
                df.loc[invalid, [total, target]] = np.nan
        for col in odds_columns:
            df[col] = pd.to_numeric(df[col], errors="raise") if col in df else np.nan
            known = df[col].dropna()
            if (~np.isfinite(known) | (known <= 1)).any():
                raise ValueError("Decimal bookmaker odds must be finite and greater than 1 when supplied")
        frames.append(df[REQUIRED + ["FTR", "Season"] + SHOT_COLUMNS + DISCIPLINE_COLUMNS + odds_columns])
        provenance.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "rows": len(df), "quarantined_stat_pairs": quarantined,
                           "season_source": "explicit column" if explicit_season else "season filename" if source_season is not None else "calendar fallback"})
    if not frames:
        raise ValueError("Provide at least one results CSV")
    data = pd.concat(frames, ignore_index=True).sort_values(
        ["Date", "HomeTeam", "AwayTeam"], kind="stable").reset_index(drop=True)
    if data.empty:
        raise ValueError("No completed matches supplied")
    if data.duplicated(["Date", "HomeTeam", "AwayTeam"]).any():
        raise ValueError("Duplicate matches found; avoid overlapping CSV files")
    appearances = pd.concat([data[["Date", "HomeTeam"]].rename(columns={"HomeTeam": "Team"}),
                             data[["Date", "AwayTeam"]].rename(columns={"AwayTeam": "Team"})])
    if appearances.duplicated(["Date", "Team"]).any():
        raise ValueError("A team appears twice on one date; check input data")
    if (data.Date > pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()).any():
        raise ValueError("Completed results cannot be dated in the future")
    return data, provenance


def attach_possession(matches, path):
    """Import explicitly timestamped historical observations, never inferred data."""
    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    keys = ["Date", "HomeTeam", "AwayTeam"]
    columns = keys + ["HomePossession", "AwayPossession", "AvailableAt"]
    if not set(columns).issubset(df.columns) or df[columns].isna().any().any():
        raise ValueError("Possession CSV requires complete Date, HomeTeam, AwayTeam, HomePossession, AwayPossession, AvailableAt")
    df = df[columns].copy()
    df["Date"] = parse_dates(df["Date"])
    if df.Date.isna().any() or df.duplicated(keys).any():
        raise ValueError("Invalid or duplicate possession match keys")
    for col in keys[1:]:
        df[col] = df[col].astype(str).str.strip()
    for col in ["HomePossession", "AwayPossession"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if (~np.isfinite(df[col]) | (df[col] < 0) | (df[col] > 100)).any():
            raise ValueError("Possession must be a percentage between 0 and 100")
    if not np.isclose(df.HomePossession + df.AwayPossession, 100, atol=1).all():
        raise ValueError("Home and away possession must sum to approximately 100")
    # Require explicit timezone information; pandas otherwise silently assumes UTC.
    if not df.AvailableAt.astype(str).str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True).all():
        raise ValueError("AvailableAt requires an explicit UTC offset or Z")
    available = pd.to_datetime(df.AvailableAt, utc=True, errors="raise", format="mixed")
    if available.isna().any() or (available.dt.tz_localize(None) < df.Date).any():
        raise ValueError("Possession availability cannot predate its match date")
    if (available > pd.Timestamp.now(tz="UTC")).any():
        raise ValueError("Possession availability cannot be in the future")
    df["AvailableAt"] = available.map(lambda x: x.isoformat())
    joined = df.merge(matches[keys], on=keys, how="left", indicator=True, validate="one_to_one")
    if not joined["_merge"].eq("both").all():
        raise ValueError("Possession contains unknown matches; use exact result dates and team names")
    merged = matches.merge(df, on=keys, how="left", validate="one_to_one").sort_values(keys).reset_index(drop=True)
    return merged, {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "rows": len(df), "kind": "timestamped_possession"}

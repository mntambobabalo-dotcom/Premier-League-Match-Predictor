"""Explicit cached downloads. Never called by training, prediction or tests."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

from .data import load_matches


def download_seasons(start, end, out_dir):
    current = datetime.now(timezone.utc)
    last = current.year if current.month >= 7 else current.year - 1
    if not 1993 <= start <= end <= last:
        raise ValueError(f"Choose valid season start years between 1993 and {last}")
    if end - start > 15:
        raise ValueError("Use a focused research period of at most 16 seasons per explicit request")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for year in range(start, end + 1):
        season = f"{year % 100:02d}{(year + 1) % 100:02d}"
        path = root / f"{season}.csv"
        url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
        cached = path.exists()
        if not cached:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "EPL-Personal-Research/0.3"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    raw = response.read(10_000_001)
                if len(raw) > 10_000_000 or not raw.lstrip(b"\xef\xbb\xbf").startswith(b"Div,"):
                    raise ValueError(f"Unexpected CSV response for {season}")
                staging = root / f"{season}.download.csv"
                staging.write_bytes(raw)
                try:
                    load_matches([staging], quarantine_invalid_shots=True)
                    staging.replace(path)
                finally:
                    staging.unlink(missing_ok=True)
            except (OSError, urllib.error.URLError) as exc:
                raise ValueError(f"Download failed for {season}: {exc}. Completed cached files remain reusable") from exc
        rows, provenance = load_matches([path], quarantine_invalid_shots=True)
        manifest.append({"season": f"{year}/{str(year + 1)[-2:]}", "file": path.name, "url": url,
                         "cached": cached, "rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "quarantined_stat_pairs_if_enabled": provenance[0]["quarantined_stat_pairs"]})
        print(f"{season}: {'cached' if cached else 'downloaded'} ({len(rows)} matches)", flush=True)
    report = {"checked_utc": current.isoformat(), "files": manifest,
              "note": "Private research source; check provider rights before commercial use. Existing files are reused without network requests. No scheduled refresh."}
    (root / "download_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

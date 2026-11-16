"""Local browser UI for the existing Premier League market model."""

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from epl_predictor.markets import forecast

ROOT = Path(__file__).resolve().parent
FIXTURES = [
    ("Man City", "Fulham"), ("Aston Villa", "Sunderland"),
    ("Bournemouth", "Nott'm Forest"), ("Chelsea", "Leeds"),
    ("Coventry", "Crystal Palace"), ("Tottenham", "Ipswich"),
    ("Newcastle", "Arsenal"),
]
MATCH_DATE = "2026-11-21"


def encode(value):
    return json.dumps(value, allow_nan=False).replace("<", "\\u003c")


def render_page(model, offline=False):
    # Archived example snapshots are permanently marked as such in the browser.
    examples = [
        forecast(model, home, away, MATCH_DATE, demo=True)
        for home, away in FIXTURES
        if model["history_through"] < MATCH_DATE and {home, away} <= model["teams"].keys()
    ]
    if offline and not examples:
        raise ValueError("The model cutoff must precede the seven example fixtures to export them.")
    config = {"offline": offline, "examples": examples, "teams": sorted(model["teams"]),
              "cutoff": model["history_through"]}
    return (ROOT / "web/index.html").read_text(encoding="utf-8").replace(
        "__BOOTSTRAP__", encode(config), 1
    )


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, model, page, **kwargs):
        self.model = model
        self.page = page
        super().__init__(*args, **kwargs)

    def respond(self, status, content, mime="application/json"):
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{mime}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/":
            self.respond(200, self.page, "text/html")
            return
        if url.path != "/api/forecast":
            self.respond(404, encode({"error": "Page unavailable."}))
            return
        query = parse_qs(url.query)
        try:
            home, away, when = (query.get(key, [""])[0] for key in ("home", "away", "date"))
            result = forecast(self.model, home, away, when)
        except ValueError as error:
            self.respond(400, encode({"error": str(error)}))
            return
        self.respond(200, encode(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/markets/model.json")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--export", type=Path, help="Save a standalone seven-fixture preview")
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    if model.get("schema") != "epl_markets_v1":
        parser.error("Use the market model from benchmark-markets.")
    if args.export:
        args.export.write_text(render_page(model, offline=True), encoding="utf-8")
        return
    server = partial(Handler, model=model, page=render_page(model))
    with ThreadingHTTPServer(("127.0.0.1", args.port), server) as http:
        print(f"Open http://127.0.0.1:{http.server_port}", flush=True)
        http.serve_forever()


if __name__ == "__main__":
    main()

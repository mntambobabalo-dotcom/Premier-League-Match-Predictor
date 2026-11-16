# Start here — Windows PowerShell / VS Code

Extract the ZIP and open the premier-league-predictor folder in VS Code. Open Terminal → New Terminal. Run these commands from the folder containing pyproject.toml. Python 3.10+ is required; no GPU or paid AI key is needed.

## Install and run the included model

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[boost]"
.\.venv\Scripts\python.exe -m epl_predictor markets --demo
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Using the environment's Python path directly avoids activation-policy problems. If Windows does not recognize python, use py -3 for the first command. In VS Code choose this .venv using Python: Select Interpreter.

The illustrative demo prints the score distribution, goals, double chance and cards. For a price check, append an example such as `--price 1X=1.45`. [Market guide](docs/MARKETS_V0_4.md)

## Fetch the 2015/16–2026/27 research files and retrain

```powershell
.\.venv\Scripts\python.exe -m epl_predictor download-data --start-season 2015 --end-season 2026 --out data
.\.venv\Scripts\python.exe -m epl_predictor data-check --data-dir data --quarantine-invalid-shots
.\.venv\Scripts\python.exe -m epl_predictor benchmark-markets --data-dir data --quarantine-invalid-shots --out artifacts/markets
.\.venv\Scripts\python.exe -m epl_predictor markets --demo --model artifacts/markets/model.json
```

Existing season files are reused without network access. The explicit quarantine flag records one impossible source shot pair as missing without altering the raw file or dropping the match result. See README for the exact row.

The boost extra installs XGBoost for the earlier outcome benchmarks. The market model runs with the base install `python -m pip install -e .`.

## macOS / Linux equivalent

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[boost]"
.venv/bin/python -m epl_predictor markets --demo
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m epl_predictor download-data --start-season 2015 --end-season 2026 --out data
.venv/bin/python -m epl_predictor benchmark-markets --data-dir data --quarantine-invalid-shots --out artifacts/markets
```

## Then open Codex

Read CODEX_HANDOFF.md and paste its opening task into Codex. AGENTS.md supplies the project rules. We are still building the Python engine; no app/backend is included.

The historical priced simulations returned −1.6% on 1X2 and −2.7% on over/under 2.5. Read [the market report](docs/MARKETS_V0_4.md) for sample sizes and limitations.

## Matchroom browser app

From this folder with the base Python package installed, run
`python web_app.py` and open http://127.0.0.1:8000. Use the environment's
Python executable if it is not activated. To export a standalone page with
the seven saved 21 November forecasts, run
`python web_app.py --export Matchroom.html`. The standalone page uses those
saved snapshots. Choose another fixture in the Python-served app for a fresh
forecast. Both interfaces accept manual odds and save paper slips in browser storage.

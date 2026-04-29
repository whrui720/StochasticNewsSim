# StochasticNewsSim — Quickstart

Code and data for the StochasticNewsSim paper. Two events covered:
**Las Vegas Shooting 2017** and **Hurricane Harvey 2017**.

## What's in this archive

- `requirements.txt` — pinned package versions (Python 3.12.3).
- `loader.py` — optional CCNews ingestion script (only needed to rebuild the CSVs from scratch).
- `events/` — TOML configs for the two events.
- `outputs/` — the per-event scored CSVs that the fitting scripts consume.
- `partAresults/` — Part A scripts (`fit_models.py`, `fit_hybrid.py`) and the
  per-event subdirectories with figures, fitted parameters, and comparison tables.
- `partBresults/` — Part B script (`fit_models_partB.py`) and per-event
  subdirectories.

## Environment

Tested on Python 3.12.3.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the paper's numbers (no data download)

The CSVs in `outputs/` are already included, so the four scripts re-fit
everything from scratch and rewrite the figures/tables under
`partAresults/<event>/` and `partBresults/<event>/`.

By default the scripts are configured to use the **Hurricane Harvey** CSV.
To run them, just:

```bash
cd partAresults && python fit_models.py && python fit_hybrid.py
cd ../partBresults && python fit_models_partB.py
```

To switch from Harvey to Las Vegas, edit the `DATA_PATH` and `OUT` constants
near the top of each script (instructions are inlined as a comment in each
file). For Las Vegas:

```python
DATA_PATH = "../outputs/las_vegas_shooting_2017_scored_part_0001_filtered.csv"
OUT       = "../partAresults/las_vegas_2017/"   # or partBresults
```

## Where to find the paper's numbers

Each per-event subdirectory contains:

- `model_comparison.csv` (and `model_comparison_with_hybrid.csv` for Part A) —
  log-likelihood, AIC, BIC, parameter counts.
- `fitted_params_all_models.json` — MLE parameter estimates (Part A only).
- `fitted_rates_all_models.csv` — per-day fitted intensities (Part A only).
- `fig*.png` — every figure used in the paper.

## Optional: rebuild the CSVs from CCNews

Only needed if you want to regenerate `outputs/*_scored_part_*.csv` from the
raw Common Crawl News stream. This requires the `datasets` package and takes
roughly an hour per event.

```bash
python loader.py --event-config events/las_vegas_shooting_2017.toml
python loader.py --event-config events/hurricane_harvey_2017.toml
```

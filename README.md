# StochasticNewsSim

Stochastic point-process models of news-article volume and reliability
distribution following major real-world events. Articles are pulled from the
[Common Crawl News (CCNews)](https://huggingface.co/datasets/stanford-oval/ccnews)
corpus and joined against the
[news_media_reliability](https://huggingface.co/datasets/sergioburdisso/news_media_reliability)
dataset. Daily counts and per-category counts are then fit with a family of
Poisson, inhomogeneous Poisson, and Hawkes-type models.

The repository contains:

- [loader.py](loader.py) — streams a year of CCNews, filters articles by
  event-specific keywords from a TOML config, joins with the reliability
  dataset, and writes scored CSV shards.
- [events/](events/) — one TOML file per event (Hurricane Harvey 2017,
  Las Vegas Shooting 2017).
- [partAresults/fit_models.py](partAresults/fit_models.py) and
  [partAresults/fit_hybrid.py](partAresults/fit_hybrid.py) — Part A models
  for total daily article counts: Standard Poisson, Inhomogeneous Poisson
  (IHP), discrete-time Hawkes, and an IHP+Hawkes hybrid.
- [partBresults/fit_models_partB.py](partBresults/fit_models_partB.py) —
  Part B models for the evolution of the reliability distribution
  (per-category daily counts): multivariate Hawkes (AR-1 limit),
  state-dependent Markov regime-switching, mean-field Markov, and a
  homogeneous-Markov baseline.
- [outputs/](outputs/) — scored CSV shards produced by `loader.py`.
- `partAresults/<event>/` and `partBresults/<event>/` — fitted parameters
  (`fitted_params_all_models.json`), per-day fitted rates
  (`fitted_rates_all_models.csv`), model comparison tables
  (`model_comparison.csv`), and figures (`fig*.png`).

## Environment

- **Python:** 3.12.3
- **Package versions** (see [requirements.txt](requirements.txt) for the
  pinned list):
  - pandas 3.0.2
  - numpy 2.4.4
  - scipy 1.17.1
  - matplotlib 3.10.8
  - datasets 4.8.4

The Hugging Face `datasets` package is only required to run `loader.py`
(it streams CCNews and downloads the reliability dataset). The fitting
scripts read the CSVs already in [outputs/](outputs/) and only need
pandas / numpy / scipy / matplotlib.

### Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the pipeline

### 1. Build the per-event CSV from CCNews

`loader.py` streams a CCNews year, filters rows whose title or body matches
keywords from the event TOML, looks up each publisher in the reliability
dataset, drops rows without a `newsguard_score`, and writes sharded CSVs
under [outputs/](outputs/).

```bash
python loader.py --event-config events/las_vegas_shooting_2017.toml
```

Useful flags:

- `--sample-size N` — only scan the first `N` CCNews rows (0 = full year).
- `--publish-cutoff YYYY-MM-DD` — override the cutoff in the TOML.
- `--rows-per-file N` / `--flush-every N` — tune sharding and checkpointing.
- `--resume` — resume from the JSON checkpoint written under
  `outputs/<prefix>.checkpoint.json`.

Each event's TOML lists two keyword groups:

- `keywords.anywhere` — phrases that may match anywhere in title or body.
- `keywords.title_only` — phrases too generic for body text, matched only
  against titles.

### 2. Fit the Part A models (total daily counts)

```bash
cd partAresults
python fit_models.py
python fit_hybrid.py
```

Both scripts have two constants near the top — `DATA_PATH` (the scored CSV
in `../outputs/`) and `OUT` (the output directory under `partAresults/`).
Edit them to switch events, e.g. for Las Vegas:

```python
DATA_PATH = "../outputs/las_vegas_shooting_2017_scored_part_0001_filtered.csv"
OUT       = "../partAresults/las_vegas_2017/"
```

Outputs:

- `fitted_params_all_models.json` — MLE parameters for each model.
- `fitted_rates_all_models.csv` — fitted λ(t) for each model on each day.
- `model_comparison.csv` and `model_comparison_with_hybrid.csv` —
  log-likelihood, AIC, BIC, parameter counts.
- `fig1_*.png` … `fig15_*.png` — fitted intensities, residuals, log-scale
  overlays, Hawkes kernel decomposition, β-profile sweep, hybrid model
  decomposition, and early-window zooms.

### 3. Fit the Part B models (reliability distribution)

```bash
cd partBresults
python fit_models_partB.py
```

Same `DATA_PATH` / `OUT` convention as Part A.

Outputs:

- `model_comparison.csv` — conditional binomial log-likelihood, AIC, BIC for
  each model.
- `fig1_*.png` … `fig9_*.png` — observed daily counts and proportion,
  fitted Hawkes branching matrix, fitted-vs-observed proportion, regime
  transition matrices, mean-field P(t) trajectory, binomial residuals, and
  cross-model comparison.

## Models

### Part A — total article volume

For each day `t`, the observed count is `N(t) ~ Poisson(λ(t))`.

| Model | Intensity |
|---|---|
| Standard Poisson | `λ(t) = λ̂` |
| Inhomogeneous Poisson (IHP) | `λ(t) = A·exp(−β·t) + c` |
| Hawkes (discrete-time) | `λ(t) = μ + α·R(t)`, `R(t) = exp(−β)·[R(t−1)+N(t−1)]` |
| Hybrid IHP + Hawkes | `λ(t) = A·exp(−β₀·t) + c + α·R(t)` |

All four are fit by maximum conditional Poisson log-likelihood with
`scipy.optimize.minimize` (L-BFGS-B). The Hawkes model is reparametrised in
`(μ, n, β)` with branching ratio `n = α/(exp(β)−1) ∈ (0,1)` to enforce
stationarity. A β-profile sweep shows the Hawkes log-likelihood is
monotonically increasing in β; the AR-1 limit `λ(t) ≈ μ + n·N(t−1)` is the
identifiable optimum and is what is reported.

### Part B — reliability distribution

Each article is labelled with `reliability_label ∈ {−1, +1}` derived from the
joined `newsguard_score`. Daily counts split into `N_neg(t)` and `N_pos(t)`.

| Model | Form |
|---|---|
| Multivariate Hawkes (AR-1 limit) | `λ_k(t) = μ_k + Σ_j n_{k,j}·N_j(t−1)` |
| State-dependent Markov | regime `x_t = argmax_k π_k(t)`, separate `P` per regime |
| Mean-field Markov | `P_{neg→pos}(t) = σ(a₀ + a₁·π_pos(t))`, `P_{pos→neg}(t) = σ(b₀ + b₁·π_pos(t))` |
| Homogeneous-Markov baseline | single constant `P` |

All families are scored on the same conditional binomial log-likelihood:
given the total daily count `N_tot(t)`, how well does each model predict
`N_pos(t)`? For the multivariate Hawkes,
`q_t = λ_pos(t) / (λ_pos(t) + λ_neg(t))`. A secondary Poisson log-likelihood
comparison (Hawkes vs per-category IHP) is also reported.

## Paper and presentation

- [StochasticNewsSim_paper.tex](StochasticNewsSim_paper.tex) /
  `StochasticNewsSim_paper.pdf` — full write-up.
- `StochasticNewsSim_presentation.pptx` — slides.

## Repository layout

```
StochasticNewsSim/                        # repo root
├── README.md
├── requirements.txt
├── loader.py                             # CCNews → per-event scored CSV
├── events/                               # one TOML per event
├── outputs/                              # scored CSV shards (output of loader)
├── partAresults/
│   ├── fit_models.py                     # Poisson / IHP / Hawkes
│   ├── fit_hybrid.py                     # IHP + Hawkes hybrid
│   ├── hurricane_harvey_2017/            # fitted params, rates, comparison, figs
│   └── las_vegas_2017/
└── partBresults/
    ├── fit_models_partB.py               # multivariate Hawkes / Markov family
    ├── hurricane_harvey_2017/
    └── las_vegas_2017/
```

# Part A: Stochastic News Simulation — Model Fitting Report
## Las Vegas Shooting 2017 Dataset

---

## 1. Dataset Overview

- **Event**: 2017 Las Vegas shooting (October 1, 2017)
- **Source file**: `las_vegas_shooting_2017_scored_part_0001_filtered.csv`
- **Date range**: 2017-10-02 → 2017-12-31 (91 days, zero-filled)
- **Total articles**: 2,702
- **Mean articles/day**: 29.69
- **Peak**: 489 articles on 2017-10-02 (day 0)
- **Time unit**: 1 day (t=0 corresponds to Oct 2, 2017 — earliest published article)

The raw data has gaps (days with no articles published); these are filled with zeros to form a complete 91-day time series before fitting.

---

## 2. Models

All three models are fit by **maximum conditional Poisson log-likelihood**, treating each day's count `N(t)` as drawn from a Poisson distribution with rate `λ(t)`. The full log-likelihood including the `log(N!)` term is used to ensure AIC/BIC comparisons are valid across models:

$$\ell = \sum_{t=0}^{T-1} \left[ N(t) \log \lambda(t) - \lambda(t) - \log N(t)! \right]$$

---

### Model A: Standard (Homogeneous) Poisson

**Form**: `λ(t) = λ`  (constant for all days)

**Parameters**: 1 (`λ`)

**MLE closed-form**: `λ̂ = mean(N) = 29.6923`

The standard Poisson assumes all days are independent and identically distributed — it has no mechanism to capture time-varying structure or temporal dependencies.

---

### Model B: Inhomogeneous Poisson (Exponential Decay)

**Form**: `λ(t) = A · exp(−β · t) + c`

**Parameters**: 3 (`A`, `β`, `c`)

- `A`: initial amplitude above baseline
- `β`: exponential decay rate
- `c`: long-run baseline rate

Each day's count is modeled as an independent Poisson draw, but the rate decays deterministically with absolute calendar time. This captures the "news cycle decay" — the phenomenon where media coverage of an event is highest immediately after and decays exponentially.

**Fitting method**: Nelder-Mead minimization of the negative log-likelihood. Multiple starting points used; convergence verified.

---

### Model C: Discrete-Time Hawkes Process

**Form**:
```
λ(t) = μ + α · R(t)
R(t) = exp(−β) · [R(t−1) + N(t−1)],   R(0) = 0
```

- `μ`: background (base) rate
- `α`: excitation coefficient
- `β`: kernel decay rate
- `R(t)`: accumulated weighted history of past counts

This is a **self-exciting process**: past article counts increase the rate of future articles. The Hawkes process is the natural model for endogenous cascades — one article triggers more articles. Unlike IHP, the intensity on day `t` is not a function of absolute calendar time, but depends on the full history of counts up to `t−1`. Days are **not** independent.

**Parameterisation**: To ensure stationarity and avoid optimization degeneracy, the model is re-parameterised as `(μ, n, β)` where:
$$n = \frac{\alpha}{\exp(\beta) - 1} \in (0, 1)$$
is the **branching ratio** — the expected number of secondary articles triggered per article. Stationarity requires `n < 1` and is guaranteed by construction.

---

## 3. Fitting Methodology

### Why not naive MLE on binned counts?

Treating each day's count as an independent Poisson observation (Poisson regression against time) would be wrong for the Hawkes model, where `λ(t)` is conditional on past counts. The correct approach is the **conditional log-likelihood** of the point process:

- For Standard Poisson and IHP: days are independent by assumption, so the log-likelihood is a sum of independent Poisson log-pmfs.
- For Hawkes: the log-likelihood is still a sum of Poisson log-pmfs, but `λ(t)` at each step is conditioned on `N(0), ..., N(t-1)`. This correctly captures the temporal dependence structure.

### Hawkes β-profile sweep

A key diagnostic was performed: for each fixed value of `β` on a log-grid from 0.05 to 32, the optimal `(μ, n)` were found by Nelder-Mead, and the profile log-likelihood was recorded:

| β     | μ̂       | n̂      | Log-Lik  |
|-------|---------|--------|---------|
| 0.05  | 17.66   | 0.424  | −3670.5 |
| 0.10  | 12.29   | 0.595  | −3165.5 |
| 0.20  | 11.15   | 0.630  | −2692.5 |
| 0.30  | 11.02   | 0.633  | −2472.4 |
| 0.50  | 11.07   | 0.630  | −2263.5 |
| 1.00  | 11.25   | 0.623  | −2092.6 |
| 2.00  | 11.40   | 0.617  | −2021.4 |
| 5.00  | 11.50   | 0.614  | −2004.4 |
| 10.00 | 11.50   | 0.614  | −2003.8 |
| 20.00 | 11.50   | 0.614  | −2003.8 |
| 32.00 | 11.50   | 0.614  | −2003.8 |

**The profile log-likelihood is monotonically increasing in β and plateaus for β ≥ 10.** This is a structural finding, not a numerical artifact.

**Interpretation**: As β → ∞, the exponential kernel `exp(−β·Δ)` collapses to a delta function at lag Δ=1. In this limit:
$$\lambda(t) \approx \mu + n \cdot N(t-1)$$
This is a discrete **AR(1) model** for the intensity. The optimal Hawkes process for this daily-resolution data is therefore a one-step lag model with branching ratio `n̂ = 0.614`. This is reported as the Hawkes result (2 free parameters: `μ` and `n`), since β is not identifiable from the plateau.

---

## 4. Results

### Fitted Parameters

| Model | Parameters | Values |
|---|---|---|
| Standard Poisson | λ | 29.69 articles/day |
| Inhomog. Poisson | A, β, c | A=443.47, β=0.2436, c=7.154 |
| Hawkes (AR-1) | μ, n | μ=11.50, n=0.6137 |

**IHP interpretation**: The exponential half-life is `ln(2)/β = 2.85 days`. Coverage halves approximately every 3 days in the first few weeks after the shooting.

**Hawkes interpretation**: Each article published today is expected to trigger `n = 0.614` additional articles the following day. The stationary mean is `μ/(1−n) = 11.50/0.386 = 29.78` articles/day, consistent with the observed mean.

---

### Model Comparison

| Model | # Params | Log-Likelihood | AIC | BIC | Pearson Var |
|---|---|---|---|---|---|
| Standard Poisson | 1 | −3835.11 | 7672.22 | 7674.73 | 196.3 |
| Inhomog. Poisson | 3 | **−513.15** | **1032.29** | **1039.82** | **8.52** |
| Hawkes (AR-1) | 2 | −2003.81 | 4011.62 | 4016.65 | 225.1 |

**Pearson dispersion** = Var[(N − λ)/√λ]. A perfectly fitting Poisson model would give a value ≈ 1. Higher values indicate underdispersion of the fit (the model's rate is too smooth relative to the data).

**Winner: Inhomogeneous Poisson** by a decisive margin.

---

### Likelihood Ratio Tests (vs Standard Poisson)

Both IHP and Hawkes are overwhelmingly preferred over Standard Poisson:

| Comparison | LR Statistic | df | p-value |
|---|---|---|---|
| SP vs IHP | 6643.9 | 2 | < 10⁻³⁰⁰ |
| SP vs Hawkes | 3662.6 | 1 | < 10⁻³⁰⁰ |

IHP vs Hawkes share 3 vs 2 parameters (no nesting), so AIC/BIC are used:
- ΔAIC (Hawkes − IHP) = **+2979.3** → IHP strongly preferred
- ΔBIC (Hawkes − IHP) = **+2976.8** → IHP strongly preferred

---

## 5. Why IHP Beats Hawkes: The Exogenous Shock Problem

The fundamental reason IHP outperforms the Hawkes process is the **exogenous initial shock**.

At `t=0` (Oct 2, 2017, the day after the shooting), the Hawkes process has no past events: `R(0) = 0`, so `λ(0) = μ = 11.50`. But the observed count is `N(0) = 489`.

The day-0 log-likelihood contributions are:
- **IHP**: `λ(0) = 443.47 + 7.15 = 450.62` → log-lik = **−5.6**
- **Hawkes**: `λ(0) = 11.50` → log-lik = **−1360.1**
- **Difference**: 1354.5, which is **~91% of the total log-likelihood gap** between the two models (1490.7 total)

The IHP model absorbs the initial burst because it has an explicit `A · exp(−β · 0) = A = 443.5` term that fits the spike on day 0. The Hawkes model, by design, must start from its baseline rate `μ` with no prior excitation.

**This is a structural mismatch**: the Hawkes process models endogenous self-excitation (articles beget articles), but the initial surge of coverage is an *exogenous* response to the event itself. There were no prior articles to trigger day 0's 489 publications.

---

## 6. Physical Interpretation

### What the models say about news dynamics

**Standard Poisson** says nothing useful — a flat rate of 29.7 articles/day cannot capture the spike or the decay. It is included as a null model only.

**Inhomogeneous Poisson** says: news coverage follows a deterministic exponential decay with half-life 2.85 days, starting from an amplitude of ~443 articles above a long-run baseline of ~7 articles/day. The decay is a function of *how long ago the event happened*, independent of how much coverage occurred in between.

**Hawkes Process** says: 61.4% of the articles published today will directly cause additional articles tomorrow (editorial follow-ups, response pieces, counter-arguments). The background rate `μ = 11.5` represents the "spontaneous" article generation rate if no recent coverage had occurred. The long-run average (`μ/(1−n) = 29.8`) matches the data.

Both IHP and Hawkes are meaningful models with different mechanistic assumptions:
- IHP: coverage decays because the *event itself* becomes less newsworthy over time
- Hawkes: coverage persists because *journalists cover other journalists' coverage*

For this data, the IHP mechanism dominates, but the significant Hawkes branching ratio (n̂ = 0.614) confirms that self-excitation is also present.

### What would make Hawkes win?

The Hawkes process would outperform IHP if:
1. The data had no large exogenous spike at `t=0` (i.e., coverage built up endogenously over time)
2. The event triggered multiple waves of coverage rather than a single exponential decay
3. A **Hawkes + exogenous background** model were used: `λ(t) = A·exp(−β₀·t) + c + α·R(t)`. This hybrid would absorb both the initial shock (via IHP term) and self-excitation (via Hawkes term), and would likely outperform both pure models.

---

## 7. Generated Output Files

| File | Description |
|---|---|
| `fit_models.py` | Full fitting and plotting script |
| `model_comparison.csv` | Summary table (params, LL, AIC, BIC, Pearson var) |
| `fig1_fitted_intensities.png` | Three separate panels: observed counts + each model's λ(t) |
| `fig2_overlay.png` | All three models overlaid on one plot |
| `fig3_model_criteria.png` | Bar charts of AIC, BIC, log-likelihood |
| `fig4_residuals.png` | Residuals (observed − fitted) per model |
| `fig5_pearson_residuals.png` | Pearson residual histograms |
| `fig6_logscale.png` | Log-scale comparison (shows tail/decay behaviour) |
| `fig7_hawkes_decomposition.png` | Hawkes background vs self-excitation stackplot |
| `fig8_hawkes_beta_profile.png` | Profile log-likelihood of Hawkes vs β (key diagnostic) |
| `fig9_early_zoom.png` | Zoom on Oct 2–15 (first two critical weeks) |

---

## 8. Conclusions

1. **All three models significantly outperform each other in the expected direction**: Standard Poisson < Hawkes < IHP by AIC/BIC.

2. **IHP is the best-fitting model** (AIC=1032 vs Hawkes=4012 vs SP=7672). The deterministic exponential decay with half-life ~2.85 days provides the best description of the Las Vegas shooting news cycle.

3. **The hypothesis that Hawkes fits best is not supported for this dataset**, primarily because the shooting creates an exogenous shock that the Hawkes process cannot account for at `t=0` (a single day accounts for 91% of the Hawkes–IHP gap).

4. **Self-excitation is present** (Hawkes branching ratio n̂=0.614, strongly significant vs Standard Poisson with ΔAIC=3660), but it is not the dominant mechanism. The news cycle is mainly driven by exogenous decay, not endogenous spreading.

5. **The optimal Hawkes kernel for daily resolution data is a single-step AR(1)**: the profile likelihood sweep confirms that the Hawkes exponential kernel collapses to `λ(t) = 11.50 + 0.614·N(t-1)`. This is a robust, identifiable 2-parameter result.

6. A **hybrid model** combining IHP's deterministic decay with Hawkes self-excitation would likely outperform both pure models and is a natural direction for future work.

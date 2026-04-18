# Part A: Stochastic News Simulation — Model Fitting Report
## Las Vegas Shooting 2017 Dataset

> **Update:** §7 adds a hybrid IHP+Hawkes model that outperforms all three base models (AIC = 752.80 vs IHP's 1032.29). The hybrid decomposes the news cycle into an exogenous trigger (IHP term) and endogenous self-excitation (Hawkes term) — see §7 for details.

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

## 7. Hybrid IHP + Hawkes Model

Section 5 conjectured that a hybrid combining IHP's exogenous decay with Hawkes self-excitation would outperform both pure models. This section confirms and quantifies that claim.

### 7.1 Model Specifications

Two hybrid variants are fit:

**Hybrid (AR-1 limit)** — 4 parameters:
$$\lambda(t) = A \cdot e^{-\beta_0 t} + c + n \cdot N(t-1)$$

Adds the `n·N(t−1)` self-excitation term (the Hawkes AR-1 limit from §3) on top of the IHP deterministic decay. At `t=0`, there are no past events, so `λ(0) = A + c` — the hybrid can still absorb the day-0 exogenous shock.

**Hybrid (full)** — 5 parameters:
$$\lambda(t) = A \cdot e^{-\beta_0 t} + c + \alpha \cdot R(t), \qquad R(t) = e^{-\beta_1}[R(t-1) + N(t-1)]$$

Full exponential-kernel Hawkes history `R(t)` instead of a single-lag shortcut. `β₁` is a free parameter (unlike pure Hawkes, where it was unidentifiable because no competing fit mechanism existed). The branching ratio is `n = α / (exp(β₁) − 1)`.

Both are fit by maximum conditional Poisson log-likelihood with the same `gammaln` normalisation as in §3, from a grid of starting points to avoid local minima.

### 7.2 Fitted Parameters

| Model | Parameters | Values |
|---|---|---|
| Hybrid (AR-1)  | A, β₀, c, n              | A=486.78, β₀=2.217, c=2.208, n=0.7248 |
| Hybrid (full)  | A, β₀, c, α, β₁          | A=538.23, β₀=0.397, c=1.597, α=0.0305, β₁=0.0848 (n=0.344) |

**Hybrid-AR1 interpretation**: The IHP term effectively collapses to a one-day spike (`β₀ = 2.22` → half-life 0.31 days, so `A·exp(−β₀·1) ≈ 53`, and by day 2 ≈ 6). The model attributes the 489 articles on day 0 almost entirely to the exogenous event, then hands propagation off to the self-excitation term with branching ratio `n = 0.725` — higher than the pure Hawkes `n = 0.614`, because the hybrid isn't being forced to overfit day 0 through branching. This is a clean decomposition: **IHP term = instantaneous exogenous trigger, Hawkes term = endogenous amplification and decay**.

**Hybrid-full interpretation**: Different regime. The exponential kernel has `β₁ = 0.085`, giving a Hawkes memory half-life of `ln(2)/β₁ ≈ 8.2 days` — much longer than one day. The IHP decay `β₀ = 0.397` (half-life 1.75 days) handles the fast initial decline; the long-memory Hawkes term (`n = 0.344`) carries the longer tail. Two separable timescales: fast exogenous decay + slow endogenous persistence.

### 7.3 Full Comparison — All Five Models

| Model | # Params | Log-Likelihood | AIC | BIC | Pearson Var |
|---|---|---|---|---|---|
| Standard Poisson         | 1 | −3835.11 | 7672.22 | 7674.73 | 196.26 |
| Inhomogeneous Poisson    | 3 |   −513.15 | 1032.29 | 1039.82 |   8.52 |
| Hawkes (AR-1)            | 2 | −2003.81 | 4011.62 | 4016.65 | 225.10 |
| **Hybrid IHP+Hawkes (AR-1)** | 4 |   −388.79 |  785.59 |  795.63 |   5.10 |
| **Hybrid IHP+Hawkes (full)** | 5 | **−371.40** | **752.80** | **765.35** | **4.48** |

**The full Hybrid wins on every criterion.** Against the previous best (pure IHP):

- ΔAIC (Hybrid-AR1 − IHP)  = **−246.70**
- ΔAIC (Hybrid-full − IHP) = **−279.49**
- ΔBIC (Hybrid-full − IHP) = **−274.47**

Pearson dispersion drops from 8.52 (IHP) to 4.48 (Hybrid-full) — roughly halved, though still above the Poisson ideal of 1 (the residual overdispersion is daily count-noise unexplained by smooth deterministic rates).

### 7.4 Likelihood Ratio Tests

Hybrid-AR1 strictly nests IHP (set `n=0`) and nests Hawkes-AR1 (drop the IHP term, replace `c` with `μ`). Hybrid-full strictly nests Hybrid-AR1 (take `β₁ → ∞`).

| Comparison | LR Statistic | df | p-value |
|---|---|---|---|
| IHP vs Hybrid-AR1         | 248.70 | 1 | ≈ 0 |
| Hybrid-AR1 vs Hybrid-full |  34.79 | 1 | 3.7 × 10⁻⁹ |

Both additions are massively significant. The richer Hawkes kernel (β₁ finite rather than the AR-1 limit) pays for its extra parameter many times over.

### 7.5 Day-0 Decomposition Revisited

Section 5 showed that Hawkes lost to IHP because Hawkes predicts `λ(0) = μ` (~11.5) when the observation is 489. The hybrid fixes this structurally:

| Model | λ(0) | Day-0 log-lik |
|---|---|---|
| IHP              | 450.6 |    −5.60 |
| Hawkes (AR-1)    |  11.5 | −1360.13 |
| Hybrid (AR-1)    | 489.0 |    **−4.02** |
| Hybrid (full)    | 539.8 |    −6.49 |

Hybrid-AR1 actually lands *exactly* on the observed 489 (`A + c = 486.78 + 2.21 = 488.99`) — the optimiser effectively fits the initial amplitude to the observation, then lets self-excitation handle the rest of the series. Hybrid-full slightly overshoots on day 0 but gains elsewhere through the richer kernel.

### 7.6 Why the Hybrid Works

The hybrid is not just a flexible regression — it matches the underlying mechanism. The Las Vegas shooting news cycle has two distinct processes operating at different timescales:

1. **Exogenous trigger** — the event itself generates a first-day wave of direct coverage. This is not triggered by prior articles; it is a response to the shooting. Modelled by `A·exp(−β₀·t) + c`.
2. **Endogenous amplification** — articles beget articles: editorials respond to news reports, opinion pieces respond to editorials, counter-arguments respond to opinions. Modelled by `α·R(t)` / `n·N(t−1)`.

Pure IHP captures only (1); pure Hawkes captures only (2) and fails on day 0. The hybrid captures both and outperforms both.

### 7.7 Known Limitation — Weekly Seasonality

Neither hybrid variant expresses **day-of-week structure**, which the data clearly exhibits. Diagnostics on the IHP-model Pearson residuals show strong weekly periodicity:

- **Day-of-week mean Pearson residuals**: Wed +2.07, Thu +1.26, Fri +0.42, Mon −0.05, Tue −0.11, Sun −1.63, Sat −1.94. Weekdays (especially Wed/Thu) systematically exceed the smooth rate; weekends fall well below it.
- **Residual autocorrelation**: lag-1 ρ = 0.59, **lag-7 ρ = 0.61**, lag-14 ρ = 0.39. The lag-7 spike is as strong as lag-1 — textbook weekly periodicity.

This explains a visual artifact of fig10: Hybrid-AR1 appears to "track" the daily bumps more closely than Hybrid-full, but this is largely because the lag-1 term `n·N(t−1)` acts as a crude proxy for day-of-week (consecutive days share DOW-neighbour structure), whereas the smooth exponential Hawkes kernel of Hybrid-full is monotone and cannot express a 7-day period. Hybrid-full still wins on LL / AIC because its smoothness avoids large Poisson penalties on days where AR-1's noise-echoing misfires — but neither model captures the weekly cycle explicitly, and the visible grouping is a real structural signal that both are missing.

A quick-exploration fit adding six multiplicative day-of-week factors (one free baseline, Σ log w = 0) confirms this:

| Model | # Params | AIC | ΔAIC vs no-weekly |
|---|---|---|---|
| Hybrid-AR1 (no weekly) | 4  | 785.59 | — |
| Hybrid-full (no weekly) | 5  | 752.80 | — |
| Hybrid-AR1 + weekly    | 10 | 665.56 | −120.03 |
| Hybrid-full + weekly   | 11 | 604.68 | **−148.12** |

Weekly factors from Hybrid-full + weekly (multiplicative): Mon 1.03, Tue 1.16, **Wed 1.46**, Thu 1.32, Fri 1.16, **Sat 0.54**, Sun 0.69 — a ~2.7× Wednesday/Saturday ratio. Once the weekly term is added, the full-kernel advantage over AR-1 widens (ΔAIC −60.88 vs −32.79 before), consistent with the interpretation that AR-1's "bump tracking" was a crude substitute for seasonality rather than a genuine dynamical advantage.

The weekly-seasonal extension is not reported as a main model here — the exploration is kept as a diagnostic. A proper model would jointly fit the exogenous decay, endogenous self-excitation, and weekly periodicity; this is a natural direction for extending the analysis.

### 7.8 Figures for §7

- `fig10_hybrid_fits.png` — two-panel: Hybrid-AR1 and Hybrid-full vs observed counts
- `fig11_all_models_overlay.png` — all five models on one plot
- `fig12_all_models_criteria.png` — AIC/BIC/LL bar charts for all five models
- `fig13_hybrid_decomposition.png` — Hybrid-AR1 stackplot: IHP component vs self-excitation component
- `fig14_hybrid_early_zoom.png` — Oct 2–15 zoom: hybrid tracks both the day-0 spike and the follow-up dynamics
- `fig15_hybrid_residuals.png` — residuals: IHP vs Hawkes vs Hybrid-AR1

---

## 8. Generated Output Files

| File | Description |
|---|---|
| `fit_models.py` | Part-A fitting script (SP, IHP, Hawkes) |
| `fit_hybrid.py` | Hybrid IHP+Hawkes fitting script (§7) |
| `model_comparison.csv` | Summary table for SP / IHP / Hawkes |
| `model_comparison_with_hybrid.csv` | Summary table with hybrid models added |
| `fitted_rates_all_models.csv` | Per-day fitted λ(t) for all five models |
| `fitted_params_all_models.json` | All fitted parameters, machine-readable |
| `fig1_fitted_intensities.png` | Three separate panels: observed counts + each model's λ(t) |
| `fig2_overlay.png` | All three base models overlaid on one plot |
| `fig3_model_criteria.png` | Bar charts of AIC, BIC, log-likelihood (base models) |
| `fig4_residuals.png` | Residuals (observed − fitted) per model |
| `fig5_pearson_residuals.png` | Pearson residual histograms |
| `fig6_logscale.png` | Log-scale comparison (shows tail/decay behaviour) |
| `fig7_hawkes_decomposition.png` | Hawkes background vs self-excitation stackplot |
| `fig8_hawkes_beta_profile.png` | Profile log-likelihood of Hawkes vs β (key diagnostic) |
| `fig9_early_zoom.png` | Zoom on Oct 2–15 (first two critical weeks) |
| `fig10_hybrid_fits.png` | Hybrid-AR1 and Hybrid-full fits vs observed |
| `fig11_all_models_overlay.png` | All five models overlaid |
| `fig12_all_models_criteria.png` | AIC/BIC/LL bar charts for all five models |
| `fig13_hybrid_decomposition.png` | Hybrid-AR1 decomposition stackplot |
| `fig14_hybrid_early_zoom.png` | Oct 2–15 zoom with hybrid models |
| `fig15_hybrid_residuals.png` | Residuals: IHP vs Hawkes vs Hybrid |

---

## 9. Conclusions

1. **Among the three base models**, the ranking by AIC/BIC is Standard Poisson < Hawkes < IHP. IHP wins because it absorbs the day-0 exogenous shock via `A·exp(0)`, while Hawkes must start from `λ(0) = μ` with no prior excitation — this single day accounts for ~91% of the Hawkes–IHP log-likelihood gap.

2. **Self-excitation is real but secondary in the base fit**. Hawkes branching ratio `n̂ = 0.614` is strongly significant vs Standard Poisson (ΔAIC = 3660), yet pure Hawkes loses to IHP because it has no mechanism to model exogenous shocks. The optimal Hawkes kernel for daily-resolution data is a single-step AR(1): the profile-likelihood sweep confirms the exponential kernel collapses to `λ(t) = 11.50 + 0.614·N(t−1)` as `β → ∞`.

3. **The hybrid IHP+Hawkes model is the overall winner** (§7). Hybrid-full achieves AIC = 752.80, BIC = 765.35 — an improvement of ΔAIC = **−279.5** over pure IHP and ΔAIC = **−3258.8** over pure Hawkes. Hybrid-AR1 already gets most of the gain (AIC = 785.59) with only four parameters, and LR tests confirm both additions (IHP → Hybrid-AR1, Hybrid-AR1 → Hybrid-full) are significant at p ≪ 10⁻⁸.

4. **The hybrid decomposition matches the physical mechanism**. In Hybrid-AR1, the IHP term collapses to a near-delta spike (`A ≈ 487, β₀ = 2.22`) that absorbs the exogenous day-0 burst almost exactly (`λ(0) = 489.0 = N(0)`), and the Hawkes term (`n = 0.725`) carries the subsequent endogenous dynamics. This is a clean separation between **exogenous trigger** and **endogenous amplification**. The full hybrid refines this with two distinct decay timescales (IHP half-life 1.75 days + Hawkes kernel half-life 8.2 days), matching the intuition that the news cycle has both a fast initial decay and a slower self-sustaining tail.

5. **Takeaway for news-count modelling**: neither pure deterministic decay nor pure self-excitation is enough on its own. A superposition is necessary whenever an event combines a large exogenous trigger (e.g. a breaking-news shock) with ongoing endogenous commentary (editorials, follow-ups, counter-arguments). The hybrid is the natural structural model for this.

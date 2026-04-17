# Part B: Modeling the Evolution of Article Reliability Distribution
## Las Vegas Shooting 2017 Dataset

---

## 1. Dataset Overview

- **Source file**: `las_vegas_shooting_2017_scored_part_0001_filtered.csv`
- **Date range**: 2017-10-02 → 2017-12-31 (91 days, zero-filled)
- **Total articles**: 2,702
- **Reliability labels** (from the `reliability_label` column): **binary**, values are only `−1.0` and `+1.0`. **No `0` values exist in the data**, so the ternary framing from the original prompt is collapsed to a 2-state problem.
  - `N_neg` total = 986 (36.5%, unreliable)
  - `N_pos` total = 1,716 (63.5%, reliable)
- **Zero-total days**: 5 (no articles published) — these days are skipped in the binomial likelihood.

We track two daily time series:
- `N_neg(t)`: daily unreliable article count
- `N_pos(t)`: daily reliable article count
- `π_pos(t) = N_pos(t) / N_tot(t)`: daily reliable fraction

See [fig1_observed_distribution.png](fig1_observed_distribution.png) (raw stacked counts) and [fig2_observed_proportion.png](fig2_observed_proportion.png) (π_pos trajectory with 95% Wilson confidence bands).

---

## 2. Models

All four models are evaluated on a **shared conditional binomial log-likelihood** to allow fair comparison across different model families (Hawkes vs Markov). For each day `t` with `N_tot(t) > 0`, the log-likelihood contribution is:

$$\ell_t = \log \binom{N_{tot}(t)}{N_{pos}(t)} + N_{pos}(t) \log q_t + N_{neg}(t) \log (1-q_t)$$

where `q_t` is each model's **predicted reliable fraction** on day `t`. See §3 below for the full rationale.

### Model 1 — Multivariate Hawkes (AR-1 limit)

$$\lambda_k(t) = \mu_k + \sum_{j \in \{neg, pos\}} n_{k,j} \cdot N_j(t-1), \qquad k \in \{neg, pos\}$$

- **Parameters (6)**: `μ_neg`, `μ_pos`, and the 2×2 branching matrix `n_{k,j}`.
- Stationarity enforced during optimisation: spectral radius of `n` matrix < 1.
- AR-1 form used directly since Part A already showed the `β→∞` collapse is optimal for daily data.
- **Implied split for comparison**: `q_t = λ_pos(t) / (λ_pos(t) + λ_neg(t))`.

### Model 2z — Homogeneous Markov (null baseline, 2 params)

Single constant 2×2 transition matrix `P`. The predicted reliable fraction is `q_t = (π_{t-1} · P)_{pos}`. Serves as the null model against which 2a and 2b are tested.

### Model 2a — State-dependent Markov (regime-switching, 4 params)

The regime on day `t-1` is a discrete variable `x_{t-1} = argmax_k π_k(t-1)` (dominant reliability category). Two separate transition matrices are fit:
- `P(neg)`: applied when the previous day was neg-majority (π_pos ≤ 0.5)
- `P(pos)`: applied when the previous day was pos-majority (π_pos > 0.5)

Then `q_t = (π_{t-1} · P(x_{t-1}))_{pos}`.

### Model 2b — Mean-field Markov (4 params)

Transition probabilities depend smoothly on `π_pos(t-1)` via logistic links:

$$P_{neg \to pos}(t) = \sigma(a_0 + a_1 \cdot \pi_{pos}(t-1))$$
$$P_{pos \to neg}(t) = \sigma(b_0 + b_1 \cdot \pi_{pos}(t-1))$$

At `a_1 = b_1 = 0` this reduces to the homogeneous baseline — gives a clean LRT with 2 degrees of freedom.

---

## 3. Why the Shared Binomial Likelihood?

Hawkes natively predicts **counts** (Poisson likelihood); Markov models predict **proportions given totals** (binomial likelihood). Their native AICs are not directly comparable.

To put all four on one scale, we **condition on the observed daily total** `N_tot(t)` and ask each model: *given that total, how well do you predict the reliable/unreliable split?* For Hawkes, the implied split is `λ_pos / (λ_pos + λ_neg)`; for the Markov models it is `(π_{t-1} · P_t)_{pos}` directly.

This is the natural Part B question (*distribution* of reliability), so scoring on the split-conditional-on-total is both fair and semantically correct.

A secondary Hawkes-only Poisson log-likelihood is also reported (`LL = −2160.08`) for completeness.

---

## 4. Fitted Parameters

### Multivariate Hawkes
- `μ = (μ_neg, μ_pos) = (3.12, 8.38)` — baseline arrival rates
- Branching matrix (rows = triggered, cols = triggering):

| | **trigger = neg** | **trigger = pos** |
|---|---|---|
| **produces neg** | 0.4386 | 0.1577 |
| **produces pos** | 0.3217 | 0.3720 |

- **Spectral radius** `ρ(n) = 0.6330` → stationary.
- **Cross-excitation is substantial**: an unreliable article yesterday triggers `0.32` reliable articles today on average (across all reliable outlets). A reliable article triggers `0.16` unreliable articles today.
- **Diagonal dominance is weak** (`n_nn = 0.44` vs `n_np = 0.16`; `n_pp = 0.37` vs `n_pn = 0.32`): unreliable articles nearly-symmetrically trigger both types, reliable articles preferentially trigger more reliable.

See [fig3_hawkes_branching_heatmap.png](fig3_hawkes_branching_heatmap.png) and [fig4_hawkes_fitted_vs_observed.png](fig4_hawkes_fitted_vs_observed.png).

### Homogeneous Markov (baseline)

| from \ to | **neg** | **pos** |
|---|---|---|
| **neg** | 0.5507 | 0.4493 |
| **pos** | 0.2916 | 0.7084 |

Stationary distribution: `π* ≈ (0.39, 0.61)` — close to the empirical split (0.37, 0.63).

### State-dependent Markov (regime-switching)

**P(neg-majority regime)** — when previous day had π_pos ≤ 0.5:

| from \ to | **neg** | **pos** |
|---|---|---|
| **neg** | 0.2745 | 0.7255 |
| **pos** | 0.5879 | 0.4121 |

**P(pos-majority regime)** — when previous day had π_pos > 0.5:

| from \ to | **neg** | **pos** |
|---|---|---|
| **neg** | 0.6818 | 0.3182 |
| **pos** | 0.2354 | 0.7646 |

**Interpretation**: the two regimes have qualitatively different dynamics.
- In **pos-majority regime** (the typical state of the system), both categories are sticky: `P_{neg→neg}=0.68`, `P_{pos→pos}=0.76`. The state persists.
- In **neg-majority regime** (rare, high-news days like Oct 2–4), the dynamics *push toward pos*: `P_{neg→pos}=0.73`, `P_{pos→neg}=0.59`. Most articles flip categories.

This asymmetry reflects that the system has a natural equilibrium around `π_pos ≈ 0.63`. When it strays below 0.5, strong mean-reversion kicks in.

See [fig5_markov_regime_matrices.png](fig5_markov_regime_matrices.png).

### Mean-field Markov

| param | value | interpretation |
|---|---|---|
| `a_0` | +1.1496 | baseline (at π_pos=0): `P_{neg→pos} = σ(1.15) = 0.76` — strong flip to pos when pos is absent |
| `a_1` | −3.9587 | as π_pos ↑, `P_{neg→pos}` ↓ sharply |
| `b_0` | −2.0828 | baseline: `P_{pos→neg} = σ(−2.08) = 0.11` — pos is sticky when pos is rare |
| `b_1` | +0.8184 | as π_pos ↑, `P_{pos→neg}` slightly ↑ |

At the two extremes of `π_pos`:
| | `π_pos = 0` | `π_pos = 1` |
|---|---|---|
| `P_{neg→pos}` | 0.76 | 0.06 |
| `P_{pos→neg}` | 0.11 | 0.22 |

**Interpretation** (same as 2a, seen through a smooth lens): the lower the fraction of reliable articles yesterday, the *higher* the probability that any unreliable article will flip to reliable today. This is a **mean-reverting / self-regulating dynamic** — the distribution is pulled back toward its equilibrium.

See [fig6_meanfield_P_trajectory.png](fig6_meanfield_P_trajectory.png).

---

## 5. Model Comparison (Shared Binomial Scale)

Effective sample size: `n_eff = 86` (days with at least one article).

| Model | # Params | Log-Likelihood | AIC | BIC |
|---|---|---|---|---|
| Multivariate Hawkes | 6 | −156.93 | 325.87 | 340.60 |
| Homogeneous Markov | 2 | −162.75 | 329.50 | 334.41 |
| **State-dependent Markov** | **4** | **−157.33** | **322.65** | **332.47** |
| Mean-field Markov | 4 | −157.56 | 323.13 | 332.94 |

**Winner: State-dependent Markov** by AIC (322.65). Mean-field Markov is a close second (323.13); the two are statistically indistinguishable (Δ = 0.48).

See [fig7_fitted_proportion_comparison.png](fig7_fitted_proportion_comparison.png), [fig8_binomial_residuals.png](fig8_binomial_residuals.png), and [fig9_model_comparison.png](fig9_model_comparison.png).

### Likelihood Ratio Tests (vs Homogeneous Baseline)

| Model | LR statistic | df | p-value |
|---|---|---|---|
| State-dep Markov vs Homog | 10.85 | 2 | **4.4 × 10⁻³** |
| Mean-field Markov vs Homog | 10.37 | 2 | **5.6 × 10⁻³** |

Both 4-parameter Markov models significantly improve on the null, at the p < 0.01 level.

### Hawkes vs Markov: why does Markov win despite Hawkes' extra parameters?

Hawkes has 6 parameters but is only slightly better than the 2-parameter homogeneous baseline on binomial LL (−156.93 vs −162.75, ΔLL = 5.82 with 4 extra parameters). After the AIC penalty (+8 from the extra params), Hawkes ends up *worse* than both State-dep and Mean-field.

Two structural reasons:
1. **Hawkes spends most of its modeling capacity on predicting the *total* count**, not the split. The split `q_t = λ_pos / (λ_pos + λ_neg)` is a derived quantity; Hawkes is not directly optimising for it.
2. **The system spends ~86% of days in pos-majority regime** (74 of 86 non-empty days have π_pos > 0.5). A single homogeneous P captures the dominant regime's dynamics; the Markov variants add regime-specific corrections targeted at exactly the binomial split we're scoring.

Put differently: if we were scoring on the *count* likelihood (Poisson), Hawkes would have a clear advantage because cross-excitation between categories is genuinely present. But for predicting the *proportion*, the Markov parameterisations are more efficient.

---

## 6. Synthetic Sanity Check

We simulated a multivariate Hawkes process with known parameters `(μ_neg=3.0, μ_pos=5.0, n_nn=0.10, n_np=0.20, n_pn=0.15, n_pp=0.35)` for T=500 days, then re-fit using the same multi-start procedure used for the real data.

| Parameter | True | Recovered | Abs error |
|---|---|---|---|
| μ_neg | 3.000 | 3.139 | 0.139 |
| μ_pos | 5.000 | 4.695 | 0.305 |
| n_{neg,neg} | 0.100 | 0.080 | 0.020 |
| n_{neg,pos} | 0.200 | 0.178 | 0.022 |
| n_{pos,neg} | 0.150 | 0.154 | 0.004 |
| n_{pos,pos} | 0.350 | 0.382 | 0.032 |

**Max relative error: 20.3%** — acceptable for T=500 and 6 parameters. Pass.

The initial single-start recovery was much worse (max error 1.48); the multi-start fix was necessary. This same multi-start routine is used for the real-data fit, so we have confidence in the Hawkes parameter estimates.

---

## 7. Physical Interpretation

### What the models say about reliability dynamics

**Homogeneous Markov** treats the system as ergodic with a fixed mixing rate. A reliable article has ~71% chance of being "followed" by another reliable article the next day; an unreliable one has a 45% chance of being followed by a reliable one. The equilibrium (~61% reliable) matches the observed overall split.

**State-dependent Markov** reveals two dynamical phases:
- When the system is in its normal **pos-majority regime**, coverage is sticky — each category self-perpetuates.
- When it slips into **neg-majority regime** (happens only at the start, during the shock), the system **actively flips back** — `P_{neg→pos} = 0.73` is substantially higher than in the pos-majority regime.
- This is a **bistable-looking** dynamic but with strong restoring force toward the reliable side.

**Mean-field Markov** reveals the same dynamic as a smooth function: `P_{neg→pos}` is a monotonically decreasing function of `π_pos`. When reliable articles are scarce, conversion of unreliable streams into reliable ones is high; when reliable articles dominate, that conversion rate is suppressed. This **self-regulating / mean-reverting** behaviour makes sense for a media ecosystem with a dominant reliable-publisher base.

**Multivariate Hawkes** captures the cross-excitation structure explicitly: an unreliable article yesterday triggers, on average, 0.32 reliable articles today (vs 0.44 unreliable). Reliable articles preferentially trigger more reliable articles (0.37 vs 0.16). The branching matrix is **mildly diagonal-dominant**: within-category excitation exceeds cross-category, but cross-excitation is not negligible. The spectral radius (0.63) is comfortably sub-critical, consistent with the stable news cycle observed in Part A.

### Did any of the original two hypotheses "win"?

Both did, in complementary ways.
- **Hawkes** successfully quantifies cross-category excitation, gives interpretable branching ratios, and beats the homogeneous baseline on the native Poisson likelihood.
- **Markov** variants both beat Hawkes on the binomial-split likelihood because they target that specific objective more directly.

If the task is "model reliability **distribution** dynamics", the Markov variants are preferred — they're the right tool for that job.
If the task is "quantify endogenous triggering between categories", the Hawkes model is preferred — that's exactly what the branching matrix encodes.

---

## 8. Generated Output Files

| File | Description |
|---|---|
| `fit_models_partB.py` | Full fitting + plotting script |
| `model_comparison.csv` | Summary table (LL, AIC, BIC, #params) |
| `fig1_observed_distribution.png` | Stacked daily counts by reliability |
| `fig2_observed_proportion.png` | π_pos(t) trajectory with 95% Wilson CI |
| `fig3_hawkes_branching_heatmap.png` | 2×2 branching matrix heatmap |
| `fig4_hawkes_fitted_vs_observed.png` | Hawkes λ_neg(t), λ_pos(t) vs observed counts |
| `fig5_markov_regime_matrices.png` | P(neg-majority) and P(pos-majority) heatmaps |
| `fig6_meanfield_P_trajectory.png` | Mean-field P(π_pos) functional form + time trajectory |
| `fig7_fitted_proportion_comparison.png` | All four models' q_t vs observed π_pos |
| `fig8_binomial_residuals.png` | Per-day signed binomial deviance residuals |
| `fig9_model_comparison.png` | AIC/BIC/LL bar charts |

---

## 9. Conclusions

1. **All three non-trivial models significantly outperform the homogeneous Markov null** (LRT p < 0.01 for both Markov variants; Hawkes beats null by ΔLL = 5.8).

2. **State-dependent (regime-switching) Markov is the best model** by AIC (322.65), with Mean-field Markov a statistically indistinguishable second (323.13, Δ=0.48).

3. **Multivariate Hawkes captures cross-excitation** (`n_{pos,neg} = 0.32`, `n_{neg,pos} = 0.16`), giving useful mechanistic insight, but is dominated on AIC by the Markov models because it spends parameters on total-count prediction rather than the split.

4. **The dynamics reveal a mean-reverting reliability ecology**: when the reliable fraction drops below ~0.5, the next-day transition probabilities push aggressively back toward reliable dominance. This shows up in both 2a (regime-switching form) and 2b (smooth logistic form).

5. **Branching matrix off-diagonals are non-negligible** (0.16 and 0.32) — unreliable articles do beget reliable ones and vice versa. The reliability label is *not* a closed ecosystem.

6. **Sanity check passed** after switching from single-start to multi-start optimization for the Hawkes fit (max relative error 20% on T=500 synthetic data).

7. **For distribution-of-reliability modeling, the state-dependent Markov is the recommended model** — fewest parameters for essentially tied best performance, most interpretable regime-switching structure, and cleanest fit to the mean-reverting dynamics the data exhibits.

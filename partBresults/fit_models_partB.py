"""
Part B — Modeling the evolution of article reliability distribution.

Data: Las Vegas 2017 shooting CSV, two reliability categories {−1, +1}.
Granularity: daily counts per category (91 days, Oct 2 → Dec 31 2017).

Models
──────
 1. Multivariate Hawkes (AR-1 limit, 6 params):
      λ_k(t) = μ_k + Σ_j n_{k,j}·N_j(t−1)          k, j ∈ {neg, pos}

 2a. State-dependent Markov (regime-switching, 4 params):
      regime x_t = argmax_k π_k(t)   (dominant reliability category on day t)
      P(neg) used on neg-majority days, P(pos) on pos-majority days
      q_t  = (π_t · P(x_t))_{pos}   → binomial likelihood on N_pos(t+1)

 2b. Mean-field Markov (4 params):
      P_{neg→pos}(t) = σ(a0 + a1·π_pos(t))
      P_{pos→neg}(t) = σ(b0 + b1·π_pos(t))
      q_t = (π_t · P_t)_{pos}       → binomial likelihood on N_pos(t+1)

Plus:
 2z. Homogeneous Markov baseline (2 params) — single constant P, null model.

Fair comparison across model families
──────────────────────────────────────
All models are scored on the SAME conditional binomial log-likelihood:
  given total N_tot(t), how well does each model predict N_pos(t)?
For Hawkes, q_t = λ_pos(t) / (λ_pos(t) + λ_neg(t)).

A secondary Poisson-LL comparison is also reported for Hawkes vs
per-category IHP (native count-based likelihood).

Author / context: follows Part A's fit_models.py patterns.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from scipy.special import gammaln, betaln
from scipy.stats import chi2

warnings.filterwarnings("ignore")

# ─── paths ────────────────────────────────────────────────────────────────────
DATA_PATH = (
    "/home/hrwang/StochasticNewsSim/StochasticNewsSim/outputs/"
    "las_vegas_shooting_2017_scored_part_0001_filtered.csv"
)
OUT = "/home/hrwang/StochasticNewsSim/StochasticNewsSim/partBresults/"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH, usecols=["published_date", "reliability_label"])
df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
df = df.dropna(subset=["published_date", "reliability_label"])

# confirm binary
vals = sorted(df["reliability_label"].unique().tolist())
assert vals == [-1.0, 1.0], f"Expected binary reliability, got {vals}"

# split & daily aggregate
df_neg = df[df["reliability_label"] == -1.0]
df_pos = df[df["reliability_label"] ==  1.0]

cnt_neg = df_neg["published_date"].dt.date.value_counts().sort_index()
cnt_pos = df_pos["published_date"].dt.date.value_counts().sort_index()

start = min(cnt_neg.index.min(), cnt_pos.index.min())
end   = max(cnt_neg.index.max(), cnt_pos.index.max())
full_idx = pd.date_range(str(start), str(end), freq="D").date

N_neg = cnt_neg.reindex(full_idx, fill_value=0).values.astype(float)
N_pos = cnt_pos.reindex(full_idx, fill_value=0).values.astype(float)
N_tot = N_neg + N_pos
T     = len(full_idx)
dates = np.array(full_idx, dtype="datetime64[D]")
pd_dates = pd.to_datetime(dates)

# observed proportions (mask days where N_tot==0)
pi_pos = np.where(N_tot > 0, N_pos / np.maximum(N_tot, 1), np.nan)

print(f"Date range  : {full_idx[0]} → {full_idx[-1]}   T = {T} days")
print(f"Totals      : neg = {int(N_neg.sum())}   pos = {int(N_pos.sum())}"
      f"   overall = {int(N_tot.sum())}")
print(f"Zero-total days : {int(np.sum(N_tot == 0))}")


# ─── shared helpers ──────────────────────────────────────────────────────────
def poisson_loglik(rates, counts):
    rates = np.maximum(rates, 1e-10)
    return float(np.sum(counts * np.log(rates) - rates - gammaln(counts + 1)))

def binom_loglik(k, n, p):
    """Binomial log-likelihood, safely handling n=0 and boundary p."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-10, 1.0 - 1e-10)
    mask = n > 0
    if not mask.any():
        return 0.0
    k_, n_, p_ = k[mask], n[mask], p[mask]
    log_coef = gammaln(n_ + 1) - gammaln(k_ + 1) - gammaln(n_ - k_ + 1)
    return float(np.sum(log_coef + k_ * np.log(p_) + (n_ - k_) * np.log(1 - p_)))

def aic(ll, k):     return -2 * ll + 2 * k
def bic(ll, k, n):  return -2 * ll + k * np.log(n)
def sigmoid(x):     return 1.0 / (1.0 + np.exp(-x))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL 1 — MULTIVARIATE HAWKES (AR-1 LIMIT)
# ═══════════════════════════════════════════════════════════════════════════════
# λ_k(t) = μ_k + n_{k,neg}·N_neg(t-1) + n_{k,pos}·N_pos(t-1)
# Params (6): μ_neg, μ_pos, n_{neg,neg}, n_{neg,pos}, n_{pos,neg}, n_{pos,pos}

def hawkes_intensities_mv(params, N_neg, N_pos):
    mu_n, mu_p, n_nn, n_np, n_pn, n_pp = params
    T_ = len(N_neg)
    lam_n = np.empty(T_); lam_p = np.empty(T_)
    lam_n[0] = mu_n
    lam_p[0] = mu_p
    for t in range(1, T_):
        lam_n[t] = mu_n + n_nn * N_neg[t-1] + n_np * N_pos[t-1]
        lam_p[t] = mu_p + n_pn * N_neg[t-1] + n_pp * N_pos[t-1]
    return lam_n, lam_p

def neg_ll_hawkes_mv(params):
    mu_n, mu_p, n_nn, n_np, n_pn, n_pp = params
    if mu_n <= 0 or mu_p <= 0:         return 1e12
    if any(v < 0 for v in [n_nn, n_np, n_pn, n_pp]):   return 1e12
    # stationarity penalty: spectral radius of n-matrix < 1
    nmat = np.array([[n_nn, n_np], [n_pn, n_pp]])
    rho = max(abs(np.linalg.eigvals(nmat)))
    if rho >= 0.9999:  return 1e12
    lam_n, lam_p = hawkes_intensities_mv(params, N_neg, N_pos)
    if np.any(lam_n <= 0) or np.any(lam_p <= 0): return 1e12
    return -(poisson_loglik(lam_n, N_neg) + poisson_loglik(lam_p, N_pos))

best_val, best_p = 1e12, None
for mu_n0 in [2.0, 5.0, 10.0]:
    for mu_p0 in [5.0, 10.0, 20.0]:
        for nval in [0.1, 0.3, 0.5]:
            x0 = [mu_n0, mu_p0, nval, nval, nval, nval]
            r = minimize(neg_ll_hawkes_mv, x0, method="Nelder-Mead",
                         options={"maxiter": 100_000, "xatol": 1e-10, "fatol": 1e-10})
            if r.fun < best_val:
                best_val, best_p = r.fun, r.x

mu_n, mu_p, n_nn, n_np, n_pn, n_pp = best_p
lam_n_hk, lam_p_hk = hawkes_intensities_mv(best_p, N_neg, N_pos)
ll_hk_native = poisson_loglik(lam_n_hk, N_neg) + poisson_loglik(lam_p_hk, N_pos)
n_matrix = np.array([[n_nn, n_np], [n_pn, n_pp]])
rho_hk = max(abs(np.linalg.eigvals(n_matrix)))

print(f"\n=== Multivariate Hawkes (AR-1 limit) ===")
print(f"  μ = (neg={mu_n:.3f}, pos={mu_p:.3f})")
print(f"  Branching matrix n_{{k,j}}  (rows = triggered k, cols = trigger j):")
print(f"                 j=neg    j=pos")
print(f"     k=neg   {n_nn:8.4f} {n_np:8.4f}")
print(f"     k=pos   {n_pn:8.4f} {n_pp:8.4f}")
print(f"  Spectral radius ρ(n) = {rho_hk:.4f}  ({'stationary' if rho_hk < 1 else 'UNSTABLE'})")
print(f"  Poisson LL (native) = {ll_hk_native:.2f}  (6 params)")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PROPORTION-SPACE LIKELIHOOD INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
# All models are compared on the *conditional binomial* likelihood:
#   for each day t with N_tot(t) > 0, score N_pos(t) ~ Binomial(N_tot(t), q_t)
# where q_t is the model-specific predicted pos-proportion on day t.
#
# For Hawkes:    q_t = λ_pos(t) / (λ_pos(t) + λ_neg(t))
# For Markov:    q_t = (π_{t-1} · P_t)_{pos}   (uses previous day's π)
# For day 0 every Markov model predicts the empirical π_0 (no past).
# Days where N_tot(t) = 0 are skipped (binomial contribution is 0).

def q_hawkes(lam_n, lam_p):
    denom = lam_n + lam_p
    return np.where(denom > 0, lam_p / np.maximum(denom, 1e-10), 0.5)

q_hk = q_hawkes(lam_n_hk, lam_p_hk)
ll_hk_binom = binom_loglik(N_pos, N_tot, q_hk)

print(f"  Binomial LL (shared scale) = {ll_hk_binom:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL 2z — HOMOGENEOUS MARKOV BASELINE (2 params)
# ═══════════════════════════════════════════════════════════════════════════════
# Single 2x2 P. Two free probs: P_{neg→pos} and P_{pos→neg}.
# Predict  q_t = (π_{t-1} · P)_{pos}  with P constant.
# Day 0 : predict empirical π_0  (no prior info).

def q_markov(P_by_day, pi_pos_series):
    """Given a (T, 2, 2) stack of transition matrices indexed by the PREVIOUS day's regime
    or continuous state, and the observed previous-day π_pos series, return q_t for t=0..T-1.
    P_by_day[t] is the matrix APPLIED from day t-1 → day t, i.e. q_t is computed as
        q_t = π_{t-1} · P_by_day[t]     for t ≥ 1
        q_0 = π_pos_series[0]           (no prior)
    """
    T_ = len(pi_pos_series)
    q = np.empty(T_)
    q[0] = pi_pos_series[0] if not np.isnan(pi_pos_series[0]) else 0.5
    for t in range(1, T_):
        pi_prev_pos = pi_pos_series[t-1]
        if np.isnan(pi_prev_pos):
            q[t] = q[t-1]
            continue
        pi_prev = np.array([1 - pi_prev_pos, pi_prev_pos])
        P_t = P_by_day[t]
        q[t] = (pi_prev @ P_t)[1]
    return q

# Replace nan pi_pos with forward-filled values for computing q_t
pi_pos_ff = pd.Series(pi_pos).ffill().bfill().values

def neg_ll_markov_homogeneous(params):
    # params: [p_np, p_pn] = P[neg→pos], P[pos→neg]
    p_np, p_pn = params
    if not (0 < p_np < 1 and 0 < p_pn < 1):
        return 1e12
    P = np.array([[1 - p_np, p_np],
                  [p_pn,     1 - p_pn]])
    P_stack = np.broadcast_to(P, (T, 2, 2))
    q_t = q_markov(P_stack, pi_pos_ff)
    return -binom_loglik(N_pos, N_tot, q_t)

res_hom = minimize(neg_ll_markov_homogeneous, [0.3, 0.3],
                   method="Nelder-Mead",
                   options={"maxiter": 50_000, "xatol": 1e-10, "fatol": 1e-10})
p_np_hom, p_pn_hom = res_hom.x
P_hom = np.array([[1 - p_np_hom, p_np_hom],
                  [p_pn_hom,     1 - p_pn_hom]])
q_hom = q_markov(np.broadcast_to(P_hom, (T, 2, 2)), pi_pos_ff)
ll_hom = binom_loglik(N_pos, N_tot, q_hom)

print(f"\n=== Homogeneous Markov (baseline) ===")
print(f"  P_{{neg→pos}} = {p_np_hom:.4f}   P_{{pos→neg}} = {p_pn_hom:.4f}")
print(f"  P matrix:")
print(f"     from\\to    neg       pos")
print(f"     neg       {1-p_np_hom:.4f}    {p_np_hom:.4f}")
print(f"     pos       {p_pn_hom:.4f}    {1-p_pn_hom:.4f}")
print(f"  Binomial LL = {ll_hom:.2f}  (2 params)")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODEL 2a — STATE-DEPENDENT (REGIME-SWITCHING) MARKOV
# ═══════════════════════════════════════════════════════════════════════════════
# Regime on day t-1 = argmax_k π_k(t-1) = neg if π_pos<=0.5 else pos.
# Fit P(neg) and P(pos) — two 2×2 matrices, 4 free params total.

def regime_of(pi_val):
    return 0 if pi_val <= 0.5 else 1   # 0 = neg-majority, 1 = pos-majority

def neg_ll_markov_statedep(params):
    p_np_n, p_pn_n, p_np_p, p_pn_p = params
    if not all(0 < v < 1 for v in params):
        return 1e12
    P_neg = np.array([[1 - p_np_n, p_np_n], [p_pn_n, 1 - p_pn_n]])
    P_pos = np.array([[1 - p_np_p, p_np_p], [p_pn_p, 1 - p_pn_p]])
    P_stack = np.empty((T, 2, 2))
    for t in range(T):
        if t == 0:
            P_stack[t] = P_neg  # unused; q[0] is set to π_0
            continue
        r = regime_of(pi_pos_ff[t-1])
        P_stack[t] = P_neg if r == 0 else P_pos
    q_t = q_markov(P_stack, pi_pos_ff)
    return -binom_loglik(N_pos, N_tot, q_t)

res_sd = minimize(neg_ll_markov_statedep, [0.3, 0.3, 0.3, 0.3],
                  method="Nelder-Mead",
                  options={"maxiter": 100_000, "xatol": 1e-10, "fatol": 1e-10})
p_np_n, p_pn_n, p_np_p, p_pn_p = res_sd.x
P_sd_neg = np.array([[1 - p_np_n, p_np_n], [p_pn_n, 1 - p_pn_n]])
P_sd_pos = np.array([[1 - p_np_p, p_np_p], [p_pn_p, 1 - p_pn_p]])

# Build the q trajectory used for plotting + validation
P_stack_sd = np.empty((T, 2, 2))
for t in range(T):
    if t == 0:
        P_stack_sd[t] = P_sd_neg
        continue
    r = regime_of(pi_pos_ff[t-1])
    P_stack_sd[t] = P_sd_neg if r == 0 else P_sd_pos
q_sd = q_markov(P_stack_sd, pi_pos_ff)
ll_sd = binom_loglik(N_pos, N_tot, q_sd)

print(f"\n=== State-dependent Markov (regime-switching) ===")
print(f"  P(neg-majority regime):")
print(f"     {P_sd_neg[0,0]:.4f}  {P_sd_neg[0,1]:.4f}")
print(f"     {P_sd_neg[1,0]:.4f}  {P_sd_neg[1,1]:.4f}")
print(f"  P(pos-majority regime):")
print(f"     {P_sd_pos[0,0]:.4f}  {P_sd_pos[0,1]:.4f}")
print(f"     {P_sd_pos[1,0]:.4f}  {P_sd_pos[1,1]:.4f}")
print(f"  Binomial LL = {ll_sd:.2f}  (4 params)")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODEL 2b — MEAN-FIELD MARKOV
# ═══════════════════════════════════════════════════════════════════════════════
# P_{neg→pos}(t) = σ(a0 + a1·π_pos(t-1))
# P_{pos→neg}(t) = σ(b0 + b1·π_pos(t-1))

def mean_field_P(params, pi_prev_pos):
    a0, a1, b0, b1 = params
    p_np = sigmoid(a0 + a1 * pi_prev_pos)
    p_pn = sigmoid(b0 + b1 * pi_prev_pos)
    return np.array([[1 - p_np, p_np], [p_pn, 1 - p_pn]])

def neg_ll_markov_meanfield(params):
    P_stack = np.empty((T, 2, 2))
    for t in range(T):
        if t == 0:
            P_stack[t] = np.eye(2)  # unused
            continue
        P_stack[t] = mean_field_P(params, pi_pos_ff[t-1])
    q_t = q_markov(P_stack, pi_pos_ff)
    return -binom_loglik(N_pos, N_tot, q_t)

best_mf_val, best_mf_p = 1e12, None
for a0_init in [-2, -1, 0, 1]:
    for b0_init in [-2, -1, 0, 1]:
        for a1_init in [-2, 0, 2]:
            for b1_init in [-2, 0, 2]:
                x0 = [a0_init, a1_init, b0_init, b1_init]
                r = minimize(neg_ll_markov_meanfield, x0, method="Nelder-Mead",
                             options={"maxiter": 30_000, "xatol": 1e-9, "fatol": 1e-9})
                if r.fun < best_mf_val:
                    best_mf_val, best_mf_p = r.fun, r.x

a0, a1, b0, b1 = best_mf_p

# Build mean-field q trajectory
P_stack_mf = np.empty((T, 2, 2))
for t in range(T):
    if t == 0:
        P_stack_mf[t] = np.eye(2)
        continue
    P_stack_mf[t] = mean_field_P(best_mf_p, pi_pos_ff[t-1])
q_mf = q_markov(P_stack_mf, pi_pos_ff)
ll_mf = binom_loglik(N_pos, N_tot, q_mf)

print(f"\n=== Mean-field Markov ===")
print(f"  a0 = {a0:.4f}   a1 = {a1:.4f}    (neg→pos logistic)")
print(f"  b0 = {b0:.4f}   b1 = {b1:.4f}    (pos→neg logistic)")
print(f"  At π_pos=0 :  P_np = {sigmoid(a0):.4f}   P_pn = {sigmoid(b0):.4f}")
print(f"  At π_pos=1 :  P_np = {sigmoid(a0+a1):.4f}   P_pn = {sigmoid(b0+b1):.4f}")
print(f"  Binomial LL = {ll_mf:.2f}  (4 params)")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MODEL COMPARISON (BINOMIAL SCALE, SHARED ACROSS ALL MODELS)
# ═══════════════════════════════════════════════════════════════════════════════
# Effective sample size: count of days with N_tot > 0
n_eff = int(np.sum(N_tot > 0))

models_summary = [
    ("Multivariate Hawkes",         ll_hk_binom, 6),
    ("Homogeneous Markov",          ll_hom,      2),
    ("State-dependent Markov",      ll_sd,       4),
    ("Mean-field Markov",           ll_mf,       4),
]

print(f"\n=== Shared Binomial Log-Likelihood Comparison (n_eff = {n_eff} days) ===")
print(f"  {'Model':30s}  {'LL':>10s}  {'#p':>3s}  {'AIC':>10s}  {'BIC':>10s}")
for name, ll, k in models_summary:
    print(f"  {name:30s}  {ll:10.2f}  {k:3d}  {aic(ll,k):10.2f}  {bic(ll,k,n_eff):10.2f}")

summary_df = pd.DataFrame({
    "Model":     [m[0] for m in models_summary],
    "LL_binom":  [round(m[1], 2) for m in models_summary],
    "# Params":  [m[2] for m in models_summary],
    "AIC":       [round(aic(m[1], m[2]), 2) for m in models_summary],
    "BIC":       [round(bic(m[1], m[2], n_eff), 2) for m in models_summary],
})
summary_df.to_csv(OUT + "model_comparison.csv", index=False)


# Likelihood-ratio tests
lr_hom_sd = 2 * (ll_sd - ll_hom);  p_hom_sd = 1 - chi2.cdf(lr_hom_sd, 2)
lr_hom_mf = 2 * (ll_mf - ll_hom);  p_hom_mf = 1 - chi2.cdf(lr_hom_mf, 2)

print(f"\n=== Likelihood Ratio Tests (vs Homogeneous Markov baseline) ===")
print(f"  State-dep vs Homog : LR = {lr_hom_sd:.2f}  df=2  p = {p_hom_sd:.4e}")
print(f"  Mean-field vs Homog: LR = {lr_hom_mf:.2f}  df=2  p = {p_hom_mf:.4e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SYNTHETIC SANITY CHECK (Hawkes param recovery)
# ═══════════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(42)
def simulate_hawkes_mv(params, T_sim):
    mu_n_, mu_p_, n_nn_, n_np_, n_pn_, n_pp_ = params
    N_n = np.zeros(T_sim); N_p = np.zeros(T_sim)
    N_n[0] = rng.poisson(mu_n_)
    N_p[0] = rng.poisson(mu_p_)
    for t in range(1, T_sim):
        lam_n_ = mu_n_ + n_nn_ * N_n[t-1] + n_np_ * N_p[t-1]
        lam_p_ = mu_p_ + n_pn_ * N_n[t-1] + n_pp_ * N_p[t-1]
        N_n[t] = rng.poisson(max(lam_n_, 0.01))
        N_p[t] = rng.poisson(max(lam_p_, 0.01))
    return N_n, N_p

true_params = [3.0, 5.0, 0.1, 0.2, 0.15, 0.35]
N_n_sim, N_p_sim = simulate_hawkes_mv(true_params, 500)
def neg_ll_sim(params):
    mu_n_, mu_p_, n_nn_, n_np_, n_pn_, n_pp_ = params
    if mu_n_ <= 0 or mu_p_ <= 0: return 1e12
    if any(v < 0 for v in [n_nn_, n_np_, n_pn_, n_pp_]): return 1e12
    nmat_ = np.array([[n_nn_, n_np_], [n_pn_, n_pp_]])
    if max(abs(np.linalg.eigvals(nmat_))) >= 0.9999: return 1e12
    lam_n_ = np.empty(len(N_n_sim)); lam_p_ = np.empty(len(N_n_sim))
    lam_n_[0] = mu_n_; lam_p_[0] = mu_p_
    for t in range(1, len(N_n_sim)):
        lam_n_[t] = mu_n_ + n_nn_ * N_n_sim[t-1] + n_np_ * N_p_sim[t-1]
        lam_p_[t] = mu_p_ + n_pn_ * N_n_sim[t-1] + n_pp_ * N_p_sim[t-1]
    if np.any(lam_n_ <= 0) or np.any(lam_p_ <= 0): return 1e12
    return -(poisson_loglik(lam_n_, N_n_sim) + poisson_loglik(lam_p_, N_p_sim))

# Multi-start to mirror the real-fit routine
best_sim_val, best_sim_x = 1e12, None
for mu_n0 in [1.0, 3.0, 5.0, 8.0]:
    for mu_p0 in [2.0, 5.0, 8.0]:
        for nval in [0.05, 0.15, 0.3]:
            x0 = [mu_n0, mu_p0, nval, nval, nval, nval]
            r = minimize(neg_ll_sim, x0, method="Nelder-Mead",
                         options={"maxiter": 80_000, "xatol": 1e-10, "fatol": 1e-10})
            if r.fun < best_sim_val:
                best_sim_val, best_sim_x = r.fun, r.x

print(f"\n=== Synthetic sanity check (Hawkes, T=500, multi-start) ===")
print(f"  True params: {['{:.3f}'.format(v) for v in true_params]}")
print(f"  Recovered  : {['{:.3f}'.format(v) for v in best_sim_x]}")
err = max(abs(np.array(best_sim_x) - np.array(true_params)))
rel_err = max(abs(np.array(best_sim_x) - np.array(true_params)) /
              np.maximum(np.abs(true_params), 0.1))
print(f"  Max |error|: {err:.3f}   Max relative error: {rel_err:.3f}  "
      f"({'OK' if rel_err < 0.3 else 'CHECK — consider longer simulation or multi-start widening'})")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
NEG_C = "#c0392b"; POS_C = "#2980b9"
HK_C  = "#27ae60"; HOM_C = "#7f8c8d"; SD_C = "#e67e22"; MF_C = "#8e44ad"
FMT   = mdates.DateFormatter("%b %d")
LOC   = mdates.WeekdayLocator(byweekday=0, interval=2)


# ── Fig 1: observed daily counts, stacked ────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(pd_dates, N_pos, color=POS_C, alpha=0.85, label=f"Reliable (+1)  total={int(N_pos.sum())}", width=0.85)
ax.bar(pd_dates, N_neg, bottom=N_pos, color=NEG_C, alpha=0.85, label=f"Unreliable (−1)  total={int(N_neg.sum())}", width=0.85)
ax.set_xlabel("Date", fontsize=12); ax.set_ylabel("Articles / day", fontsize=12)
ax.set_title("Daily Article Counts by Reliability — Las Vegas 2017", fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(FMT); ax.xaxis.set_major_locator(LOC)
plt.xticks(rotation=30, ha="right")
ax.legend(fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig(OUT + "fig1_observed_distribution.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig1")


# ── Fig 2: observed proportion trajectory ────────────────────────────────────
# With 95% Wilson binomial confidence interval
def wilson_ci(k, n, z=1.96):
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / np.maximum(n, 1)
        denom = 1 + z**2 / np.maximum(n, 1)
        center = (p + z**2/(2*np.maximum(n,1))) / denom
        rad = z * np.sqrt(p*(1-p)/np.maximum(n,1) + z**2/(4*np.maximum(n,1)**2)) / denom
        lo, hi = center - rad, center + rad
    return lo, hi

lo, hi = wilson_ci(N_pos, N_tot)
mask = N_tot > 0

fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(pd_dates[mask], lo[mask], hi[mask], color=POS_C, alpha=0.18, label="95% Wilson CI")
ax.plot(pd_dates[mask], pi_pos[mask], "o-", color=POS_C, lw=1.5, ms=4, label=r"$\pi_{+1}(t)$  observed")
ax.axhline(0.5, color="gray", ls="--", lw=1)
ax.axhline(N_pos.sum()/N_tot.sum(), color="black", ls=":", lw=1,
           label=f"Overall mean = {N_pos.sum()/N_tot.sum():.3f}")
ax.set_ylim(0, 1)
ax.set_xlabel("Date", fontsize=12); ax.set_ylabel(r"$\pi_{+1}(t)$ = reliable fraction", fontsize=12)
ax.set_title("Observed Reliable-Fraction Trajectory", fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(FMT); ax.xaxis.set_major_locator(LOC)
plt.xticks(rotation=30, ha="right")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT + "fig2_observed_proportion.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig2")


# ── Fig 3: Hawkes branching matrix heatmap ──────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(n_matrix, cmap="YlGn", vmin=0, vmax=max(n_matrix.max(), 0.1))
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{n_matrix[i,j]:.4f}", ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if n_matrix[i,j] > 0.5*n_matrix.max() else "black")
ax.set_xticks([0,1]); ax.set_xticklabels(["Triggered by −1", "Triggered by +1"])
ax.set_yticks([0,1]); ax.set_yticklabels(["→ produces −1", "→ produces +1"])
ax.set_title(f"Hawkes Branching Matrix n_{{k,j}}\n"
             f"spectral radius ρ = {rho_hk:.3f}   (< 1: stationary)", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(OUT + "fig3_hawkes_branching_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig3")


# ── Fig 4: Hawkes fitted vs observed per category ────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].bar(pd_dates, N_neg, color=NEG_C, alpha=0.4, label="Observed N_neg(t)", width=0.85)
axes[0].plot(pd_dates, lam_n_hk, color="darkred", lw=2.2, label="Hawkes λ_neg(t)")
axes[0].set_ylabel("Articles / day", fontsize=10)
axes[0].set_title(f"Unreliable (−1) — μ={mu_n:.2f}  n_nn={n_nn:.3f}  n_np={n_np:.3f}",
                  fontsize=11, fontweight="bold")
axes[0].legend(fontsize=9)
axes[1].bar(pd_dates, N_pos, color=POS_C, alpha=0.4, label="Observed N_pos(t)", width=0.85)
axes[1].plot(pd_dates, lam_p_hk, color="navy", lw=2.2, label="Hawkes λ_pos(t)")
axes[1].set_ylabel("Articles / day", fontsize=10)
axes[1].set_title(f"Reliable (+1) — μ={mu_p:.2f}  n_pn={n_pn:.3f}  n_pp={n_pp:.3f}",
                  fontsize=11, fontweight="bold")
axes[1].set_xlabel("Date", fontsize=11)
axes[1].legend(fontsize=9)
axes[1].xaxis.set_major_formatter(FMT); axes[1].xaxis.set_major_locator(LOC)
plt.xticks(rotation=30, ha="right")
fig.suptitle("Multivariate Hawkes: Fitted Intensity vs Observed Counts",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT + "fig4_hawkes_fitted_vs_observed.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig4")


# ── Fig 5: regime transition matrices ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, P_mat, title in [
    (axes[0], P_sd_neg, "Regime: neg-majority day"),
    (axes[1], P_sd_pos, "Regime: pos-majority day"),
]:
    im = ax.imshow(P_mat, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{P_mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if P_mat[i,j] > 0.5 else "black")
    ax.set_xticks([0,1]); ax.set_xticklabels(["→ neg", "→ pos"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["from neg", "from pos"])
    ax.set_title(title, fontsize=11, fontweight="bold")
fig.suptitle("State-dependent Markov: Transition Matrices per Regime",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT + "fig5_markov_regime_matrices.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig5")


# ── Fig 6: mean-field P trajectories ─────────────────────────────────────────
pi_grid = np.linspace(0, 1, 200)
p_np_curve = sigmoid(a0 + a1 * pi_grid)
p_pn_curve = sigmoid(b0 + b1 * pi_grid)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(pi_grid, p_np_curve, color=NEG_C, lw=2.5, label=r"$P_{neg \to pos}(\pi_{+1})$")
axes[0].plot(pi_grid, p_pn_curve, color=POS_C, lw=2.5, label=r"$P_{pos \to neg}(\pi_{+1})$")
axes[0].axhline(p_np_hom, color=NEG_C, ls="--", alpha=0.7, label=f"Homog. P_np = {p_np_hom:.3f}")
axes[0].axhline(p_pn_hom, color=POS_C, ls="--", alpha=0.7, label=f"Homog. P_pn = {p_pn_hom:.3f}")
axes[0].set_xlabel(r"$\pi_{+1}(t-1)$", fontsize=11)
axes[0].set_ylabel("Transition probability", fontsize=11)
axes[0].set_title("Mean-field: transition probs as function of prior distribution",
                  fontsize=11, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, 1)

# Empirical π_pos(t-1) → P_t trajectory
p_np_t = np.array([sigmoid(a0 + a1*pi_pos_ff[t-1]) for t in range(1, T)])
p_pn_t = np.array([sigmoid(b0 + b1*pi_pos_ff[t-1]) for t in range(1, T)])
axes[1].plot(pd_dates[1:], p_np_t, color=NEG_C, lw=2, label=r"$P_{neg \to pos}(t)$")
axes[1].plot(pd_dates[1:], p_pn_t, color=POS_C, lw=2, label=r"$P_{pos \to neg}(t)$")
axes[1].set_xlabel("Date", fontsize=11)
axes[1].set_ylabel("Transition probability", fontsize=11)
axes[1].set_title("Mean-field: P_t trajectory over observed days",
                  fontsize=11, fontweight="bold")
axes[1].xaxis.set_major_formatter(FMT); axes[1].xaxis.set_major_locator(LOC)
axes[1].legend(fontsize=9)
axes[1].set_ylim(0, 1)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUT + "fig6_meanfield_P_trajectory.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig6")


# ── Fig 7: fitted q_t overlay comparison ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.fill_between(pd_dates[mask], lo[mask], hi[mask], color="steelblue", alpha=0.15, label="Observed 95% CI")
ax.plot(pd_dates[mask], pi_pos[mask], "o", color="steelblue", ms=5, alpha=0.7, label=r"Observed $\pi_{+1}(t)$")
ax.plot(pd_dates, q_hk,  color=HK_C,  lw=2,   ls="-",  label=f"Hawkes         (LL={ll_hk_binom:.1f})")
ax.plot(pd_dates, q_hom, color=HOM_C, lw=1.8, ls="--", label=f"Homog. Markov  (LL={ll_hom:.1f})")
ax.plot(pd_dates, q_sd,  color=SD_C,  lw=2,   ls="-.", label=f"State-dep Markov (LL={ll_sd:.1f})")
ax.plot(pd_dates, q_mf,  color=MF_C,  lw=2,   ls=":",  label=f"Mean-field Markov (LL={ll_mf:.1f})")
ax.set_xlabel("Date", fontsize=12); ax.set_ylabel(r"Predicted $q_t$ = P(reliable)", fontsize=12)
ax.set_title("Fitted Reliable-Fraction Trajectories vs Observed", fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(FMT); ax.xaxis.set_major_locator(LOC)
plt.xticks(rotation=30, ha="right")
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(OUT + "fig7_fitted_proportion_comparison.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig7")


# ── Fig 8: binomial deviance residuals per day ───────────────────────────────
def binom_deviance(k, n, p):
    k = np.asarray(k, dtype=float); n = np.asarray(n, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-10, 1 - 1e-10)
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(k > 0, k * np.log(k / (n * p)), 0.0)
        term2 = np.where(k < n, (n - k) * np.log((n - k) / (n * (1 - p))), 0.0)
    d2 = 2 * (term1 + term2)
    d2 = np.where(n > 0, d2, 0.0)
    sign = np.where(k > n * p, 1.0, -1.0)
    return sign * np.sqrt(np.maximum(d2, 0))

fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
for ax, (label, q, clr) in zip(axes, [
    ("Hawkes",               q_hk,  HK_C),
    ("Homogeneous Markov",   q_hom, HOM_C),
    ("State-dependent",      q_sd,  SD_C),
    ("Mean-field",           q_mf,  MF_C),
]):
    dev = binom_deviance(N_pos, N_tot, q)
    ax.bar(pd_dates, dev, color=clr, alpha=0.8, width=0.85)
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_ylabel("Deviance resid.", fontsize=9)
    ax.set_title(f"{label}   Σd² = {np.sum(dev**2):.1f}", fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(FMT); ax.xaxis.set_major_locator(LOC)
axes[-1].set_xlabel("Date", fontsize=11)
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
fig.suptitle("Per-day Binomial Deviance Residuals  (signed)", fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT + "fig8_binomial_residuals.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig8")


# ── Fig 9: AIC / BIC / LL bar charts ─────────────────────────────────────────
mns  = ["Hawkes\n(MV)", "Homog.\nMarkov", "State-dep\nMarkov", "Mean-field\nMarkov"]
lls_ = [ll_hk_binom, ll_hom, ll_sd, ll_mf]
aics_ = [aic(m[1], m[2]) for m in models_summary]
bics_ = [bic(m[1], m[2], n_eff) for m in models_summary]
pal  = [HK_C, HOM_C, SD_C, MF_C]
xs   = np.arange(4)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for vals, ylabel, title, better, ax in [
    (aics_, "AIC",            "AIC  (lower = better)",            "min", axes[0]),
    (bics_, "BIC",            "BIC  (lower = better)",            "min", axes[1]),
    (lls_,  "Log-Likelihood", "Binomial LL  (higher = better)",    "max", axes[2]),
]:
    bars = ax.bar(xs, vals, color=pal, edgecolor="white", width=0.55)
    ax.set_xticks(xs); ax.set_xticklabels(mns, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    rng = abs(max(vals) - min(vals))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + rng*0.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    best_i = np.argmin(vals) if better == "min" else np.argmax(vals)
    bars[best_i].set_edgecolor("black"); bars[best_i].set_linewidth(2.5)
fig.suptitle("Model Comparison — Shared Binomial Likelihood Scale  (bold border = best)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "fig9_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig9")


print(f"\n=== All outputs written to {OUT} ===")

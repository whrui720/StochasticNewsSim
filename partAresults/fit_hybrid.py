"""
Hybrid IHP + Hawkes Process for Las Vegas 2017 daily article counts.
──────────────────────────────────────────────────────────────────────────────
Motivation
──────────
Part A established that:
  • IHP wins over pure Hawkes primarily because it absorbs the exogenous day-0
    burst via A·exp(0) = A, which pure Hawkes cannot do (λ(0) = μ).
  • However, Hawkes captures real self-excitation (n̂ = 0.614 is significant).

A natural unifying model combines both mechanisms:

  Hybrid (full):
      λ(t) = A·exp(−β₀·t) + c + α·R(t)
      R(t) = exp(−β₁)·[R(t−1) + N(t−1)],   R(0) = 0

  Hybrid (AR-1 limit, β₁ → ∞):
      λ(t) = A·exp(−β₀·t) + c + n·N(t−1)

The IHP part absorbs deterministic news-cycle decay from the external event;
the Hawkes part models endogenous article-begets-article dynamics on top of it.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.special import gammaln

warnings.filterwarnings("ignore")

DATA_PATH = (
    "/home/hrwang/StochasticNewsSim/StochasticNewsSim/outputs/"
    "las_vegas_shooting_2017_scored_part_0001_filtered.csv"
)
OUT = "/home/hrwang/StochasticNewsSim/StochasticNewsSim/partAresults/"

# ─── 1. data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, usecols=["published_date"])
df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
df = df.dropna(subset=["published_date"])

date_counts = df["published_date"].dt.date.value_counts().sort_index()
full_idx    = pd.date_range(str(date_counts.index[0]),
                            str(date_counts.index[-1]), freq="D").date
counts_s    = date_counts.reindex(full_idx, fill_value=0)

dates    = np.array(counts_s.index, dtype="datetime64[D]")
N        = counts_s.values.astype(float)
T        = len(N)
t_grid   = np.arange(T, dtype=float)
pd_dates = pd.to_datetime(dates)

print(f"Date range : {full_idx[0]} → {full_idx[-1]}  ({T} days)")
print(f"Total arts : {int(N.sum())}   mean/day : {N.mean():.2f}   max : {int(N.max())}")

# ─── helpers ──────────────────────────────────────────────────────────────────
def poisson_loglik(rates, counts):
    rates = np.maximum(rates, 1e-10)
    return float(np.sum(counts * np.log(rates) - rates - gammaln(counts + 1)))

def aic(ll, k):    return -2 * ll + 2 * k
def bic(ll, k, n): return -2 * ll + k * np.log(n)
def pearson_var(rates, counts):
    return float(np.var((counts - rates) / np.sqrt(np.maximum(rates, 1e-10))))

# ─── 2. baseline reference fits (reproduce from fit_models.py) ─────────────────
# Standard Poisson
lam_sp = N.mean()
rates_sp = np.full(T, lam_sp)
ll_sp = poisson_loglik(rates_sp, N);  k_sp = 1

# IHP
def neg_ll_ihp(p):
    A, b, c = p
    if A <= 0 or b <= 0 or c <= 0: return 1e12
    return -poisson_loglik(A * np.exp(-b * t_grid) + c, N)

res_ihp = minimize(neg_ll_ihp, [N[0], 0.2, max(N[-10:].mean(), 1.0)],
                   method="Nelder-Mead",
                   options={"maxiter": 100_000, "xatol": 1e-10, "fatol": 1e-10})
A_ihp, beta_ihp, c_ihp = res_ihp.x
rates_ihp = A_ihp * np.exp(-beta_ihp * t_grid) + c_ihp
ll_ihp = poisson_loglik(rates_ihp, N);  k_ihp = 3

# Hawkes AR-1
def hk_ar1(mu, n, counts):
    lam = np.empty(len(counts));  lam[0] = mu
    for t in range(1, len(counts)):
        lam[t] = mu + n * counts[t - 1]
    return lam

def neg_ll_hk(p):
    mu, n = p
    if mu <= 0 or n <= 0 or n >= 1: return 1e12
    return -poisson_loglik(hk_ar1(mu, n, N), N)

best_val, best_p = 1e12, None
for mu0 in [5, 11, 20, 30]:
    for n0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
        r = minimize(neg_ll_hk, [mu0, n0], method="Nelder-Mead",
                     options={"maxiter": 50_000, "xatol": 1e-10, "fatol": 1e-10})
        if r.fun < best_val and r.x[0] > 0 and 0 < r.x[1] < 1:
            best_val, best_p = r.fun, r.x
mu_hk, n_hk = best_p
rates_hk = hk_ar1(mu_hk, n_hk, N)
ll_hk = poisson_loglik(rates_hk, N);  k_hk = 2

print(f"\nBaselines:")
print(f"  SP    LL={ll_sp:.2f}  AIC={aic(ll_sp,k_sp):.2f}")
print(f"  IHP   LL={ll_ihp:.2f}  AIC={aic(ll_ihp,k_ihp):.2f}   Â={A_ihp:.2f} β̂={beta_ihp:.4f} ĉ={c_ihp:.3f}")
print(f"  Hawk  LL={ll_hk:.2f}  AIC={aic(ll_hk,k_hk):.2f}    μ̂={mu_hk:.3f}  n̂={n_hk:.3f}")


# ─── 3. Hybrid AR-1: λ(t) = A·exp(−β₀·t) + c + n·N(t-1) ───────────────────────
def hyb_ar1_rates(A, b0, c, n, counts):
    T_ = len(counts)
    lam = np.empty(T_)
    ihp_part = A * np.exp(-b0 * np.arange(T_)) + c
    lam[0] = ihp_part[0]                # no past events at t=0 → no Hawkes
    lam[1:] = ihp_part[1:] + n * counts[:-1]
    return lam

def neg_ll_hyb_ar1(p):
    A, b0, c, n = p
    if A <= 0 or b0 <= 0 or c <= 0 or n <= 0 or n >= 1: return 1e12
    lam = hyb_ar1_rates(A, b0, c, n, N)
    if np.any(lam <= 0): return 1e12
    return -poisson_loglik(lam, N)

best_val, best_p = 1e12, None
for A0 in [N[0], N[0]*0.7, 200.0]:
    for b0_ in [0.15, 0.25, 0.4]:
        for c0 in [2.0, 7.0, 15.0]:
            for n0 in [0.05, 0.15, 0.3]:
                r = minimize(neg_ll_hyb_ar1, [A0, b0_, c0, n0],
                             method="Nelder-Mead",
                             options={"maxiter": 200_000, "xatol": 1e-10, "fatol": 1e-10})
                if r.fun < best_val and all(r.x > 0) and r.x[3] < 1:
                    best_val, best_p = r.fun, r.x
A_h1, b0_h1, c_h1, n_h1 = best_p
rates_h1 = hyb_ar1_rates(A_h1, b0_h1, c_h1, n_h1, N)
ll_h1 = poisson_loglik(rates_h1, N);  k_h1 = 4

print(f"\n=== Hybrid (AR-1 limit) ===")
print(f"  Â={A_h1:.3f}  β̂₀={b0_h1:.4f}  ĉ={c_h1:.4f}  n̂={n_h1:.4f}")
print(f"  Half-life IHP = {np.log(2)/b0_h1:.2f} d    Branching = {n_h1:.3f}")
print(f"  LL={ll_h1:.2f}  AIC={aic(ll_h1,k_h1):.2f}  BIC={bic(ll_h1,k_h1,T):.2f}")
print(f"  Pearson var = {pearson_var(rates_h1, N):.2f}")


# ─── 4. Hybrid full (IHP + exponential-kernel Hawkes, β₁ free) ────────────────
def hyb_full_rates(A, b0, c, alpha, b1, counts):
    T_ = len(counts)
    lam = np.empty(T_)
    decay = np.exp(-b1)
    R = 0.0
    for t in range(T_):
        lam[t] = A * np.exp(-b0 * t) + c + alpha * R
        R = decay * (R + counts[t])
    return lam

def neg_ll_hyb_full(p):
    A, b0, c, alpha, b1 = p
    if A <= 0 or b0 <= 0 or c <= 0 or alpha <= 0 or b1 <= 0: return 1e12
    # branching ratio n = alpha / (exp(b1) − 1) must be < 1
    n = alpha / (np.exp(b1) - 1.0)
    if n >= 1 or n <= 0: return 1e12
    lam = hyb_full_rates(A, b0, c, alpha, b1, N)
    if np.any(lam <= 0): return 1e12
    return -poisson_loglik(lam, N)

best_val, best_p = 1e12, None
for A0 in [N[0]*0.7, 300.0]:
    for b0_ in [0.2, 0.4]:
        for c0 in [5.0, 10.0]:
            for n0 in [0.1, 0.3]:
                for b1_ in [0.5, 1.5, 3.0]:
                    alpha0 = n0 * (np.exp(b1_) - 1.0)
                    r = minimize(neg_ll_hyb_full, [A0, b0_, c0, alpha0, b1_],
                                 method="Nelder-Mead",
                                 options={"maxiter": 300_000, "xatol": 1e-10, "fatol": 1e-10})
                    if r.fun < best_val:
                        A_, b0_x, c_, al_, b1_x = r.x
                        if all(r.x > 0):
                            nv = al_ / (np.exp(b1_x) - 1.0)
                            if 0 < nv < 1:
                                best_val, best_p = r.fun, r.x
A_h2, b0_h2, c_h2, alpha_h2, b1_h2 = best_p
n_h2 = alpha_h2 / (np.exp(b1_h2) - 1.0)
rates_h2 = hyb_full_rates(A_h2, b0_h2, c_h2, alpha_h2, b1_h2, N)
ll_h2 = poisson_loglik(rates_h2, N);  k_h2 = 5

print(f"\n=== Hybrid (full, β₁ free) ===")
print(f"  Â={A_h2:.3f}  β̂₀={b0_h2:.4f}  ĉ={c_h2:.4f}  α̂={alpha_h2:.4f}  β̂₁={b1_h2:.4f}")
print(f"  Branching ratio n = α/(exp(β₁)−1) = {n_h2:.4f}")
print(f"  LL={ll_h2:.2f}  AIC={aic(ll_h2,k_h2):.2f}  BIC={bic(ll_h2,k_h2,T):.2f}")
print(f"  Pearson var = {pearson_var(rates_h2, N):.2f}")


# ─── 5. summary table ────────────────────────────────────────────────────────
models = [
    ("Standard Poisson",       k_sp,  ll_sp,  rates_sp),
    ("Inhomogeneous Poisson",  k_ihp, ll_ihp, rates_ihp),
    ("Hawkes (AR-1)",          k_hk,  ll_hk,  rates_hk),
    ("Hybrid IHP+Hawkes (AR-1)", k_h1, ll_h1, rates_h1),
    ("Hybrid IHP+Hawkes (full)", k_h2, ll_h2, rates_h2),
]
summary = pd.DataFrame({
    "Model":          [m[0] for m in models],
    "# Params":       [m[1] for m in models],
    "Log-Likelihood": [round(m[2], 2) for m in models],
    "AIC":            [round(aic(m[2], m[1]), 2) for m in models],
    "BIC":            [round(bic(m[2], m[1], T), 2) for m in models],
    "Pearson Var":    [round(pearson_var(m[3], N), 2) for m in models],
})
print("\n=== Full Model Comparison (with Hybrid) ===")
print(summary.to_string(index=False))
summary.to_csv(OUT + "model_comparison_with_hybrid.csv", index=False)

# Likelihood-ratio tests (nested)
# Hybrid (AR-1, 4) nests IHP (3, n=0) and Hawkes-AR1 (2, A=0, c→μ)
lr_ihp_h1 = 2 * (ll_h1 - ll_ihp);  p_ihp_h1 = 1 - chi2.cdf(lr_ihp_h1, k_h1 - k_ihp)
lr_hk_h1  = 2 * (ll_h1 - ll_hk);   # non-nested in strict sense (reparam), reported as descriptive
lr_h1_h2  = 2 * (ll_h2 - ll_h1);   p_h1_h2  = 1 - chi2.cdf(lr_h1_h2, k_h2 - k_h1)

print(f"\nLR: IHP vs Hybrid-AR1        LR={lr_ihp_h1:.2f}  df=1   p={p_ihp_h1:.3e}")
print(f"LR: Hybrid-AR1 vs Hybrid-full LR={lr_h1_h2:.2f}  df=1   p={p_h1_h2:.3e}")

# Day-0 decomposition to compare where Hybrid gains over Hawkes / IHP
day0_ihp = N[0]*np.log(max(rates_ihp[0],1e-10)) - rates_ihp[0] - gammaln(N[0]+1)
day0_hk  = N[0]*np.log(max(rates_hk[0], 1e-10)) - rates_hk[0]  - gammaln(N[0]+1)
day0_h1  = N[0]*np.log(max(rates_h1[0], 1e-10)) - rates_h1[0]  - gammaln(N[0]+1)
day0_h2  = N[0]*np.log(max(rates_h2[0], 1e-10)) - rates_h2[0]  - gammaln(N[0]+1)
print(f"\nDay-0 log-lik contributions:")
print(f"  IHP       = {day0_ihp:.2f}  (λ(0)={rates_ihp[0]:.1f})")
print(f"  Hawkes    = {day0_hk:.2f}   (λ(0)={rates_hk[0]:.1f})")
print(f"  Hybrid-AR1= {day0_h1:.2f}   (λ(0)={rates_h1[0]:.1f})")
print(f"  Hybrid-full={day0_h2:.2f}   (λ(0)={rates_h2[0]:.1f})")


# ═══ FIGURES ══════════════════════════════════════════════════════════════════
SP_C  = "#e74c3c";  IHP_C = "#f39c12";  HK_C = "#27ae60"
HYB_C = "#8e44ad";  HYB2_C = "#2980b9"
FMT   = mdates.DateFormatter("%b %d")
LOC   = mdates.WeekdayLocator(byweekday=0, interval=2)


# ── Fig 10: Hybrid panels (AR-1 and full) vs observed ────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for ax, (lbl, rates, clr) in zip(axes, [
    (f"Hybrid IHP+Hawkes (AR-1)  Â={A_h1:.1f}, β̂₀={b0_h1:.3f}, ĉ={c_h1:.2f}, n̂={n_h1:.3f}",
     rates_h1, HYB_C),
    (f"Hybrid IHP+Hawkes (full)  Â={A_h2:.1f}, β̂₀={b0_h2:.3f}, ĉ={c_h2:.2f}, α̂={alpha_h2:.3f}, β̂₁={b1_h2:.3f}  (n={n_h2:.3f})",
     rates_h2, HYB2_C),
]):
    ax.bar(pd_dates, N, color="steelblue", alpha=0.35, label="Observed", width=0.85)
    ax.plot(pd_dates, rates, color=clr, lw=2.5, label="Fitted λ(t)")
    ax.set_ylabel("Articles / day", fontsize=10)
    ax.set_title(lbl, fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(FMT);  ax.xaxis.set_major_locator(LOC)
axes[-1].set_xlabel("Date", fontsize=11)
plt.xticks(rotation=30, ha="right")
fig.suptitle("Hybrid IHP + Hawkes Models — Las Vegas 2017",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT + "fig10_hybrid_fits.png", dpi=150, bbox_inches="tight")
plt.close();  print("\nSaved fig10")


# ── Fig 11: overlay — all 5 models ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.bar(pd_dates, N, color="steelblue", alpha=0.28, label="Observed", width=0.85)
ax.plot(pd_dates, rates_sp,  color=SP_C,   lw=1.5, ls=":",  label=f"Std Poisson")
ax.plot(pd_dates, rates_ihp, color=IHP_C,  lw=1.8, ls="--", label=f"IHP")
ax.plot(pd_dates, rates_hk,  color=HK_C,   lw=1.8, ls="-.", label=f"Hawkes (AR-1)")
ax.plot(pd_dates, rates_h1,  color=HYB_C,  lw=2.5, ls="-",  label=f"Hybrid (AR-1)")
ax.plot(pd_dates, rates_h2,  color=HYB2_C, lw=2.5, ls="-",  label=f"Hybrid (full)")
ax.set_xlabel("Date", fontsize=12);  ax.set_ylabel("Articles / day", fontsize=12)
ax.set_title("All Five Models — Las Vegas Shooting 2017", fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(FMT);  ax.xaxis.set_major_locator(LOC)
plt.xticks(rotation=30, ha="right");  ax.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig(OUT + "fig11_all_models_overlay.png", dpi=150, bbox_inches="tight")
plt.close();  print("Saved fig11")


# ── Fig 12: AIC / BIC / LL bar charts for all 5 models ───────────────────────
mn = ["Std\nPoisson", "IHP", "Hawkes\n(AR-1)", "Hybrid\n(AR-1)", "Hybrid\n(full)"]
aics = [aic(m[2], m[1]) for m in models]
bics = [bic(m[2], m[1], T) for m in models]
lls  = [m[2] for m in models]
pal  = [SP_C, IHP_C, HK_C, HYB_C, HYB2_C]
x = np.arange(5)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for vals, ylabel, title, better, ax in [
    (aics, "AIC",            "AIC  (lower = better)",            "min", axes[0]),
    (bics, "BIC",            "BIC  (lower = better)",            "min", axes[1]),
    (lls,  "Log-Likelihood", "Log-Likelihood  (higher = better)", "max", axes[2]),
]:
    bars = ax.bar(x, vals, color=pal, edgecolor="white", width=0.6)
    ax.set_xticks(x);  ax.set_xticklabels(mn, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    rng = abs(max(vals) - min(vals))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (rng*0.01 if bar.get_height() >= 0 else -rng*0.03),
                f"{v:.0f}", ha="center",
                va="bottom" if bar.get_height() >= 0 else "top", fontsize=8)
    best_i = np.argmin(vals) if better=="min" else np.argmax(vals)
    bars[best_i].set_edgecolor("black");  bars[best_i].set_linewidth(2.5)
fig.suptitle("Model Selection Criteria — 5 Models  (bold border = best)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "fig12_all_models_criteria.png", dpi=150, bbox_inches="tight")
plt.close();  print("Saved fig12")


# ── Fig 13: Hybrid decomposition (AR-1) — IHP vs self-excitation components ──
ihp_comp = A_h1 * np.exp(-b0_h1 * t_grid) + c_h1
exc_comp = np.concatenate([[0.0], n_h1 * N[:-1]])

fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
axes[0].bar(pd_dates, N, color="steelblue", alpha=0.35, width=0.85, label="Observed")
axes[0].plot(pd_dates, rates_h1, color=HYB_C, lw=2.5, label="Hybrid λ(t)")
axes[0].set_ylabel("Articles / day", fontsize=11);  axes[0].legend(fontsize=9)
axes[0].set_title("Hybrid (AR-1) Fitted Intensity vs Observed",
                  fontsize=11, fontweight="bold")
axes[1].stackplot(pd_dates, ihp_comp, exc_comp,
                  labels=[f"IHP part  A·exp(−β₀t)+c   (A={A_h1:.1f}, β₀={b0_h1:.3f}, c={c_h1:.2f})",
                          f"Self-excitation  n·N(t−1)  (n={n_h1:.3f})"],
                  colors=[IHP_C, HK_C], alpha=0.75)
axes[1].set_ylabel("Intensity component", fontsize=11)
axes[1].set_xlabel("Date", fontsize=11)
axes[1].set_title("Hybrid Decomposition: Exogenous Decay + Endogenous Self-Excitation",
                  fontsize=11, fontweight="bold")
axes[1].legend(fontsize=9, loc="upper right")
axes[1].xaxis.set_major_formatter(FMT);  axes[1].xaxis.set_major_locator(LOC)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUT + "fig13_hybrid_decomposition.png", dpi=150, bbox_inches="tight")
plt.close();  print("Saved fig13")


# ── Fig 14: Early-zoom Oct 2–15, focus on hybrid vs IHP vs Hawkes ────────────
zoom = pd_dates <= pd.Timestamp("2017-10-15")
fig, ax = plt.subplots(figsize=(11, 5))
zd = pd_dates[zoom]
ax.bar(zd, N[zoom], color="steelblue", alpha=0.35, width=0.85, label="Observed")
ax.plot(zd, rates_ihp[zoom], color=IHP_C,  lw=2,   ls="--", label="IHP")
ax.plot(zd, rates_hk[zoom],  color=HK_C,   lw=2,   ls="-.", label="Hawkes (AR-1)")
ax.plot(zd, rates_h1[zoom],  color=HYB_C,  lw=2.5, ls="-",  label="Hybrid (AR-1)")
ax.plot(zd, rates_h2[zoom],  color=HYB2_C, lw=2.5, ls="-",  label="Hybrid (full)")
ax.set_xlabel("Date", fontsize=12);  ax.set_ylabel("Articles / day", fontsize=12)
ax.set_title("Early Period Zoom (Oct 2–15): Hybrid Absorbs Both Burst and Follow-up",
             fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=30, ha="right");  ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT + "fig14_hybrid_early_zoom.png", dpi=150, bbox_inches="tight")
plt.close();  print("Saved fig14")


# ── Fig 15: Residuals — Hybrid vs IHP vs Hawkes ──────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, (lbl, rates, clr) in zip(axes, [
    ("IHP",                     rates_ihp, IHP_C),
    ("Hawkes (AR-1)",           rates_hk,  HK_C),
    ("Hybrid IHP+Hawkes (AR-1)", rates_h1, HYB_C),
]):
    resid = N - rates
    rss = np.sum(resid**2)
    ax.bar(pd_dates, resid, color=clr, alpha=0.70, width=0.85)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Residual", fontsize=10)
    ax.set_title(f"{lbl}  (RSS = {rss:.0f})", fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(FMT);  ax.xaxis.set_major_locator(LOC)
axes[-1].set_xlabel("Date", fontsize=11)
plt.xticks(rotation=30, ha="right")
fig.suptitle("Residuals Comparison: Hybrid vs IHP vs Hawkes",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "fig15_hybrid_residuals.png", dpi=150, bbox_inches="tight")
plt.close();  print("Saved fig15")

# ── save fitted rates / parameters for downstream use ────────────────────────
pd.DataFrame({
    "date": pd_dates,
    "N": N.astype(int),
    "rates_sp": rates_sp,
    "rates_ihp": rates_ihp,
    "rates_hk": rates_hk,
    "rates_hybrid_ar1": rates_h1,
    "rates_hybrid_full": rates_h2,
}).to_csv(OUT + "fitted_rates_all_models.csv", index=False)

params = {
    "sp":         {"lambda": float(lam_sp)},
    "ihp":        {"A": float(A_ihp), "beta": float(beta_ihp), "c": float(c_ihp)},
    "hawkes_ar1": {"mu": float(mu_hk), "n": float(n_hk)},
    "hybrid_ar1": {"A": float(A_h1), "beta0": float(b0_h1),
                   "c": float(c_h1), "n": float(n_h1)},
    "hybrid_full":{"A": float(A_h2), "beta0": float(b0_h2), "c": float(c_h2),
                   "alpha": float(alpha_h2), "beta1": float(b1_h2),
                   "n": float(n_h2)},
}
import json
with open(OUT + "fitted_params_all_models.json", "w") as f:
    json.dump(params, f, indent=2)
print("Saved fitted_rates_all_models.csv and fitted_params_all_models.json")


print("\n" + "="*65)
print("HYBRID ANALYSIS — FINAL COMPARISON")
print("="*65)
print(summary.to_string(index=False))
print(f"""
Interpretation
──────────────
• Hybrid-AR1 ({k_h1} params): λ(t) = A·exp(−β₀·t) + c + n·N(t−1)
  A={A_h1:.2f}  β₀={b0_h1:.4f} (half-life {np.log(2)/b0_h1:.2f}d)
  c={c_h1:.2f}  n={n_h1:.4f} (branching ratio)
  ΔAIC vs IHP:    {aic(ll_h1,k_h1)-aic(ll_ihp,k_ihp):+.2f}
  ΔAIC vs Hawkes: {aic(ll_h1,k_h1)-aic(ll_hk,k_hk):+.2f}

• Hybrid-full ({k_h2} params): λ(t) = A·exp(−β₀·t) + c + α·R(t)
  β₁={b1_h2:.3f}  branching n={n_h2:.3f}
  ΔAIC vs Hybrid-AR1: {aic(ll_h2,k_h2)-aic(ll_h1,k_h1):+.2f}

Day-0 log-lik: IHP={day0_ihp:.1f}, Hawkes={day0_hk:.1f},
               Hybrid-AR1={day0_h1:.1f}, Hybrid-full={day0_h2:.1f}
""")

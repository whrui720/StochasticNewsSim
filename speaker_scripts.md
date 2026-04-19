# Speaker Script Outlines

Stochastics of News Article Propagation — final-presentation scripts.
Each section covers material **not explicitly written on the slide** so that
the slide stays clean while the speaker fills in the depth.

Random presenter assignment (seeded so it is fixed):

| Slide | Title | Presenter |
|---|---|---|
| 1 | Title | Harry Wang (lead-in for all three) |
| 2 | Dataset Overview | Sriyan Madugula |
| 3 | Part A — Base Models | Harry Wang |
| 4 | Part A — Hybrid Winner | Ruthesh Thavamani |
| 5 | Part B — Setup & Shared Binomial Likelihood | Sriyan Madugula |
| 6 | Part B — Model Specifications | Sriyan Madugula |
| 7 | Part B — Multivariate Hawkes | Harry Wang |
| 8 | Part B — Markov Variants | Ruthesh Thavamani | Sriyan
| 9 | Part B — Comparison | Sriyan Madugula |
| 10 | Part B — Interpretation | Harry Wang | Sriyan | Ruthesh
| 11 | Conclusions and Next Steps | Ruthesh Thavamani |

---

## Slide 1 — Title  *(Harry Wang)*

- Welcome the audience and introduce the three presenters by name.
- One-sentence framing: "We modelled how news articles about a single major
  event propagate through time, and how the *reliability* composition of that
  coverage evolves day-by-day."
- Note that the project decomposes into two parts: **Part A** is a univariate
  counting-process problem, **Part B** lifts the same machinery to a
  *multivariate* / *distributional* setting.
- Tee up the running example: the 2017 Las Vegas shooting, chosen because
  the day-0 spike is sharp and the tail is long enough to discriminate models.

---

## Slide 2 — Dataset Overview  *(Sriyan Madugula)*

- Where the data comes from: CC-News (Common Crawl's news subset) gives URL +
  publish date + body; NewsGuard provides domain-level reliability scores.
- How they were combined: domain-join on the article URL → each article
  inherits the score of its publishing outlet. The continuous score is
  thresholded at the median to give a binary `±1` reliability label.
- Why we filtered: the unfiltered CC-News crawl is enormous; we keyword-matched
  to the Las Vegas event and zero-filled missing dates so every model sees the
  same 91-day grid.
- Call out that the figure on the right is the *raw* stacked daily counts
  before any modelling — just to anchor what the input looks like.
- Spend a moment on the limitations bullets:
  - Binary label loses the magnitude of NewsGuard scores (a 92 vs a 55 are
    both just "reliable").
  - Day-level granularity hides intraday bursts (a press-conference at 3 PM
    will be invisible).
  - Single event window means we cannot generalise across event *types*
    (mass-casualty event vs slow-burning policy story would be quite
    different).
- Hand off: "With that in mind, here is what we found in Part A …"

---

## Slide 3 — Part A: Modelling the Counting Process  *(Harry Wang)*

- Re-state the Part-A goal in one sentence: "Given the daily article count
  $N(t)$, fit a stochastic intensity $\lambda(t)$ that explains the time
  series."
- Walk through the **fitting methodology** (not on the slide):
  - All three are scored on the *same* conditional Poisson log-likelihood,
    including the `log N!` term, so AIC/BIC are directly comparable.
  - Hawkes log-likelihood is computed *recursively* (R(t) depends on past
    counts), which is the only correct treatment for self-exciting processes
    on binned data.
  - Optimisation: Nelder–Mead with multi-start to dodge local minima.
- Explain the **Hawkes β-profile** chart on the right: we fixed β on a log-grid
  and re-optimised (μ, n); the profile log-likelihood is monotone in β and
  plateaus at β ≥ 10. The interpretation matters: the optimal exponential
  kernel collapses to an AR(1) one-step lag, so the "effective" Hawkes is
  λ(t) = 11.50 + 0.614 · N(t−1).
- Highlight the **Pearson dispersion** column in the table: it measures how
  smooth the fitted rate is relative to the noise. Standard Poisson and Hawkes
  both have ~200; IHP is at 8.5.
- Set up the punchline: IHP wins by a huge margin because the Hawkes process
  has no mechanism to absorb the day-0 exogenous shock — that single day
  accounts for ~91 % of the Hawkes-vs-IHP gap. This motivates the next slide.

---

## Slide 4 — Part A: Hybrid IHP + Hawkes Wins  *(Ruthesh Thavamani)*

- Frame: "If pure IHP captures the exogenous decay and pure Hawkes captures
  endogenous self-excitation, the natural fix is to *add* them."
- Walk through what the **hybrid decomposition figure** shows: the IHP term
  collapses almost to a delta at day 0 (β₀ = 2.22 → half-life 0.31 days), and
  from day 1 onwards the Hawkes term carries the dynamics with branching ratio
  n = 0.725.
- Note the **two distinct timescales** in the full hybrid: IHP half-life ≈ 1.75
  days for the fast initial decay, Hawkes kernel half-life ≈ 8.2 days for the
  slow self-sustaining tail. This is structurally different from the AR-1
  variant.
- Day-0 sanity check (not on the slide): Hybrid-AR1 lands λ(0) = 489.0
  *exactly* on the observed N(0) = 489 — the optimiser drove A + c right onto
  the data.
- Call out the **known limitation** that we discovered but didn't fold into
  the main model: a clear weekly seasonality. Wednesday/Thursday systematically
  over-shoot, weekends under-shoot. Adding six day-of-week multipliers buys
  another ΔAIC ≈ −148, which we kept as a diagnostic rather than a main result.
- Tee up: "Now we move to *which kind* of articles are being published,
  not how many."

---

## Slide 5 — Part B: Setup and the Shared Binomial Likelihood  *(Sriyan Madugula)*

- Restate the shift: in Part A we modelled the count $N(t)$; in Part B we
  model the *reliability split* $q_t = P(\text{article reliable} \mid t)$,
  tracked empirically by $\pi_{pos}(t) = N_{pos}(t) / N_{tot}(t)$.
- Why we need a shared likelihood in the first place:
  - Hawkes natively scores the *pair of counts* $(N_{neg}(t), N_{pos}(t))$
    under a multivariate Poisson log-likelihood.
  - Markov natively scores a one-step *transition* under a multinomial /
    binomial on the proportion.
  - The two native AICs live on completely different scales — directly
    comparing them would be meaningless.
- Derivation of the shared scale — the **Poisson–binomial superposition
  identity** (right panel, box 1):
  - If $X \sim \mathrm{Pois}(\lambda_1)$ and $Y \sim \mathrm{Pois}(\lambda_2)$
    independently, then $X \mid (X + Y = n) \sim
    \mathrm{Binomial}(n, \lambda_1 / (\lambda_1 + \lambda_2))$.
  - Applied to our setup, $N_{pos}(t) \mid N_{tot}(t) \sim
    \mathrm{Binomial}(N_{tot}(t), q_t)$.
  - This is *the* natural noise model for "$k$ positives out of $N$" when
    outcomes are conditionally independent.
- Walk through the **per-day log-likelihood** (box 2):
  - $\ell_t = \log \binom{N_{tot}(t)}{N_{pos}(t)} + N_{pos}(t) \log q_t +
    N_{neg}(t) \log(1 - q_t)$.
  - The binomial coefficient is a constant in $q_t$, so it cancels in every
    model comparison — it only affects the absolute LL, not ΔAIC or LRT
    statistics.
  - Sum over the 86 non-empty days (five days in the 91-day window have zero
    articles and are skipped).
- Point at the **model-specific $q_t$** box (box 3):
  - Hawkes implied split: $q_t = \lambda_{pos}(t) /
    (\lambda_{pos}(t) + \lambda_{neg}(t))$ — a direct consequence of the
    Poisson superposition identity.
  - Markov transition: $q_t = (\pi_{t-1} P_t)_{pos}$ — just a matrix-vector
    product on the empirical distribution.
- Hand-off: "Given that common scoring rule, the next slide gives the full
  functional form of each candidate model."

---

## Slide 6 — Part B: Model Specifications  *(Sriyan Madugula)*

- Framing: one slide, four boxes, each box is a self-contained model
  specification — formula, parameters, one-line derivation, and how it
  plugs into the shared binomial $q_t$.

### Box 1 — Multivariate Hawkes (AR-1 limit, 6 parameters)
- Formula: $\lambda_k(t) = \mu_k + \sum_{j \in \{neg, pos\}} n_{k,j}
  N_j(t-1)$ for $k \in \{neg, pos\}$.
- Derivation: direct two-dimensional lift of Part A's AR(1) Hawkes winner.
  Each category gets its own baseline $\mu_k$ and its own row of the $2
  \times 2$ branching matrix, so cross-excitation between categories is
  first-class.
- Stationarity: require the spectral radius $\rho(n) < 1$. Enforced as a
  constraint during optimisation (log-barrier) and checked at the fitted
  value.
- Poisson counts $\Rightarrow$ superposition gives
  $q_t = \lambda_{pos}(t) / (\lambda_{pos}(t) + \lambda_{neg}(t))$, which
  is what we score on.

### Box 2 — Homogeneous Markov (null, 2 parameters)
- Formula: a single constant $2 \times 2$ transition matrix $P$; rows sum
  to one, so only two free parameters (one per row).
- Derivation: treat an article's reliability label as a first-order Markov
  chain over a fictitious "yesterday's sampled article" and aggregate to
  daily proportions.
- Score function: $q_t = (\pi_{t-1} P)_{pos}$ directly.
- Purpose: null benchmark against which 2a and 2b are likelihood-ratio
  tested.
- Stationary distribution $\pi^*$ satisfies $\pi^* P = \pi^*$ and sits at
  $\approx (0.39, 0.61)$, close to the empirical (0.37, 0.63).

### Box 3 — State-dependent (regime-switching) Markov (4 parameters)
- Regime indicator: $x_{t-1} = \arg\max_k \pi_k(t-1) \in \{neg, pos\}$ —
  just "which class was in the majority yesterday?".
- Formula: two separate transition matrices $P^{(neg)}$ and $P^{(pos)}$,
  and $q_t = (\pi_{t-1} P^{(x_{t-1})})_{pos}$.
- Derivation idea: the shock regime (neg-majority, only 12 of 86 non-empty
  days) is dynamically different from the equilibrium regime; let the data
  fit two matrices instead of one.
- Caveat: the cut-off at $\pi_{pos} = 0.5$ is model-imposed and abrupt;
  that motivates 2b.

### Box 4 — Mean-field Markov (4 parameters)
- Formulas: logistic links
  - $P_{neg \to pos}(t) = \sigma(a_0 + a_1 \pi_{pos}(t-1))$,
  - $P_{pos \to neg}(t) = \sigma(b_0 + b_1 \pi_{pos}(t-1))$;
  other entries fixed by the row-sum constraint.
- Derivation idea: replace the discrete regime indicator with a smooth
  function of yesterday's empirical proportion — same mean-field intuition
  as 2a but differentiable everywhere.
- Nesting: at $a_1 = b_1 = 0$ it reduces exactly to the homogeneous null
  (box 2), giving a clean LRT with df = 2.
- This model is the discrete-time precursor of the McKean–Vlasov direction
  flagged on the conclusions slide.

### Closing beat
- "All four models slot into the same $\ell_t = N_{pos} \log q_t +
  N_{neg} \log(1 - q_t) + \text{const}$ likelihood — only the functional
  form of $q_t$ differs. That's the whole reason the comparison is fair."

---

## Slide 7 — Part B: Multivariate Hawkes — Cross-Excitation  *(Harry Wang)*

- Justification (not on slide): the multivariate Hawkes is the *natural*
  generalisation of Part A's Hawkes — same self-exciting kernel, but with a
  matrix-valued branching ratio that lets the two reliability streams
  cross-trigger.
- Walk through the **branching matrix heatmap** on the right: rows are the
  triggered category, columns the triggering category. The diagonal entries
  (within-category) dominate but are not overwhelming.
- Spend a beat on the **off-diagonal asymmetry**:
  - $n_{pos,neg} = 0.32$: an unreliable article today triggers about 0.32
    reliable articles tomorrow on average — likely fact-checks, debunks, and
    follow-ups by reliable outlets.
  - $n_{neg,pos} = 0.16$: half as much in the reverse direction — reliable
    articles trigger fewer unreliable follow-ups.
- Reference the **fitted-vs-observed** chart: visually the model tracks both
  streams, with the usual under-prediction of weekday spikes (same weekly
  seasonality issue as Part A).
- Spectral radius 0.633 means the process is comfortably sub-critical (no
  blow-up), consistent with the stable news cycle observed in Part A.
- Foreshadow: "But the binomial-likelihood scoring is going to penalise
  Hawkes for spending six parameters on the *total*, when only the split
  matters."

---

## Slide 8 — Part B: Markov Variants — Regime Switching & Mean-Field  *(Ruthesh Thavamani)*

- Justification: Markov models *directly* parameterise the conditional
  distribution we are scoring on. They are the most parameter-efficient choice
  if all you care about is the next-day split.
- Walk through the **state-dependent matrices** on the right (top heatmap pair):
  - In the *pos-majority* regime, both diagonal entries are ~0.7 — the system
    is sticky.
  - In the *neg-majority* regime (rare; happens during the initial shock),
    most articles flip categories — $P_{neg \to pos} = 0.73$.
- Move to the **mean-field smooth analogue** (bottom right): the curve shows
  $P_{neg \to pos}$ as a logistic function of yesterday's $\pi_{pos}$. The
  steeply decreasing shape *is* the mean-reversion signature.
- Compare the two parameterisations (not on slide):
  - State-dependent is easier to interpret (two named regimes) but the
    threshold at 0.5 is arbitrary.
  - Mean-field is smooth, identifiable everywhere, and reduces to the null
    when slopes vanish — that's what makes the LRT clean (df = 2).
- Both have only **4 parameters** vs Hawkes' 6 — important context for the
  AIC comparison on the next slide.

---

## Slide 9 — Part B: Model Comparison  *(Sriyan Madugula)*

- Walk through the table left-to-right: log-likelihood, AIC, BIC.
  - State-dependent Markov wins outright on AIC (322.65).
  - Mean-field Markov is statistically indistinguishable (Δ = 0.48).
  - Hawkes is *better* than the homogeneous baseline on raw log-likelihood
    but loses after the AIC penalty for its 4 extra parameters.
- Explain the **LRT block**: both 4-parameter Markov variants are significant
  vs the homogeneous baseline at p < 0.01. We can't LR-test Hawkes vs Markov
  because the families don't nest, so we fall back on AIC.
- "Why Markov beats Hawkes here" — expand on the bullets:
  - Hawkes spends parameters predicting the *total count*. The fact that it
    matches Markov on log-likelihood despite this overhead is actually
    impressive, but the AIC bookkeeping doesn't reward it.
  - 86 % of days are pos-majority, so a single transition matrix already
    captures most of the dynamics. The state-dependent tweak adds value
    *exactly* where it matters: the initial neg-majority window.
- Reference the **fitted-proportion comparison** on the top-right: visually
  all four models track the empirical $\pi_{pos}$ similarly after day 5; the
  difference is concentrated in the first week.
- Reference the **AIC/BIC bar chart** on the bottom-right as the visual
  summary of the table.

---

## Slide 10 — Part B: Interpretation  *(Harry Wang)*

- Centre the discussion on the phrase **mean-reverting reliability ecology**.
  The data exhibits a stable equilibrium near $\pi_{pos} \approx 0.63$, and
  the Markov variants make explicit the mechanism that pulls the system back
  toward that equilibrium when it strays.
- Tie the two model families together: the state-dependent Markov and the
  mean-field Markov tell the *same* story, just in discrete vs smooth form.
  The Hawkes branching matrix tells a *complementary* story: it explains
  *what kind of article triggers what kind of article*, which is something
  the Markov models cannot articulate.
- Spend time on the "model choice depends on the question" point:
  - If the audience is a content moderator who just wants to forecast
    tomorrow's reliability mix, hand them the state-dependent Markov.
  - If the audience is a media-studies researcher who wants to claim that
    "unreliable outlets seed reliable follow-ups", they want the Hawkes
    branching matrix.
- Mention the **synthetic recovery sanity check** (max relative error 20 % on
  T = 500): we ran a self-test before trusting any of the Hawkes parameter
  estimates on the real data. The single-start optimiser missed badly; the
  multi-start version is what we report.
- Briefly acknowledge what we *don't* explain: the Markov framework discards
  count-magnitude information entirely, and the Hawkes framework discards the
  AR(1)-vs-AR(7) periodicity that the Part A diagnostics revealed.

---

## Slide 11 — Conclusions and Next Steps  *(Ruthesh Thavamani)*

- Summarise the **two headline results** in one breath:
  1. Counts → hybrid IHP + Hawkes is the right model: exogenous shock plus
     endogenous amplification at distinct timescales.
  2. Reliability split → state-dependent Markov is the most efficient model;
     the dynamics are mean-reverting around the empirical equilibrium.
- Walk through the **next-steps** column on the right:
  - **McKean–Vlasov**: replace the discrete-time transition matrix $P_t$
    with a continuous-time SDE whose drift depends on the empirical measure
    of $\pi_{pos}$. This generalises mean-field Markov to a true
    distribution-dependent process and lets us compute things like first-
    passage times analytically.
  - **Continuous reliability scores**: drop the median-threshold binarisation
    and model the raw NewsGuard score as a real-valued mark on each event.
    Enables regression-style or functional models on a continuous reliability
    axis and recovers within-class heterogeneity that the binary label hides.
  - **Cross-event replication**: fit the same machinery on the other event
    files we already prepared (Nice 2016, Syrian civil war, Zika outbreak)
    to test whether the hybrid + Markov conclusions generalise across event
    types.
  - **Weekly seasonality**: fold the day-of-week multipliers from the Part-A
    diagnostic into a joint model (ΔAIC ≈ −148 in the exploratory fit).
- Close with one sentence inviting questions, and thank the audience.

"""Build the project presentation .pptx (academic, professional)."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from PIL import Image

ROOT = Path(__file__).parent
PARTA = ROOT / "partAresults"
PARTB = ROOT / "partBresults"
OUT = ROOT / "StochasticNewsSim_presentation.pptx"

NAVY = RGBColor(0x0B, 0x2A, 0x4A)
SLATE = RGBColor(0x33, 0x33, 0x33)
ACCENT = RGBColor(0x8B, 0x1A, 0x1A)
LIGHT = RGBColor(0xF5, 0xF5, 0xF5)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height

BLANK = prs.slide_layouts[6]


def add_rect_band(slide, top, height, color):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, top, SW, height)
    s.line.fill.background()
    s.fill.solid()
    s.fill.fore_color.rgb = color
    return s


def add_text(slide, left, top, width, height, text, *, size=18, bold=False,
             color=SLATE, align=PP_ALIGN.LEFT, font="Calibri"):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    lines = text.split("\n") if isinstance(text, str) else text
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = font
        run.font.color.rgb = color
    return tb


def add_bullets(slide, left, top, width, height, bullets, *, size=16,
                color=SLATE, font="Calibri", line_spacing=1.15):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        if isinstance(b, tuple):
            text, level = b
        else:
            text, level = b, 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        p.line_spacing = line_spacing
        p.space_after = Pt(2)
        run = p.add_run()
        run.text = ("\u2022  " if level == 0 else "\u2013  ") + text
        run.font.size = Pt(size - level * 2)
        run.font.name = font
        run.font.color.rgb = color
    return tb


def slide_header(slide, title, subtitle=None):
    add_rect_band(slide, 0, Inches(0.9), NAVY)
    add_text(slide, Inches(0.4), Inches(0.18), Inches(12.5), Inches(0.55),
             title, size=26, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    if subtitle:
        add_text(slide, Inches(0.4), Inches(0.55), Inches(12.5), Inches(0.35),
                 subtitle, size=13, color=RGBColor(0xCF, 0xD8, 0xE5))
    add_rect_band(slide, Inches(7.15), Inches(0.04), ACCENT)
    add_text(slide, Inches(0.4), Inches(7.20), Inches(12.5), Inches(0.25),
             "Stochastics of News Article Propagation  \u2014  Wang \u00b7 Madugula \u00b7 Thavamani",
             size=10, color=RGBColor(0x77, 0x77, 0x77))


def add_image_fit(slide, path, left, top, max_w, max_h):
    with Image.open(path) as im:
        w, h = im.size
    ratio = min(max_w / w, max_h / h)
    fw, fh = int(w * ratio), int(h * ratio)
    pic_l = left + (max_w - fw) // 2
    pic_t = top + (max_h - fh) // 2
    slide.shapes.add_picture(str(path), pic_l, pic_t, width=fw, height=fh)


# ---------------- Slide 1: Title ----------------
s = prs.slides.add_slide(BLANK)
add_rect_band(s, 0, SH, RGBColor(0xFA, 0xFA, 0xFC))
add_rect_band(s, Inches(2.9), Inches(0.06), NAVY)
add_rect_band(s, Inches(4.6), Inches(0.02), ACCENT)
add_text(s, Inches(0.8), Inches(1.4), Inches(11.7), Inches(0.5),
         "STAT 433  \u00b7  Final Project", size=14, bold=True, color=ACCENT,
         align=PP_ALIGN.CENTER)
add_text(s, Inches(0.8), Inches(3.0), Inches(11.7), Inches(1.4),
         "Stochastics of News Article Propagation",
         size=44, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
add_text(s, Inches(0.8), Inches(4.3), Inches(11.7), Inches(0.5),
         "Modelling counts and reliability dynamics around the 2017 Las Vegas shooting",
         size=18, color=SLATE, align=PP_ALIGN.CENTER, font="Calibri")
add_text(s, Inches(0.8), Inches(5.4), Inches(11.7), Inches(0.5),
         "Harry Wang  \u00b7  Sriyan Madugula  \u00b7  Ruthesh Thavamani",
         size=18, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
add_text(s, Inches(0.8), Inches(5.85), Inches(11.7), Inches(0.4),
         "University of Michigan", size=14, color=SLATE, align=PP_ALIGN.CENTER)

# ---------------- Slide 2: Dataset Overview ----------------
s = prs.slides.add_slide(BLANK)
slide_header(s, "Dataset Overview",
             "CC-News articles cross-joined with NewsGuard reliability scores")
add_bullets(s, Inches(0.45), Inches(1.15), Inches(6.4), Inches(5.6), [
    "Source: CC-News crawl, filtered to articles covering the 2017 Las Vegas shooting",
    "Date range: 2017-10-02 \u2192 2017-12-31 (91 days, zero-filled)",
    "2,702 articles  \u00b7  29.7 articles/day mean  \u00b7  peak 489 on day 0",
    "Reliability label constructed by joining CC-News with NewsGuard scores",
    ("Score thresholded at the median to produce a binary label \u00b1 1", 1),
    ("63.5% reliable  /  36.5% unreliable across the corpus", 1),
    "Limitations of the dataset",
    ("Binary reliability label \u2014 the underlying NewsGuard score is continuous", 1),
    ("Daily granularity only \u2014 within-day burst structure is invisible", 1),
    ("Single event window \u2014 generality across event types is untested", 1),
    ("Coverage gaps on quiet days are zero-filled, not imputed", 1),
], size=15)
add_image_fit(s, PARTB / "fig1_observed_distribution.png",
              Inches(7.0), Inches(1.2), Inches(6.0), Inches(5.7))
add_text(s, Inches(7.0), Inches(6.85), Inches(6.0), Inches(0.3),
         "Stacked daily counts by reliability category",
         size=10, color=RGBColor(0x77, 0x77, 0x77), align=PP_ALIGN.CENTER)

# ---------------- Slide 3: Part A models & base comparison ----------------
s = prs.slides.add_slide(BLANK)
slide_header(s, "Part A  \u00b7  Modelling the Counting Process",
             "Three base models fit by maximum conditional Poisson log-likelihood")
add_bullets(s, Inches(0.45), Inches(1.15), Inches(6.3), Inches(2.6), [
    "Standard Poisson  \u2014  \u03bb(t) = \u03bb  (1 param, null model)",
    "Inhomogeneous Poisson  \u2014  \u03bb(t) = A\u00b7e^(\u2212\u03b2t) + c  (3 params)",
    "Discrete-time Hawkes  \u2014  \u03bb(t) = \u03bc + \u03b1\u00b7R(t),  R(t)=e^(\u2212\u03b2)[R(t\u22121)+N(t\u22121)]  (2-3 params)",
    "Hawkes \u03b2-profile sweep collapses to AR(1) limit at \u03b2 \u2192 \u221e",
    ("Optimal kernel: \u03bb(t) = 11.50 + 0.614\u00b7N(t\u22121)", 1),
], size=14)

# Comparison table
from pptx.util import Inches as I
left = Inches(0.45); top = Inches(4.10); width = Inches(6.3); height = Inches(2.5)
rows = 4; cols = 5
table = s.shapes.add_table(rows, cols, left, top, width, height).table
headers = ["Model", "k", "log L", "AIC", "Pearson Var"]
data = [
    ["Standard Poisson", "1", "\u22123835.11", "7672.22", "196.3"],
    ["Inhomog. Poisson", "3", "\u2212513.15", "1032.29", "8.52"],
    ["Hawkes (AR-1)", "2", "\u22122003.81", "4011.62", "225.1"],
]
for j, h in enumerate(headers):
    c = table.cell(0, j)
    c.text = h
    for p in c.text_frame.paragraphs:
        for r in p.runs:
            r.font.bold = True; r.font.size = Pt(12)
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    c.fill.solid(); c.fill.fore_color.rgb = NAVY
for i, row in enumerate(data, 1):
    for j, val in enumerate(row):
        c = table.cell(i, j); c.text = val
        for p in c.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(11); r.font.name = "Calibri"
                if i == 2:
                    r.font.bold = True; r.font.color.rgb = ACCENT
add_text(s, Inches(0.45), Inches(6.65), Inches(6.3), Inches(0.4),
         "IHP wins decisively  \u2014  absorbs the day-0 exogenous shock the Hawkes cannot.",
         size=12, bold=True, color=NAVY)

add_image_fit(s, PARTA / "fig11_all_models_overlay.png",
              Inches(7.0), Inches(1.15), Inches(6.0), Inches(3.1))
add_image_fit(s, PARTA / "fig8_hawkes_beta_profile.png",
              Inches(7.0), Inches(4.30), Inches(6.0), Inches(2.85))

# ---------------- Slide 4: Part A hybrid winner ----------------
s = prs.slides.add_slide(BLANK)
slide_header(s, "Part A  \u00b7  Hybrid IHP + Hawkes Wins",
             "Combining exogenous trigger with endogenous self-excitation")
add_text(s, Inches(0.45), Inches(1.10), Inches(6.4), Inches(0.5),
         "Hybrid (full):  \u03bb(t) = A\u00b7e^(\u2212\u03b2\u2080 t) + c + \u03b1\u00b7R(t)",
         size=15, bold=True, color=NAVY)
add_bullets(s, Inches(0.45), Inches(1.55), Inches(6.4), Inches(3.0), [
    "IHP term absorbs the day-0 burst (489 articles)",
    "Hawkes term carries the self-exciting tail",
    "Two timescales: fast exogenous decay (1.75 d half-life) + slow endogenous persistence (8.2 d)",
    "LRT IHP \u2192 Hybrid-AR1: \u0394\u2113 = 124.4, p \u2248 0",
    "LRT Hybrid-AR1 \u2192 Hybrid-full: p = 3.7\u00d710\u207b\u2079",
], size=13)

left = Inches(0.45); top = Inches(4.45); width = Inches(6.4); height = Inches(2.4)
rows = 6; cols = 4
table = s.shapes.add_table(rows, cols, left, top, width, height).table
headers = ["Model", "k", "AIC", "BIC"]
data = [
    ["Standard Poisson", "1", "7672.22", "7674.73"],
    ["Inhomog. Poisson", "3", "1032.29", "1039.82"],
    ["Hawkes (AR-1)", "2", "4011.62", "4016.65"],
    ["Hybrid IHP+Hawkes (AR-1)", "4", "785.59", "795.63"],
    ["Hybrid IHP+Hawkes (full)", "5", "752.80", "765.35"],
]
for j, h in enumerate(headers):
    c = table.cell(0, j); c.text = h
    for p in c.text_frame.paragraphs:
        for r in p.runs:
            r.font.bold = True; r.font.size = Pt(11); r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    c.fill.solid(); c.fill.fore_color.rgb = NAVY
for i, row in enumerate(data, 1):
    for j, val in enumerate(row):
        c = table.cell(i, j); c.text = val
        for p in c.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(10); r.font.name = "Calibri"
                if i == 5:
                    r.font.bold = True; r.font.color.rgb = ACCENT

add_image_fit(s, PARTA / "fig13_hybrid_decomposition.png",
              Inches(7.0), Inches(1.10), Inches(6.0), Inches(3.0))
add_image_fit(s, PARTA / "fig14_hybrid_early_zoom.png",
              Inches(7.0), Inches(4.20), Inches(6.0), Inches(2.95))

# ---------------- Slide 5: Part B  setup ----------------
s = prs.slides.add_slide(BLANK)
slide_header(s, "Part B  \u00b7  Modelling the Reliability Distribution",
             "From counts to the reliable / unreliable split per day")
add_bullets(s, Inches(0.45), Inches(1.15), Inches(6.4), Inches(5.6), [
    "Goal: model q_t = P(article on day t is reliable | day t)",
    "Two daily series: N_neg(t), N_pos(t); empirical \u03c0_pos(t) = N_pos / N_tot",
    "Shared evaluation: conditional binomial log-likelihood given N_tot(t)",
    ("Hawkes implied split: q_t = \u03bb_pos / (\u03bb_pos + \u03bb_neg)", 1),
    ("Markov: q_t = (\u03c0_{t\u22121} \u00b7 P_t)_pos", 1),
    "Four candidate models",
    ("Multivariate Hawkes (AR-1 limit, 6 params, stationarity enforced)", 1),
    ("Homogeneous Markov \u2014 single transition matrix (2 params, null)", 1),
    ("State-dependent Markov \u2014 separate P for neg-/pos-majority days (4)", 1),
    ("Mean-field Markov \u2014 logistic links P(\u03c0_pos) (4 params)", 1),
], size=14)
add_image_fit(s, PARTB / "fig2_observed_proportion.png",
              Inches(7.05), Inches(1.20), Inches(6.0), Inches(5.7))
add_text(s, Inches(7.05), Inches(6.85), Inches(6.0), Inches(0.3),
         "Empirical \u03c0_pos(t) with 95% Wilson confidence bands",
         size=10, color=RGBColor(0x77, 0x77, 0x77), align=PP_ALIGN.CENTER)

# ---------------- Slide 6: Part B  Multivariate Hawkes ----------------
s = prs.slides.add_slide(BLANK)
slide_header(s, "Part B  \u00b7  Multivariate Hawkes \u2014 Cross-Excitation",
             "Branching matrix between reliable and unreliable streams")
add_bullets(s, Inches(0.45), Inches(1.15), Inches(6.4), Inches(2.8), [
    "Baselines: \u03bc_neg = 3.12, \u03bc_pos = 8.38",
    "Spectral radius \u03c1(n) = 0.633  \u2192  stationary",
    "Within-category excitation: n_nn = 0.44, n_pp = 0.37",
    "Cross-excitation is non-trivial: n_pn = 0.32, n_np = 0.16",
    "Reliability is not a closed ecosystem \u2014 streams interact",
], size=14)
add_text(s, Inches(0.45), Inches(4.0), Inches(6.4), Inches(0.5),
         "Justification: Hawkes is the natural multivariate generalisation of Part A's winner;",
         size=12, color=SLATE)
add_text(s, Inches(0.45), Inches(4.35), Inches(6.4), Inches(0.5),
         "branching matrix gives mechanistic, interpretable cross-excitation estimates.",
         size=12, color=SLATE)
add_text(s, Inches(0.45), Inches(5.05), Inches(6.4), Inches(0.5),
         "Interpretation",
         size=14, bold=True, color=NAVY)
add_bullets(s, Inches(0.45), Inches(5.45), Inches(6.4), Inches(1.8), [
    "An unreliable article today triggers ~0.32 reliable articles tomorrow",
    "A reliable article today triggers ~0.16 unreliable articles tomorrow",
    "Mildly diagonal-dominant, sub-critical, and consistent with Part A",
], size=12)
add_image_fit(s, PARTB / "fig3_hawkes_branching_heatmap.png",
              Inches(7.05), Inches(1.10), Inches(6.0), Inches(3.0))
add_image_fit(s, PARTB / "fig4_hawkes_fitted_vs_observed.png",
              Inches(7.05), Inches(4.20), Inches(6.0), Inches(2.95))

# ---------------- Slide 7: Part B  Markov variants ----------------
s = prs.slides.add_slide(BLANK)
slide_header(s, "Part B  \u00b7  Markov Variants \u2014 Regime Switching & Mean-Field",
             "Two parameterisations of state-dependent transition probabilities")
add_text(s, Inches(0.45), Inches(1.10), Inches(6.4), Inches(0.4),
         "State-dependent (regime-switching) Markov", size=14, bold=True, color=NAVY)
add_bullets(s, Inches(0.45), Inches(1.50), Inches(6.4), Inches(2.0), [
    "Two transition matrices keyed on yesterday's majority state",
    "Pos-majority regime: both classes are sticky (P_nn=0.68, P_pp=0.76)",
    "Neg-majority regime: dynamics push toward reliable (P_neg\u2192pos = 0.73)",
], size=12)
add_text(s, Inches(0.45), Inches(3.55), Inches(6.4), Inches(0.4),
         "Mean-field Markov", size=14, bold=True, color=NAVY)
add_bullets(s, Inches(0.45), Inches(3.95), Inches(6.4), Inches(2.6), [
    "Logistic links: P_neg\u2192pos = \u03c3(a\u2080 + a\u2081\u00b7\u03c0_pos),  P_pos\u2192neg = \u03c3(b\u2080 + b\u2081\u00b7\u03c0_pos)",
    "Fitted: a\u2080=+1.15, a\u2081=\u22123.96 \u2192 strong mean reversion",
    "When reliable share \u2193, conversion to reliable \u2191 sharply (0.76 \u2192 0.06)",
    "Smooth, identifiable analogue of the regime-switching matrix",
], size=12)
add_image_fit(s, PARTB / "fig5_markov_regime_matrices.png",
              Inches(7.05), Inches(1.10), Inches(6.0), Inches(3.0))
add_image_fit(s, PARTB / "fig6_meanfield_P_trajectory.png",
              Inches(7.05), Inches(4.20), Inches(6.0), Inches(2.95))

# ---------------- Slide 8: Part B  Comparison ----------------
s = prs.slides.add_slide(BLANK)
slide_header(s, "Part B  \u00b7  Model Comparison on Shared Binomial Scale",
             "Effective sample size n_eff = 86 days with at least one article")

left = Inches(0.45); top = Inches(1.15); width = Inches(6.4); height = Inches(2.4)
rows = 5; cols = 5
table = s.shapes.add_table(rows, cols, left, top, width, height).table
headers = ["Model", "k", "log L", "AIC", "BIC"]
data = [
    ["Multivariate Hawkes", "6", "\u2212156.93", "325.87", "340.60"],
    ["Homogeneous Markov", "2", "\u2212162.75", "329.50", "334.41"],
    ["State-dependent Markov", "4", "\u2212157.33", "322.65", "332.47"],
    ["Mean-field Markov", "4", "\u2212157.56", "323.13", "332.94"],
]
for j, h in enumerate(headers):
    c = table.cell(0, j); c.text = h
    for p in c.text_frame.paragraphs:
        for r in p.runs:
            r.font.bold = True; r.font.size = Pt(12); r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    c.fill.solid(); c.fill.fore_color.rgb = NAVY
for i, row in enumerate(data, 1):
    for j, val in enumerate(row):
        c = table.cell(i, j); c.text = val
        for p in c.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(11); r.font.name = "Calibri"
                if i == 3:
                    r.font.bold = True; r.font.color.rgb = ACCENT

add_text(s, Inches(0.45), Inches(3.7), Inches(6.4), Inches(0.4),
         "Likelihood ratio tests vs homogeneous baseline",
         size=13, bold=True, color=NAVY)
add_bullets(s, Inches(0.45), Inches(4.05), Inches(6.4), Inches(1.4), [
    "State-dep Markov:  LR = 10.85, df = 2,  p = 4.4\u00d710\u207b\u00b3",
    "Mean-field Markov:  LR = 10.37, df = 2,  p = 5.6\u00d710\u207b\u00b3",
], size=12)
add_text(s, Inches(0.45), Inches(5.45), Inches(6.4), Inches(0.5),
         "Why Markov beats Hawkes here",
         size=14, bold=True, color=NAVY)
add_bullets(s, Inches(0.45), Inches(5.85), Inches(6.4), Inches(1.4), [
    "Hawkes spends parameters on total counts, not the split",
    "~86% of days are pos-majority \u2014 a single P captures most dynamics",
], size=12)

add_image_fit(s, PARTB / "fig7_fitted_proportion_comparison.png",
              Inches(7.05), Inches(1.10), Inches(6.0), Inches(3.0))
add_image_fit(s, PARTB / "fig9_model_comparison.png",
              Inches(7.05), Inches(4.20), Inches(6.0), Inches(2.95))

# ---------------- Slide 9: Part B interpretation ----------------
s = prs.slides.add_slide(BLANK)
slide_header(s, "Part B  \u00b7  Interpretation \u2014 A Mean-Reverting Reliability Ecology",
             "What the winning models say about the news ecosystem")
add_bullets(s, Inches(0.45), Inches(1.15), Inches(12.4), Inches(5.5), [
    "The system has a stable equilibrium near \u03c0_pos \u2248 0.63 \u2014 the empirical reliable share",
    "When the reliable share drops below 0.5 (only at the initial shock), strong restoring forces fire",
    ("State-dependent Markov: P_neg\u2192pos jumps to 0.73 in the neg-majority regime", 1),
    ("Mean-field Markov: P_neg\u2192pos is a steeply decreasing function of \u03c0_pos", 1),
    "Hawkes branching matrix encodes the same story mechanistically:",
    ("Reliable articles preferentially trigger more reliable articles (n_pp > n_np)", 1),
    ("Unreliable articles trigger reliable and unreliable articles nearly symmetrically", 1),
    "Choice of model depends on the question being asked",
    ("Distribution dynamics  \u2192  Markov variants are most efficient", 1),
    ("Mechanistic cross-excitation  \u2192  Multivariate Hawkes is most informative", 1),
    "Synthetic recovery passed (max rel error 20% on T=500) \u2014 estimates are trustworthy",
], size=15)

# ---------------- Slide 10: Conclusions + next steps ----------------
s = prs.slides.add_slide(BLANK)
slide_header(s, "Conclusions and Next Steps")
add_text(s, Inches(0.45), Inches(1.10), Inches(6.2), Inches(0.45),
         "Summary of findings", size=16, bold=True, color=NAVY)
add_bullets(s, Inches(0.45), Inches(1.55), Inches(6.2), Inches(5.5), [
    "Counting process is best modelled by a hybrid IHP + Hawkes",
    ("Exogenous shock + endogenous amplification at distinct timescales", 1),
    "Reliability split is best modelled by a state-dependent Markov chain",
    ("Mean-field Markov is statistically indistinguishable and smoother", 1),
    "Cross-excitation between reliable and unreliable streams is non-trivial",
    "Reliability ecology is mean-reverting around the empirical equilibrium",
], size=14)
add_text(s, Inches(7.0), Inches(1.10), Inches(6.0), Inches(0.45),
         "Next steps", size=16, bold=True, color=NAVY)
add_bullets(s, Inches(7.0), Inches(1.55), Inches(6.0), Inches(5.5), [
    "McKean\u2013Vlasov formulation for continuous-time, distribution-dependent transitions",
    ("Replace discrete P_t with a SDE driven by the empirical \u03c0_pos measure", 1),
    "Use raw NewsGuard scores (continuous) instead of the binary label",
    ("Recovers within-class heterogeneity lost to the median threshold", 1),
    ("Enables regression / functional models on a real-valued reliability axis", 1),
    "Replicate across additional events to test generality of the hybrid + Markov story",
    "Extend hybrid model with explicit weekly seasonality (\u0394AIC \u2248 \u2212148 in exploration)",
], size=14)

prs.save(OUT)
print(f"Wrote {OUT}")

#!/usr/bin/env python3
"""Build the EUROMECH talk deck (editable .pptx) for the vortex-gust JEPA paper.

Figures are pre-rasterized PNGs in paper/talk/figs/ (see the conversion step in
the build notes). Numbers come from paper/HEADLINE_NUMBERS.md.

Run:  python paper/talk/make_talk.py   ->   paper/talk/euromech_gust_jepa.pptx
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figs"
OUT = HERE / "euromech_gust_jepa.pptx"

# ---- palette / type ---------------------------------------------------------
NAVY = RGBColor(0x1F, 0x38, 0x64)
ACCENT = RGBColor(0xC0, 0x52, 0x10)   # warm rust for emphasis / glyphs
INK = RGBColor(0x24, 0x29, 0x33)
MUTE = RGBColor(0x6B, 0x72, 0x80)
RULE = RGBColor(0xD6, 0xDB, 0xE3)
LIGHT = RGBColor(0xC9, 0xD3, 0xE6)
FONT = "Calibri"

W, H = 13.333, 7.5

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]


# ---- low-level helpers ------------------------------------------------------
def slide():
    return prs.slides.add_slide(BLANK)


def rect(s, l, t, w, h, fill=None, line=None):
    sp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    sp.shadow.inherit = False
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(1.0)
    return sp


def _runs(p, text, size, color, base_bold=False, bold_color=NAVY):
    for i, seg in enumerate(text.split("**")):
        if seg == "":
            continue
        r = p.add_run()
        r.text = seg
        r.font.name = FONT
        r.font.size = Pt(size)
        emph = (i % 2 == 1)
        r.font.bold = base_bold or emph
        r.font.color.rgb = bold_color if emph else color


def textbox(s, l, t, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    return tf


def para(tf, text, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT,
         space_before=0, space_after=6, bold_color=None, first=False, glyph=None):
    if first and not tf.paragraphs[0].runs:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    if glyph:
        g = p.add_run()
        g.text = glyph
        g.font.name = FONT
        g.font.size = Pt(size)
        g.font.bold = True
        g.font.color.rgb = bold_color or ACCENT
    _runs(p, text, size, color, base_bold=bold, bold_color=bold_color or NAVY)
    return p


def bullets(s, items, l, t, w, h, base=17):
    tf = textbox(s, l, t, w, h)
    first = True
    for it in items:
        lvl = it.get("lvl", 0)
        sz = it.get("sz", base if lvl == 0 else base - 3)
        sa = it.get("sa", 9 if lvl == 0 else 5)
        if lvl == 0:
            para(tf, it["t"], size=sz, color=INK, space_after=sa, glyph="•  ", first=first)
        else:
            para(tf, "      –  " + it["t"], size=sz, color=MUTE, space_after=sa, first=first)
        first = False
    return tf


def fig(s, name, l, t, w, h, halign="center", valign="middle"):
    path = FIG / name
    iw, ih = Image.open(path).size
    box_ar, img_ar = w / h, iw / ih
    if img_ar > box_ar:
        nw, nh = w, w / img_ar
    else:
        nh, nw = h, h * img_ar
    x = l + (w - nw) / 2 if halign == "center" else (l if halign == "left" else l + (w - nw))
    y = t + (h - nh) / 2 if valign == "middle" else (t if valign == "top" else t + (h - nh))
    s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(nw), Inches(nh))


def caption(s, text, l, t, w):
    tf = textbox(s, l, t, w, 0.4)
    para(tf, text, size=11, color=MUTE, align=PP_ALIGN.CENTER, first=True)


def header(s, title, kicker=None):
    if kicker:
        tf = textbox(s, 0.55, 0.30, 11.5, 0.32)
        para(tf, kicker.upper(), size=12.5, color=ACCENT, bold=True, space_after=0, first=True)
    tf2 = textbox(s, 0.55, 0.60, 12.25, 0.95)
    para(tf2, title, size=26, color=NAVY, bold=True, space_after=0, first=True)
    rect(s, 0.55, 1.50, 12.23, 0.022, fill=RULE)
    rect(s, 0.55, 1.485, 2.1, 0.05, fill=ACCENT)


def left_footer(s):
    tf = textbox(s, 0.55, 7.06, 9.5, 0.3)
    para(tf, "Predictive latent dynamics for extreme gusts  ·  EUROMECH colloquium",
         size=9.5, color=MUTE, first=True)


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


# ---- slide templates --------------------------------------------------------
def s_title():
    s = slide()
    rect(s, 0, 0, 0.30, H, fill=NAVY)
    tf = textbox(s, 0.95, 1.35, 11.7, 2.5)
    para(tf, "A predictive latent dynamics model for extreme gust encounters",
         size=33, color=NAVY, bold=True, space_after=6, first=True)
    para(tf, "Toward model-based flow control", size=21, color=ACCENT, bold=True, space_after=0)
    rect(s, 1.0, 3.95, 3.0, 0.06, fill=ACCENT)
    tf2 = textbox(s, 1.0, 4.2, 11.5, 1.5)
    para(tf2, "Forward physical closure of predictive vs reconstructive latents at Re = 5000",
         size=15, color=INK, first=True, space_after=12)
    para(tf2, "A. Solera-Rico, A. Miró, O. Lehmkuhl, C. Sanmiguel Vila",
         size=14, color=INK, space_after=3)
    para(tf2, "INTA  ·  Universidad Carlos III de Madrid  ·  UPC  ·  Barcelona Supercomputing Center",
         size=11.5, color=MUTE, space_after=0)
    tf3 = textbox(s, 1.0, 6.55, 11.5, 0.5)
    para(tf3, "EUROMECH Colloquium  ·  Data-driven active control in flows: from model-based to reinforcement learning",
         size=12, color=NAVY, bold=True, first=True)
    notes(s, "Frame: extreme gust control needs a fast, faithful forward model. Our work "
              "asks which reduced state is worth planning against, and shows a predictive "
              "(JEPA) latent captures the wake physics that reconstructive and linear "
              "baselines lose. Audience is new to JEPA, so I will spend a slide on what it is.")
    return s


def s_bullets(title, kicker, items, note="", lead=None, base=18):
    s = slide()
    header(s, title, kicker)
    if lead:
        tf = textbox(s, 0.6, 1.78, 12.1, 0.8)
        para(tf, lead, size=19, color=INK, first=True)
        bullets(s, items, 0.7, 2.65, 12.0, 4.1, base=base)
    else:
        bullets(s, items, 0.7, 1.9, 12.0, 5.0, base=base)
    left_footer(s)
    if note:
        notes(s, note)
    return s


def s_fig_right(title, kicker, items, figname, cap=None, note="", fl=7.7, fw=5.2, base=17):
    s = slide()
    header(s, title, kicker)
    bullets(s, items, 0.6, 1.85, fl - 0.85, 5.0, base=base)
    fig(s, figname, fl, 1.82, fw, 4.8, valign="top")
    if cap:
        caption(s, cap, fl, 6.68, fw)
    left_footer(s)
    if note:
        notes(s, note)
    return s


def s_fig_below(title, kicker, items, figname, cap=None, note="", ft=3.55, fh=3.05, base=17):
    s = slide()
    header(s, title, kicker)
    bullets(s, items, 0.7, 1.78, 12.0, ft - 1.85, base=base)
    fig(s, figname, 0.8, ft, 11.7, fh, valign="top")
    if cap:
        caption(s, cap, 0.8, ft + fh + 0.02, 11.7)
    left_footer(s)
    if note:
        notes(s, note)
    return s


def s_question(title, kicker, items, quote, note=""):
    s = slide()
    header(s, title, kicker)
    bullets(s, items, 0.7, 1.85, 12.0, 2.4, base=18)
    rect(s, 0.7, 4.55, 11.9, 1.7, fill=RGBColor(0xF3, 0xF5, 0xF9))
    rect(s, 0.7, 4.55, 0.10, 1.7, fill=ACCENT)
    tf = textbox(s, 1.05, 4.55, 11.3, 1.7, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, quote, size=22, color=NAVY, bold=True, first=True)
    left_footer(s)
    if note:
        notes(s, note)
    return s


def s_divider(title, sub=None):
    s = slide()
    rect(s, 0, 0, W, H, fill=NAVY)
    tf = textbox(s, 1.0, 2.85, 11.3, 1.6, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, title, size=36, color=RGBColor(0xFF, 0xFF, 0xFF), bold=True, first=True)
    if sub:
        para(tf, sub, size=16, color=LIGHT, space_before=6)
    rect(s, 1.05, 4.25, 3.0, 0.06, fill=ACCENT)
    return s


def b0(t, **k):
    return {"t": t, "lvl": 0, **k}


def b1(t, **k):
    return {"t": t, "lvl": 1, **k}


# ---- build ------------------------------------------------------------------
s_title()

# I. Motivation & why JEPA could matter
s_fig_below(
    "The control problem", "Motivation",
    [b0("Small and micro air vehicles increasingly fly where the **gust speed is comparable to or larger than the flight speed**, urban canyons, building and ship wakes"),
     b0("Gust ratio **G > 1**: large, fast load transients that conventional gust models miss"),
     b0("Active control needs a **fast, faithful forward model**, model-based control and RL world-models both rest on a learned reduced dynamics"),
     b0("This talk: **what reduced state is worth planning against?**")],
    "graphical_abstract.png",
    cap="Parametric vortex-gust encounters on a NACA 0012 at Re = 5000.",
    ft=4.6, fh=2.1,
    note="Pitch to a control audience: the bottleneck for model-based control / RL in this "
         "regime is a reduced dynamics model that stays trustworthy under rollout.")

s_fig_right(
    "Why it is hard: it is the wake, not just the forces", "The physics",
    [b0("The load transient is built by **leading-edge-vortex (LEV)** formation, growth and shedding, plus the wake left behind"),
     b0("The discriminating information lives in the **wake**, not only in the integrated forces C_L, C_D"),
     b0("A model that tracks forces but loses the wake looks fine **instantaneously, yet fails under rollout**"),
     b0("α = 14°, Re = 5000: massively separated, nonlinear coupling with the natural shedding")],
    "scale_decomp.png",
    cap="Large-scale wake / LEV organisation governs the transient.",
    note="Key message that motivates the whole study: forces are necessary but not sufficient; "
         "the wake is the hard, control-relevant part.")

s_question(
    "The opening for JEPA", "The idea",
    [b0("Most flow ROMs, POD/DMD, autoencoders, are trained to **reconstruct the field**"),
     b1("reconstruction fixes the latent only up to a diffeomorphism, so its geometry is not constrained to be predictable"),
     b0("Alternative: learn a state that is **predictive by construction**, and plan against it")],
    "Which reduced state stays physically closed when propagated forward,\n"
    "so that every observable remains recoverable along the rollout?",
    note="Set up the central question. Closure = the rolled-out latent still lets you read off "
         "the physical observables. This is exactly the property a planner or RL agent needs.")

# II. What a JEPA is, and our instantiation
s_fig_below(
    "Data", "Data",
    [b0("**DNS** (SOD2D, no subgrid-scale model), NACA 0012, α = 14°, **Re = 5000**"),
     b0("Taylor-vortex gusts parametrised by strength **G**, core diameter **D**, wall-normal offset **Y**"),
     b0("84 cases; impact-centred encounters; six observables: C_L, C_D, impulse I_y, **wake enstrophy**, ±circulation"),
     b0("Held-out **test_b** (interpolation) and **test_c** (|G| = 4 extrapolation)")],
    "paramspace.png",
    cap="Training envelope in (G, D, Y); held-out interpolation and extrapolation.",
    ft=4.55, fh=2.15,
    note="Keep the DNS claim bounded (solver-resolution metadata pending). Emphasise the six "
         "observables, wake enstrophy being the one that separates the methods.")

s_fig_right(
    "What is a JEPA?", "Method, primer",
    [b0("**JEPA** = Joint-Embedding Predictive Architecture (LeCun's world-model program)"),
     b0("Encoder x → z; a predictor advances **z** in time; the **loss lives in latent space, the field is never reconstructed** (an anti-collapse term replaces the decoder)"),
     b0("So the encoder drops detail not predictive of the future, and the predictor gets no pixel-level shortcuts"),
     b0("Lineage: I-JEPA / V-JEPA (image, video); from-pixels LeWM, LeJEPA, PLDM, so far only gridworld and toy visual tasks"),
     b0("**Why for control:** a learned world-model, cheap latent rollout, keeps control-relevant dynamics, sensor-observable; the **first end-to-end JEPA on a parametric flow** (to our knowledge)")],
    "predict_vs_reconstruct.png",
    cap="Reconstruct the field vs predict in latent space.",
    note="Audience is new to JEPA: spend time here. The contrast with reconstruction is the "
         "crux. Stress the world-model framing for the control venue.", fl=7.9, fw=5.0, base=16)

s_fig_right(
    "Our architecture for gust encounters", "Method",
    [b0("**Encoder** (hybrid CNN + ViT) is **unconditional**: mid-plane vorticity ω_z → latent z (d = 32 / 64)"),
     b0("**Predictor** (autoregressive transformer): the gust **c = (G, D, Y) enters the predictor only** (AdaLN, RoPE, causal mask)"),
     b0("The predictor **is** the latent dynamics model, the object you plan or learn against"),
     b0("A visualisation decoder is trained on the **frozen** encoder, never in the loss"),
     b0("Compared at matched latent dimension vs a reconstructive AE (Fukami lineage) and POD")],
    "jepa_architecture.png",
    note="Conditioning split is deliberate: the encoder is a pure state map; the gust enters "
         "the dynamics. For control, an actuation channel would enter the same way.", fl=8.2, fw=4.8)

# III. Protocol
s_fig_below(
    "Forward-closure protocol", "Evaluation",
    [b0("**Forward closure:** roll the predictor recursively to **H = 16** frames after impact, then probe each observable from the predicted latent"),
     b0("**Matched protocol:** same predictor architecture and same probe family, **trained / fitted separately per latent**, so differences are the encoder's"),
     b0("**Conditioning-only floor:** KRR on (G, D, Y) alone, the latent must beat it to prove the state, not the parameters, does the work"),
     b0("Reported with bootstrap CIs, 3 encoder seeds, 5-fold probe CV; held-out test_b / test_c")],
    "eval_protocol.png",
    cap="The same predictor and probe map every encoder family to the observables, in two modes.",
    ft=4.35, fh=2.1,
    note="Fairness is the whole point: only the encoder changes. The conditioning floor rules "
         "out 'the parameters did it'.")

# IV. Results
s_fig_right(
    "Main result: forward closure", "Result",
    [b0("Closure R² (rolled-out latent → observable, mean over 6): **JEPA 0.84 · Fukami 0.43 · POD 0.56**"),
     b0("**Wake enstrophy** is the discriminator: **0.93** vs 0.28 (Fukami) / 0.37 (POD)"),
     b0("Held-out MAE at H = 16: JEPA wake enstrophy **2.4× lower** than the AE, **3× lower** than POD"),
     b0("POD stays competitive only on the integrated impulse I_y")],
    "closure.png",
    cap="Held-out MAE at H = 16 by observable (test B, test C); JEPA lowest on the wake.",
    note="Headline. Wake enstrophy is where predictive wins decisively; forces are easy for "
         "everyone.")

s_fig_right(
    "It is the wake, not the forces", "Result",
    [b0("**Forces** (C_L, C_D) are forecast **redundantly by every family**, many coordinates carry them"),
     b0("Only the predictive latent carries a **distributed wake-forecast code** (rank corr 0.83; best single coordinate 0.44)"),
     b0("Reconstructive / linear: best single coordinate ≈ the whole latent → **no collective wake structure**"),
     b0("And it clears the **conditioning-only floor** → the latent, not the parameters, does the work")],
    "wake_code.png",
    cap="Wake forecast skill: full latent (dark) vs best single coordinate (light).",
    note="This explains the main result mechanistically at the representation level: the wake "
         "is a collective code only the predictive objective builds.")

s_fig_right(
    "Controls: objective and supervision, not architecture", "Result, controls",
    [b0("**2 × 2:** objective {predictive, reconstructive} × architecture {CNN, CNN+ViT}, auxiliary heads matched"),
     b0("Predictive beats reconstructive at **both** architectures (wake R² 0.46 / 0.45 vs 0.16 / 0.29) → **not the ViT**"),
     b0("Remove the wake head from the predictive model → wake closure **collapses below floor (R² = −1.03)**"),
     b0("→ the gain needs the **predictive objective and wake supervision together**")],
    "controls_fairness.png",
    cap="Objective × architecture controls, three seeds per cell.",
    note="Pre-empt the skeptic: isolate the cause before the mechanism. Honest two-part answer: "
         "objective carries it, but the wake head is a necessary ingredient.")

s_fig_right(
    "Diagnostic 1: latent drift", "Mechanism",
    [b0("**Why this question:** a planner / RL agent queries the model at states reached by its **own rollout**, one-step error is not enough"),
     b0("**Mahalanobis distance:** covariance-aware distance from a reference cloud, d = sqrt((z−μ)' S^-1 (z−μ)); ≈ 1 is one standard-deviation unit"),
     b0("We compare the **rolled-out** latent to the distribution of **DNS-encoded** latents (ratio = rollout / encoded)"),
     b0("**Result:** the reconstructive rollout drifts **~10× off-manifold (9.9)**; JEPA 0.85 and POD 0.81 stay inside")],
    "horizon_sweep.png",
    cap="Forecast skill vs rollout horizon, held-out test_b.",
    note="Non-standard metric, so define it. The point: low one-step error is worthless if the "
         "rollout walks off the data manifold where probes and value estimates are invalid.",
    base=16)

s_fig_right(
    "Diagnostic 2: topology of the encounter", "Mechanism",
    [b0("**Why:** shedding and gust encounters are **cyclic**; a faithful dynamics model should keep the trajectory **one closed loop** under rollout"),
     b0("**Persistent homology:** counts topological loops (1-cycles) and how long they persist as a scale parameter grows; one long-lived cycle = a clean loop, many short-lived ones = fragmentation"),
     b0("**Result:** the predictive latent traces a **single persistent cycle**; the reconstructive latent **fragments**"),
     b0("→ the predictive state is a coherent object to integrate forward")],
    "cycle.png",
    cap="The encounter is a single cycle in the predictive latent.",
    note="Persistent homology is unfamiliar to most fluids people: one slide to define it and "
         "say why a clean cycle matters for stable rollout.", base=16)

s_fig_right(
    "Diagnostic 3: transport geometry", "Mechanism",
    [b0("**Why:** Euclidean / pixel distance ignores **where structures move**; control cares about advection of the LEV and shear layer"),
     b0("**Optimal-transport (OT) distance:** the least cost to rearrange one vorticity field into another, sensitive to transport"),
     b0("We Spearman-correlate the **latent** distance matrix with the **OT** distance matrix along each encounter"),
     b0("**Result:** JEPA **0.63** vs Fukami 0.45, order-preservation, not an isometry; latent steps mean physically sensible field changes")],
    "ot.png",
    cap="Latent vs OT distance alignment (per-encounter Shepard).",
    note="Third non-standard metric. Be precise: we claim order-preservation of transport "
         "distances, not a metric isometry.", base=16)

s_fig_right(
    "Physical space: the large-scale wake is recovered", "Result, physical space",
    [b0("Decode the latents back to fields (frozen-encoder diagnostic, never in the loss)"),
     b0("The predictive decode tracks the **large-scale LEV / wake**: Pearson **0.89** at impact, **0.91** at H = 16"),
     b0("The reconstructive decode is sharper pixel-wise but **loses the transported wake** under rollout"),
     b0("This is the explicit trade the predictive objective makes")],
    "flow_recovery.png",
    cap="Large-scale wake recovery, predictive vs reconstructive.",
    note="Connect the scalar wake result back to the flow field a fluids audience wants to see.")

# V. Control relevance
s_fig_right(
    "Control relevance: observable and forecastable", "Control relevance",
    [b0("The predictive state is **observable from sparse wall pressure**: (G, D) recoverable from the latent (R² ≈ 0.46 / 0.80 on test_b)"),
     b0("**Lead-time impact-C_L:** predictor-in-loop **R² = 0.35 at 10 frames pre-impact**, vs 0.13 from direct pressure sensing (oracle 0.68)"),
     b0("The reconstructive latent's oracle is **negative**, a representation failure, not a probe failure"),
     b0("A deployment-relevant **state estimate plus short-horizon forecast**: the ingredients a controller consumes")],
    "cl_inference_simple.png",
    cap="Inferring impact-C_L from K = 8 wall-pressure sensors (test_b): the predictor-rolled latent beats direct sensing well before impact.",
    note="The venue hook: the predictive latent gives both a sensor-observable state and a "
         "useful pre-impact forecast, the two ingredients a controller needs.",
    fl=7.5, fw=5.5, base=16)

# VI. Close
s_bullets(
    "Takeaways", "Takeaways",
    [b0("A **predictive objective + wake supervision** yields a latent dynamics model that is:"),
     b1("**forward-closed** (wake R² 0.75 representational / 0.84 mean), **on-manifold** under rollout, a **single cycle**, **transport-aligned**"),
     b1("**compact**, d = 32 ≈ d = 64, participation ratio ≈ 1.7"),
     b1("**observable** from sparse wall pressure"),
     b0("A **conditional forward-closure model**, a substrate for **model-based control and RL world-models**"),
     b0("The discriminator throughout is the **wake**, which force-only and reconstruction-only states lose")],
    note="Land the one-sentence message: predictive + wake supervision buys a compact, "
         "trustworthy-under-rollout, observable latent dynamics model.")

s_bullets(
    "Outlook", "Outlook",
    [b0("Replace the gust conditioning channel with an **actuation channel**, the same forward-closure machinery then applies to control inputs"),
     b0("**Closed-loop** model-based control / RL world-models built on the latent"),
     b0("**3D observability** at strong gusts (|G| = 4 is a mid-plane-slice limit)"),
     b0("The predictive-latent recipe carried to **other parametric flows**")],
    note="Point forward to the RL half of the colloquium: this is the world-model substrate; "
         "closing the loop is the natural next step.")

# ---- backups ----------------------------------------------------------------
s_divider("Backup slides", "Supporting detail and ablations")

s_fig_right(
    "Dataset and protocol", "Backup",
    [b0("Partition **v2** (locked, sha256-anchored): 84 cases"),
     b0("**226** train encounters · **42** test_b · **24** test_c (|G| = 4)"),
     b0("ω_z pipeline v1: spatial mask + p99.99 clip + 3σ scale (train_std 3.55), no mean shift (preserves vorticity antisymmetry)"),
     b0("Reporting: bootstrap n = 2000 CIs · 3 encoder seeds · 5-fold probe CV")],
    "paramspace.png", cap="Training envelope in (G, D, Y).")

s_bullets(
    "Matched-capacity: d = 32 vs d = 64", "Backup",
    [b0("Halving the latent leaves the representation and **every mechanism diagnostic** intact"),
     b1("representational wake R² 0.74 (d=32) vs 0.75 (d=64); drift ratio 0.86 vs 0.85; OT 0.61 vs 0.63"),
     b0("Cost is **in-distribution forecast sharpness**: H = 16 wake closure 0.45 → 0.21"),
     b0("Participation ratio ≈ 1.7, leading PC ≈ ¾ of variance → the effective dimension is a handful")])

s_bullets(
    "Seed variance", "Backup",
    [b0("Three encoder seeds per cell; wake-closure means reported ± standard deviation"),
     b0("The reconstructive CNN+ViT cell has **large seed variance (±0.27)**, consistent with the drift mechanism: without a forward-predictable geometry, closure is unstable across initialisations"),
     b0("Predictive cells are tight (±0.03–0.06)")])

s_fig_below(
    "Decoded reconstructions", "Backup",
    [b0("Decoded fields: predictive vs reconstructive vs POD vs DNS"),
     b0("The predictive decode is **blurrier pixel-wise** but preserves the **transported large-scale wake**; the reconstructive decode is sharp yet drifts under rollout")],
    "reconstructions.png", cap="Held-out reconstructions; the predictive objective trades pixel sharpness for transported structure.",
    ft=3.4, fh=3.2)

s_fig_below(
    "Sparse sensor placement", "Backup",
    [b0("Sparse wall-pressure sensors selected by **TCSI / qDEIM** vs uniform"),
     b0("Pressure → latent map: KernelRidge(RBF) on the K-sensor × pre-impact-window vector"),
     b0("K = 2 / 4 / 8 / 16; the LSTM/KRR comparison is 5-fold-CV selected to guard small-sample overfitting")],
    "sensor_placement.png", cap="Optimal sparse sensor placement on the airfoil surface.",
    ft=3.5, fh=3.1)

s_fig_below(
    "Parameter observability from the latent", "Backup",
    [b0("z → (G, D, Y) probe R² from the rolled-out latent, K = 8 pre-impact sensors"),
     b0("test_b: **G 0.46 · D 0.80 · Y 0.10**, gust impulse and depth recoverable; cross-stream offset Y is marginal"),
     b0("test_c (|G| = 4): negative, the 3D observability boundary of a single mid-plane slice")],
    "pressure_observability.png", cap="Gust parameters implicit in the latent on held-out cases.",
    ft=3.7, fh=2.6)

s_bullets(
    "SSIM convention", "Backup",
    [b0("Reconstruction SSIM uses the **Wang convention** (K1 = 0.01, K2 = 0.03) on pipeline-normalised ω"),
     b0("Data range **L = 2 · global p99.9(|target|) ≈ 8.31** (split v2)"),
     b0("Decoder test_a SSIM ≈ **0.71** at this convention")])

s_bullets(
    "Conditioning-only floor", "Backup",
    [b0("KRR-RBF on (G, D, Y) → observable at impact; train / test_b / test_c R²"),
     b1("on train, (G, D, Y) interpolates C_L and C_D well (226 points in 3-D)"),
     b0("On **test_b** the floor **collapses** across observables; on **test_c** it is negative for 5 of 6"),
     b0("→ JEPA's generalisation is **not** explained by the conditioning alone")])

s_fig_right(
    "Predictor detail", "Backup",
    [b0("Predictor: 6-layer autoregressive transformer, hidden 384, 16 heads, dropout 0.1"),
     b0("**AdaLN-Zero** conditioning on (G, D, Y, φ_t); **RoPE** temporal positions; **causal** mask"),
     b0("Training: teacher forcing + scheduled-sampling rollout (H_roll = 8); **SIGReg** anti-collapse (VICReg fallback)")],
    "predictor_detail.png", fl=8.0, fw=4.9)

s_fig_right(
    "Phase–amplitude reading", "Backup",
    [b0("Phase–amplitude decomposition of the encounter cycle in the predictive latent"),
     b0("Connects the predictor to the **sensitivity-function control** designed for these flows"),
     b0("An actuation channel would enter where the gust does, the predictor conditioning")],
    "phase_amplitude.png", fl=7.9, fw=5.0)

# ---- page numbers (skip the title) -----------------------------------------
for i, sl in enumerate(prs.slides):
    if i == 0:
        continue
    tf = textbox(sl, 12.35, 7.06, 0.75, 0.3)
    para(tf, str(i + 1), size=9.5, color=MUTE, align=PP_ALIGN.RIGHT, first=True)

prs.save(str(OUT))
print(f"wrote {OUT}  ({len(prs.slides)} slides)")

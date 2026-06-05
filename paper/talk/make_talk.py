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


ASSET = HERE / "assets"
_ASSET_CACHE = FIG / "_assetcache"
ASSETS = {
    "euromech": "IMG_2285.jpeg", "inta": "IMG_2286.png", "uc3m": "IMG_2289.png",
    "bsc": "IMG_2287.png", "upc": "IMG_2288.png", "bbva": "IMG_2283.jpeg",
    "redleonardo": "IMG_2282.png", "photo_solera": "IMG_2292.png",
    "photo_miro": "IMG_2290.jpeg", "photo_lehmkuhl": "IMG_2291.jpeg",
}


def asset_path(key):
    p = ASSET / ASSETS[key]
    return p if p.exists() else None


def place_image(s, path, l, t, w, h, halign="center", valign="middle"):
    iw, ih = Image.open(path).size
    box_ar, img_ar = w / h, iw / ih
    if img_ar > box_ar:
        nw, nh = w, w / img_ar
    else:
        nh, nw = h, h * img_ar
    x = l + (w - nw) / 2 if halign == "center" else (l if halign == "left" else l + (w - nw))
    y = t + (h - nh) / 2 if valign == "middle" else (t if valign == "top" else t + (h - nh))
    s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(nw), Inches(nh))


def logo(s, key, l, t, w, h, halign="center", valign="middle", crop=None):
    """Aspect-fit a logo (by asset key) into a box; silently skip if missing."""
    p = asset_path(key)
    if p is None:
        return
    if crop is not None:
        _ASSET_CACHE.mkdir(exist_ok=True)
        p = _ASSET_CACHE / f"{key}_crop.png"
        Image.open(asset_path(key)).crop(crop).save(p)
    place_image(s, p, l, t, w, h, halign, valign)


def square_photo(key, focus_x=0.5, focus_y=0.45):
    """Center-crop a photo to a square (with focus bias), cache, return path or None."""
    p = asset_path(key)
    if p is None:
        return None
    _ASSET_CACHE.mkdir(exist_ok=True)
    out = _ASSET_CACHE / f"{key}_sq.png"
    im = Image.open(p).convert("RGB")
    w, h = im.size
    side = min(w, h)
    cx, cy = int(w * focus_x), int(h * focus_y)
    left = max(0, min(w - side, cx - side // 2))
    top = max(0, min(h - side, cy - side // 2))
    im.crop((left, top, left + side, top + side)).save(out)
    return str(out)


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
    para(tf, "Sparse-sensing state estimation, vortex-gust interactions  ·  EUROMECH colloquium",
         size=9.5, color=MUTE, first=True)


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


# ---- slide templates --------------------------------------------------------
def s_title():
    s = slide()
    rect(s, 0, 0, 0.30, H, fill=NAVY)
    tf = textbox(s, 0.95, 1.35, 11.7, 2.5)
    para(tf, "Sparse-sensing state estimation of vortex-gust airfoil interactions",
         size=31, color=NAVY, bold=True, space_after=6, first=True)
    para(tf, "Toward model-based flow control", size=21, color=ACCENT, bold=True, space_after=0)
    rect(s, 1.0, 3.95, 3.0, 0.06, fill=ACCENT)
    tf2 = textbox(s, 1.0, 4.2, 11.5, 1.5)
    para(tf2, "Forward physical closure of predictive vs reconstructive latents at Re = 5000",
         size=15, color=INK, first=True, space_after=12)
    para(tf2, "A. Solera-Rico, A. Miró, O. Lehmkuhl, C. Sanmiguel Vila",
         size=14, color=INK, space_after=3)
    # conference logo (top-right) and institution logo strip (bottom)
    logo(s, "euromech", 11.55, 0.42, 1.15, 0.72, crop=(0, 0, 460, 282))
    _ly = 5.35
    logo(s, "inta", 1.0, _ly, 0.66, 0.66, halign="left")
    logo(s, "uc3m", 1.95, _ly + 0.08, 1.55, 0.5, halign="left")
    logo(s, "bsc", 3.75, _ly, 0.74, 0.66, halign="left")
    logo(s, "upc", 4.72, _ly, 0.66, 0.66, halign="left")
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


# ---- native schematic helpers (large, editable; no tiny raster text) --------
DGRAY = RGBColor(0x8A, 0x93, 0xA3)
AE_RED = RGBColor(0xB0, 0x3A, 0x2E)
AE_FILL = RGBColor(0xF7, 0xEC, 0xEA)
JP_GRN = RGBColor(0x2E, 0x7D, 0x4F)
JP_FILL = RGBColor(0xEA, 0xF2, 0xEC)
BOX_FILL = RGBColor(0xEE, 0xF1, 0xF7)
COND_FILL = RGBColor(0xFD, 0xF0, 0xD9)


def dbox(s, l, t, w, h, text, fill=BOX_FILL, edge=NAVY, tcolor=INK, size=14):
    sp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    sp.shadow.inherit = False
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    sp.line.color.rgb = edge
    sp.line.width = Pt(1.5)
    tf = sp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Pt(3)
    tf.margin_right = Pt(3)
    tf.margin_top = Pt(1)
    tf.margin_bottom = Pt(1)
    for j, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        _runs(p, line, size, tcolor, base_bold=True)
    return sp


def ashape(s, kind, l, t, w, h, fill=DGRAY):
    sp = s.shapes.add_shape(kind, Inches(l), Inches(t), Inches(w), Inches(h))
    sp.shadow.inherit = False
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    sp.line.fill.background()
    return sp


def harrow(s, l, t, w, h=0.18, fill=DGRAY):
    return ashape(s, MSO_SHAPE.RIGHT_ARROW, l, t, w, h, fill)


def uparrow(s, l, t, w, h, fill=DGRAY):
    return ashape(s, MSO_SHAPE.UP_ARROW, l, t, w, h, fill)


def dlabel(s, l, t, w, text, size=12.5, color=MUTE, align=PP_ALIGN.LEFT, bold=False):
    tf = textbox(s, l, t, w, 0.5)
    para(tf, text, size=size, color=color, align=align, bold=bold, first=True)


def slide_jepa_concept():
    s = slide()
    header(s, "What is a JEPA?  (1 / 2)", "Method, primer")
    bx = [0.9, 3.15, 5.5, 7.45, 9.9]
    bw = [1.7, 1.7, 1.3, 1.8, 2.55]
    ax = [2.6, 4.85, 6.8, 9.35]
    hb = 0.72
    # Row A: autoencoder (reconstruct)
    dlabel(s, 0.9, 1.74, 11.8, "Autoencoder (most flow ROMs): reconstruct the field",
           size=15, color=AE_RED, bold=True)
    ya = 2.22
    ae = ["field ω(t)", "Encoder", "latent z", "Decoder", "reconstructed field"]
    for i, txt in enumerate(ae):
        dbox(s, bx[i], ya, bw[i], hb, txt, fill=AE_FILL, edge=AE_RED, size=13.5)
    for axx in ax:
        harrow(s, axx, ya + hb / 2 - 0.09, 0.5)
    dlabel(s, 0.9, 3.06, 11.8, "loss compares the reconstruction to the input in PIXEL space",
           size=13, color=AE_RED)
    # Row B: JEPA (predict)
    dlabel(s, 0.9, 4.04, 11.8, "JEPA: predict the next latent, no decoder",
           size=15, color=JP_GRN, bold=True)
    yb = 4.52
    jp = ["field ω(t)", "Encoder", "latent z(t)", "Predictor", "predicted z(t+1)"]
    for i, txt in enumerate(jp):
        dbox(s, bx[i], yb, bw[i], hb, txt, fill=JP_FILL, edge=JP_GRN, size=13.5)
    for axx in ax:
        harrow(s, axx, yb + hb / 2 - 0.09, 0.5)
    dlabel(s, 0.9, 5.36, 11.8,
           "loss compares predicted z(t+1) to z(t+1) = Encoder(next frame) in LATENT space; "
           "an anti-collapse term replaces the decoder", size=13, color=JP_GRN)
    tf = textbox(s, 0.9, 6.18, 11.8, 0.6)
    para(tf, "**Reconstruction ties the latent to pixels; prediction makes it forecastable**, "
             "which is what a controller needs.", size=15, color=NAVY, first=True)
    left_footer(s)
    notes(s, "Pedagogic core for a non-ML audience. Same first three boxes; the only change is "
             "Decoder -> Predictor and where the loss lives: pixels (AE) vs latent space (JEPA). "
             "No decoder in JEPA; the anti-collapse term stops the encoder collapsing.")
    return s


def slide_our_arch():
    s = slide()
    header(s, "Our architecture for gust encounters", "Method")
    ya = 1.95
    hb = 0.95
    bx = [0.6, 2.85, 5.15, 7.05, 9.35]
    bw = [2.0, 2.0, 1.6, 1.95, 2.0]
    arx = [2.6, 4.85, 6.65, 9.05]
    boxes = ["mid-plane\nvorticity ω(t)", "Encoder\n(CNN + ViT)\nunconditional",
             "latent z(t)\nd = 32 / 64", "Predictor\n(AR transformer)", "predicted\nz(t+1)"]
    fills = [BOX_FILL, BOX_FILL, JP_FILL, COND_FILL, JP_FILL]
    edges = [NAVY, NAVY, JP_GRN, ACCENT, JP_GRN]
    for i, txt in enumerate(boxes):
        dbox(s, bx[i], ya, bw[i], hb, txt, fill=fills[i], edge=edges[i], size=13)
    for axx in arx:
        harrow(s, axx, ya + hb / 2 - 0.09, 0.45)
    dbox(s, 7.05, 3.72, 1.95, 0.66, "gust c = (G, D, Y)", fill=COND_FILL, edge=ACCENT, size=13)
    uparrow(s, 7.85, 2.98, 0.35, 0.7, fill=ACCENT)
    dlabel(s, 9.15, 3.86, 4.0, "enters the predictor ONLY", size=12.5, color=ACCENT, bold=True)
    bullets(s, [b0("The **encoder is unconditional**: a pure state map ω(t) → z; the gust never touches it"),
                b0("The gust **c = (G, D, Y) enters the predictor only** (AdaLN); the predictor **is** the latent dynamics model you plan or learn against"),
                b0("A **visualisation decoder** is trained on the **frozen** encoder, never in the loss (figures only)"),
                b0("Compared at matched latent dimension against a reconstructive AE (Fukami) and POD")],
            0.7, 4.75, 12.0, 2.1, base=15)
    left_footer(s)
    notes(s, "Conditioning split is deliberate: encoder = pure state map; the gust enters the "
             "dynamics. For control, an actuation channel would enter where the gust does.")
    return s


# ---- build ------------------------------------------------------------------
s_title()

# I. Motivation & why JEPA could matter
s_fig_below(
    "The control problem", "Motivation",
    [b0("Small and micro air vehicles increasingly fly where the **gust speed is comparable to or above the flight speed** (urban canyons, building and ship wakes): gust ratio **G > 1** brings large, fast load transients classical models miss"),
     b0("Active control, model-based or RL, needs a **fast, faithful forward model** of the flow: a learned reduced dynamics it can trust under rollout"),
     b0("This talk: **what reduced state is worth planning against?**")],
    "graphical_abstract_talk.png",
    cap="Top: encode-then-decode reconstruction of the field 16 frames after impact (case G = -3, d = 64); the predictive latent keeps the wake. Bars: held-out test_b forecast closure (predictor rollout, H = 16).",
    ft=3.9, fh=2.95, base=16,
    note="Pitch to a control audience: the bottleneck for model-based control / RL in this "
         "regime is a reduced dynamics model that stays trustworthy under rollout.")

s_fig_right(
    "Why it is hard: it is the wake, not just the forces", "The physics",
    [b0("The load transient is built by the **leading-edge vortex (LEV)** and the wake it leaves; the discriminating information is in the **wake**, not the integrated forces C_L, C_D"),
     b0("**Gaussian scale split** (Motoori & Goto 2019): low-pass the vorticity at σ/c = 0.05 into a large-scale part u_L (the LEV and shear layer that carry the lift) and a fine part u_S, then track the **large-scale wake enstrophy** through the encounter to see which reduced state keeps that load-bearing structure"),
     b0("Top: large-scale vorticity at impact+16 for the **strongest test gust** (G = -3, D = 1.5, Y = -0.1): simulation vs the predictive and reconstructive **reconstructions** (defined shortly); predictive keeps the LEV/wake, reconstructive smooths it away"),
     b0("Bands = **mean ± 1 s.d. across the test_b encounters** (encounter spread, not a confidence interval)")],
    "scale_decomp.png",
    cap="Large-scale (σ/c = 0.05) wake vorticity at impact+16 (top) and large-scale wake enstrophy vs frame (bottom), by family.",
    note="Explain the scale split and why: isolate the load-bearing large-scale wake. Top triptych "
         "is the strongest test_b gust (G=-3, D=1.5, Y=-0.1); bands are mean +/- 1 s.d. over test_b "
         "encounters. These are reconstructions (encode-then-decode); the forecast comes later.",
    base=14)

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

slide_jepa_concept()

s_bullets(
    "What is a JEPA?  (2 / 2): the recipe and why for control", "Method, primer",
    [b0("**Training, in one line:** minimise the **latent** prediction error plus an **anti-collapse term** so the encoder cannot cheat by mapping everything to a constant (no decoder, no pixel loss)"),
     b0("So the encoder **keeps only what predicts the future** (the control-relevant dynamics), and the predictor **gets no pixel-level shortcuts**"),
     b0("**Cheap to roll out:** planning happens in the d = 32-64 latent, two to three orders of magnitude cheaper than evolving the full field"),
     b0("Lineage: I-JEPA / V-JEPA (image, video); from-pixels LeWM, LeJEPA, PLDM, so far only gridworld and toy visual tasks"),
     b0("**Why for control:** a learned world-model that is **observable from sensors**; to our knowledge the **first end-to-end JEPA on a parametric flow**")],
    note="Second JEPA slide: recipe (latent loss + anti-collapse, no decoder), cheap latent "
         "rollout, lineage, and the control framing.", base=17)

slide_our_arch()

# Illustrated alternatives (Fukami style); compare with the box diagrams on slides 6 & 8
s_fig_below(
    "Illustrated alternative: the autoencoder route", "Method, alternative",
    [b0("Reconstructive route (Fukami-Taira lineage): a **CNN encoder**, an **MLP latent ξ** (which also feeds a lift head to C_L), and a **CNN decoder**; the loss compares the reconstructed field to the input in **pixel space**")],
    "fukami_ae.png",
    cap="An autoencoder reconstructs the field; the latent is shaped by pixel reconstruction.",
    ft=2.7, fh=3.6, base=16,
    note="Illustrated alternative to the box diagram (slide 6); keep whichever style you prefer.")

s_fig_below(
    "Illustrated alternative: the JEPA route", "Method, alternative",
    [b0("Same encoder, but a **predictor** advances the latent and the loss lives in **latent space**, with **no decoder** (an anti-collapse term keeps z informative); the gust c = (G, D, Y) enters the predictor only")],
    "fukami_jepa.png",
    cap="JEPA predicts the next latent in latent space, with no field reconstruction.",
    ft=2.7, fh=3.7, base=16,
    note="Illustrated alternative to the box architecture (slide 8); keep whichever style you prefer.")

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
    [b0("Closure R² (rolled-out latent to observable, mean over six): **JEPA 0.84 · Fukami 0.43 · POD 0.56**"),
     b0("**Wake enstrophy** is the discriminator: **0.93** vs 0.28 / 0.37; held-out MAE **2.4 to 3× lower**. POD competes only on the impulse I_y"),
     b0("**Wake enstrophy** E_w: integral of ω_z² over the wake region (x/c in [0.5, 4], |y/c| ≤ 1)"),
     b0("**Impulse** I_y: integral of x·ω_z over the slice, a wake-transport diagnostic, not the impulse-theorem lift"),
     b0("Each panel's MAE is in that observable's own units; R² is the cross-observable comparison")],
    "closure_H16.png",
    cap="Held-out MAE at H = 16 by observable (test B top, test C bottom); each observable in its own units.",
    note="Headline. Wake enstrophy is where the predictive latent wins; forces are easy for "
         "every family. State the two unfamiliar observables and the per-observable-units point here.",
    fl=8.0, fw=4.9, base=16)

s_fig_right(
    "Forward closure at shorter horizons: H = 8", "Result",
    [b0("The same comparison, rolled out to **H = 8** instead of 16"),
     b0("The family ordering is unchanged: the predictive latent is lowest on the wake, near the DNS-oracle floor"),
     b0("Absolute errors are smaller at the shorter horizon: wake-enstrophy MAE ≈ **30** for JEPA, vs ≈ 41 at H = 16"),
     b0("Reconstructive and POD stay well above the floor on the wake")],
    "closure_H8.png",
    cap="Held-out MAE at H = 8 by observable (test B top, test C bottom).",
    note="Closure tightens at the shorter horizon, but the family separation persists.",
    fl=6.9, fw=6.1, base=16)

s_fig_right(
    "Forward closure at shorter horizons: H = 4", "Result",
    [b0("Rolled out to **H = 4**, four frames after impact"),
     b0("The predictive latent is still lowest on the wake enstrophy and closest to the oracle floor"),
     b0("Errors shrink further: wake-enstrophy MAE ≈ **27** for JEPA; the wake stays the discriminating observable"),
     b0("So the advantage holds at **every horizon we test**, not only H = 16")],
    "closure_H4.png",
    cap="Held-out MAE at H = 4 by observable (test B top, test C bottom).",
    note="Even four frames out, the family separation holds on the wake.",
    fl=6.9, fw=6.1, base=16)

_anim = slide()
header(_anim, "Forward closure in action: the rollout tracks the encounter", "Result")
bullets(_anim,
        [b0("The predictor **rolls the latent forward from impact**; each scalar is read off the rolled latent by a fixed linear probe"),
         b0("**Wake enstrophy** and **lift** track the DNS truth (black) through impact and the lift dip"),
         b0("Reconstruction (green) = encoded-latent **ceiling**; prediction (orange) = the **forecast**"),
         b0("A representative **low-error** encounter (G = -1.5), not the hardest case")],
        0.7, 2.05, 4.7, 4.6, base=15)
_anim.shapes.add_movie(str(FIG / "forecast_anim.mp4"), Inches(5.7), Inches(1.75),
                       Inches(6.9), Inches(4.98),
                       poster_frame_image=str(FIG / "forecast_poster.png"), mime_type="video/mp4")
left_footer(_anim)
notes(_anim, "Embedded MP4 (plays in PowerPoint; a still shows otherwise). Two observables "
             "(wake enstrophy and lift): original (DNS) vs encoded-latent reconstruction vs "
             "rolled-latent prediction, for a representative low-error gust encounter.")

_field = slide()
header(_field, "The same forecast in physical space", "Result, physical space")
_ftf = textbox(_field, 0.7, 1.5, 11.9, 0.5)
para(_ftf, "Decode the rolled latent every frame: the forecast as full vorticity fields, not just scalars.",
     size=17, color=INK, first=True)
_field.shapes.add_movie(str(FIG / "field_anim.mp4"), Inches(0.85), Inches(2.25),
                        Inches(11.6), Inches(3.06),
                        poster_frame_image=str(FIG / "field_poster.png"), mime_type="video/mp4")
caption(_field, "DNS truth vs reconstruction (encoded latent) vs prediction (rolled latent), G = -1.5 "
                "encounter. SSIM(prediction) = 0.77 at impact, 0.53 at H = 16. The latent keeps the "
                "leading-edge vortex and shear layer; fine-scale wake is not retained at d = 64.",
        0.85, 5.5, 11.6)
left_footer(_field)
notes(_field, "Physical-space companion to the scalar forecast: same encounter, decoded with the frozen "
              "LapFiLM visualisation decoder (never part of the JEPA loss). Reconstruction is the "
              "representational ceiling; prediction is the rolled latent decoded.")

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
    "Latent coordinates group by physical function", "Result",
    [b0("Profile each of the 64 predictive-latent coordinates by its |Spearman| correlation with nine descriptors (gust G, forces, wake enstrophy, circulations Γ±, wake thickness, centroid)"),
     b0("Clustering the profiles recovers **functional groups**: **~51 wake-vorticity** coordinates (G1, G2) and **11 gust-forcing** coordinates (G3); two near-silent"),
     b0("Alone, the wake groups forecast the future wake at **≈ 0.7**, the forcing group at **0.45** (all 64 together: 0.83), the collective wake code seen earlier"),
     b0("Read **descriptively**, not as a causal claim")],
    "coord_groups.png",
    cap="Per-coordinate |ρ| with each descriptor; rows grouped into functional clusters (G1-G4), labelled with coord count and held-out wake-forecast skill.",
    note="Backs the 'it is the wake' story: the latent organises coordinates by physical "
         "function. Descriptive correlation only; the causal (SURD) part of this analysis is "
         "not in the paper.",
    fl=6.6, fw=6.4, base=15)

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
     b0("**Result:** the reconstructive rollout drifts **~10× off-manifold (9.9)**; JEPA 0.85 and POD 0.81 stay inside"),
     b0("The R² in the plot is **forward closure via a probe**: the predictor advances only the **latent**, and a fixed linear probe reads each observable off the rolled-out latent (no direct predictor of C_L or the wake)")],
    "horizon_sweep.png",
    cap="Forward-closure R² vs rollout horizon (test_b): the predictor advances the latent; a linear probe reads each observable off it.",
    note="Non-standard metric, so define it. Low one-step error is worthless if the rollout walks "
         "off the data manifold where probes are invalid. R2 here is probe-on-rolled-latent, not a "
         "direct predictor of the observables.",
    base=14)

s_fig_right(
    "Diagnostic 2: topology of the encounter", "Mechanism",
    [b0("**Why:** shedding and gust encounters are **cyclic**; a faithful state should preserve that cyclic structure"),
     b0("**Persistent homology** (coordinate-free): counts loops (1-cycles) and how long they persist as a scale grows; **one long-lived cycle** = clean loop, many short-lived = fragmentation"),
     b0("**Result:** the predictive latent has a **single persistent cycle**, the reconstructive latent **fragments**. This is the *topological* statement, **not** that the PCA curve visually closes"),
     b0("Panels use **encoded latents + the decoder, no predictor**: in (PC1, PC2) the orbit **departs the baseline cycle and only partially returns** within the 120-frame window; the staged snapshots are reconstructions")],
    "cycle.png",
    cap="(a, b) PCA of the encoded-latent trajectory; (c) staged decodes (reconstructions) with OT distances. The single-cycle is established by persistent homology; the orbit need not close in the window.",
    note="Address the obvious question: you do NOT see a fully-closed loop in PCA (it departs and "
         "partially returns). The single-cycle is a persistent-homology statement. Snapshots are "
         "encode-then-decode reconstructions; no temporal predictor here.", base=14)

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

s_fig_below(
    "Decoded reconstructions", "Result, physical space",
    [b0("Decoded fields: predictive vs reconstructive vs POD vs DNS"),
     b0("The predictive decode is **blurrier pixel-wise** but preserves the **transported large-scale wake**; the reconstructive decode is sharp yet drifts under rollout")],
    "reconstructions.png", cap="Held-out reconstructions; the predictive objective trades pixel sharpness for transported structure.",
    ft=3.4, fh=3.2)

s_fig_below(
    "Sparse sensor placement", "Control relevance",
    [b0("Sparse wall-pressure sensors selected by **TCSI / qDEIM** vs uniform"),
     b0("Pressure → latent map: KernelRidge(RBF) on the K-sensor × pre-impact-window vector"),
     b0("K = 2 / 4 / 8 / 16; the LSTM/KRR comparison is 5-fold-CV selected to guard small-sample overfitting")],
    "sensor_placement.png", cap="Optimal sparse sensor placement on the airfoil surface.",
    ft=3.5, fh=3.1)

s_fig_right(
    "Flow recovered from sparse wall pressure", "Control relevance",
    [b0("Estimate the **impact-frame predictive latent from K wall-pressure taps** (TCSI placement, kernel ridge), then **decode** it to a field"),
     b0("**K = 8 taps** recover the leading-edge vortex and shear layer; **K = 2** coarsens but keeps the gross wake structure"),
     b0("Benchmarked against the **oracle decode** (decode of the simulation-encoded latent) and the simulation"),
     b0("So the predictive state is **reconstructible from a few wall sensors**, a deployment-relevant observability result")],
    "flow_recovery.png",
    cap="Flow recovered from sparse wall pressure: simulation, oracle decode, and decode of the latent estimated from K = 8 and K = 2 taps (held-out encounters).",
    note="Reframed to match the figure (flow from pressure). The predictive-vs-reconstructive wake "
         "tracking is already on slide 3 (scale decomposition).", base=16)

_sense = slide()
header(_sense, "Sparse-sensor state estimation in action", "Control relevance")
_stf = textbox(_sense, 0.7, 1.5, 11.9, 0.5)
para(_stf, "No predictor here: a per-frame map from 16 wall-pressure taps to the latent, decoded to a field.",
     size=17, color=INK, first=True)
_sense.shapes.add_movie(str(FIG / "sensing_anim.mp4"), Inches(1.75), Inches(2.15),
                        Inches(9.8), Inches(3.59),
                        poster_frame_image=str(FIG / "sensing_poster.png"), mime_type="video/mp4")
caption(_sense, "DNS truth (green dots mark the 16 TCSI taps) vs the field recovered from wall pressure "
                "alone, on a held-out encounter. Held-out-encounter latent R-squared = 0.72, SSIM ~ 0.5. "
                "Wall pressure fixes the near-body LEV and shear layer; the far wake is not "
                "surface-observable.",
        0.9, 5.85, 11.5)
left_footer(_sense)
notes(_sense, "Dynamic companion to the static flow-recovery slide: a causal 6-frame pressure window at "
              "16 TCSI taps -> MLP -> latent -> frozen decoder, applied frame by frame. This is state "
              "ESTIMATION, not forecasting; the JEPA predictor is not used here.")

# V. Control relevance
s_fig_right(
    "Control relevance: observable and forecastable", "Control relevance",
    [b0("The predictive state is **observable from sparse wall pressure**: (G, D) recoverable from the latent (R² ≈ 0.46 / 0.80 on test_b)"),
     b0("**Lead-time impact-C_L:** predictor-in-loop **R² = 0.35 at 10 frames pre-impact**, vs 0.13 from direct pressure sensing (oracle 0.68)"),
     b0("The reconstructive latent's oracle is **negative**, a representation failure, not a probe failure"),
     b0("A deployment-relevant **state estimate plus short-horizon forecast**: the ingredients a controller consumes")],
    "cl_inference_simple.png",
    cap="Impact-C_L inferred at each lead time before impact (test_b R²): oracle = probe on the DNS-encoded latent; direct = pressure sensors to C_L; predictor-in-loop = roll the latent to impact, then probe.",
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

# Acknowledgements
_ack = slide()
header(_ack, "Acknowledgements", "Thanks")
_atf = textbox(_ack, 0.7, 1.5, 11.9, 0.5)
para(_atf, "This work is a collaboration with:", size=18, color=INK, bold=True, first=True)

# author photo tiles
_people = [("photo_solera", "Alberto Solera-Rico", "INTA · UC3M", 0.50, 0.45),
           ("photo_miro", "Arnau Miró", "UPC · BSC", 0.50, 0.45),
           ("photo_lehmkuhl", "Oriol Lehmkuhl", "BSC", 0.33, 0.45)]
_cols = [2.55, 6.65, 10.75]
_ps = 1.7
for (_key, _name, _aff, _fx, _fy), _cx in zip(_people, _cols):
    _sp = square_photo(_key, focus_x=_fx, focus_y=_fy)
    if _sp:
        place_image(_ack, _sp, _cx - _ps / 2, 2.2, _ps, _ps)
    _ntf = textbox(_ack, _cx - 1.85, 4.0, 3.7, 0.75)
    para(_ntf, _name, size=15, color=NAVY, bold=True, align=PP_ALIGN.CENTER, first=True)
    para(_ntf, _aff, size=12, color=MUTE, align=PP_ALIGN.CENTER)

# funding (text left, BBVA + Red Leonardo logos right)
rect(_ack, 0.7, 5.05, 11.9, 1.75, fill=RGBColor(0xF3, 0xF5, 0xF9))
rect(_ack, 0.7, 5.05, 0.10, 1.75, fill=ACCENT)
_ftf = textbox(_ack, 1.05, 5.22, 8.2, 1.5)
para(_ftf, "Work produced with the support of a 2024 Leonardo Grant for Researchers and Cultural "
           "Creators, BBVA Foundation, project PREVENT (grant LEO24-2-15988). The Foundation takes "
           "no responsibility for the opinions, statements and contents of this project, which are "
           "entirely the responsibility of its authors.", size=13, color=INK, first=True)
logo(_ack, "bbva", 9.55, 5.3, 2.7, 0.62)
logo(_ack, "redleonardo", 9.55, 6.05, 2.7, 0.62)
left_footer(_ack)
notes(_ack, "Acknowledgements: author photos (Solera-Rico, Miró, Lehmkuhl) and the BBVA Foundation / "
            "Red Leonardo funding logos. Funding: 2024 Leonardo Grant, project PREVENT (LEO24-2-15988).")

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
    "Model detail: encoder and predictor", "Backup",
    [b0("**Encoder** = a CNN stem (3 down-sampling stages) + a 6-layer Vision Transformer; the field becomes 288 patch tokens, and a learned [CLS] token is mapped by a small MLP (with BatchNorm) to the latent z (d = 32 / 64)"),
     b0("**Predictor** = a 6-layer autoregressive transformer (hidden 384, 16 heads, dropout 0.1): it reads z(t) and emits the next latent z(t+1)"),
     b0("**Causal mask:** each step sees only past steps, never the future; **RoPE** encodes the frame ordering so the transformer knows what is earlier or later"),
     b0("**AdaLN-Zero:** how the gust (G, D, Y) and shedding phase φ enter, they rescale the transformer's internal normalisation, initialised to do nothing so they perturb training gently"),
     b0("**Training:** match the next latent (teacher forcing) + **scheduled sampling** (feed the model its own predictions over an 8-step rollout, so it tolerates its own errors) + **SIGReg** anti-collapse (keeps z informative without a decoder); AdamW, bf16, 80k steps")],
    "predictor_detail.png", fl=7.3, fw=5.6, base=12.5)

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

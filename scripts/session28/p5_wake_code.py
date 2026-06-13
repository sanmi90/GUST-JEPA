"""Physics Track P5: where the wake code lives (v2.1; extends D162/D163).

Master plan Phase C, Physics Track P5. Three honest re-measurements on the v2.1
UNCONDITIONED latents (jepa_tf_noc, fukami, pod at matched d=64), reusing the
committed machinery. D162/D163 are NOT assumed: every number is reported as
measured on v2.1; if a result weakens or reverses, the README says so.

PROBE REGIME DECLARATION (CLAUDE.md probe methodology): every probe here is a
STATE-DESCRIPTOR probe on PER-frame z, fitted on the family's TRAIN per-frame
latents over POST-IMPACT frames against the per-frame DNS wake observable, read
out at frame t + H (H = 16). No parameter (G, D, Y) probe lives here.

PART (i) -- COLLECTIVE CODE (committed Fig 15 + S5.1; regenerated, D163).
Per family, toward the future wake enstrophy at H = 16 on held-out test_b
(targets aligned by (case_id, encounter) from the v2.1 per_frame_targets):
combination skill (ridge over all coordinates, standardised on train), best
single-coordinate skill, the GAP (= how distributed/collective the forecast code
is), forecast-beyond-forces partial correlation, and #coordinates with single
skill > 0.5. The gap is the headline "collective vs redundant code" number.
Plus FUNCTIONAL GROUPING (D163: coordinates group into ~2 physical functions):
the wake-active coordinates are clustered by the sign+magnitude structure of
their per-coordinate wake/force loadings, and any subspace overlap between the
groups is reported AGAINST the random K/d null (CLAUDE.md: random mean cos^2 for
two K-dim subspaces in d-dim ambient is K/d, ~0.05 for K=3, d=64; a pairwise
cos^2 within ~0.01 of K/d is NOT overlap).

PART (ii) -- ENERGY-INFORMATION CURVE (new). Per family, held-out wake-forecast
skill (R^2, ridge on train, evaluated on test_b) vs the number of leading PCs
of the latent retained (PCA fit on train post-impact latents), and -- for POD --
vs the retained ENERGY fraction. This resolves the participation-ratio-vs-
distributed-code tension quantitatively: how many latent PCs does each family
need to carry the wake? The KNEE (the smallest #PCs reaching 90% of the
asymptotic skill) is reported per family.

PART (iii) -- FOOTPRINT (new; supersedes D163's qualitative decode). The
wake-forecast DIRECTION in latent space (the standardised ridge weight vector u
of the H = 16 wake probe) is back-propagated through the FROZEN jepa encoder:
saliency = | d (z . u) / d (input omega) |, averaged over held-out impact-frame
inputs. The mean saliency map is OVERLAP-SCORED (|saliency|^2-energy-weighted
mean of the structure indicator) against (a) the thresholded |omega| mask and
(b) the Q-criterion vortex mask, both at the impact frame. Q is computed from
the RAW /u velocity (PREVENT_ROOT) when available (the true 2D Q-criterion,
reusing the _oneoff_pod_q_overlap / exp4_structures_shap machinery); if /u is
missing the script documents the omega-magnitude fallback. The overlap is
compared to a PERMUTATION CHANCE BASELINE (the mask shuffled spatially): if the
overlap lands inside the null band the footprint is reported as NOT
structure-localised.

GPU: the saliency autograd uses require_rtx6000(gpu_index=1) by default (the
L40S cards are FORBIDDEN). --device cpu forces a small CPU pass (stated in the
README). Parts (i) and (ii) are CPU numpy/sklearn.

STATS: case-clustered CIs via stats_lib; for the paired family comparison the
sign test + case-clustered CI (NOT case_permutation_p; B6 degeneracy). The
energy-information knee and the saliency-overlap-vs-chance are trend / overlap
measures, not paired location tests.

Output (all absolute under the repo):
  outputs/session28/p5_code/p5_results.npz   per-family arrays
  outputs/session28/p5_code/p5_results.json  all numbers + config
  outputs/session28/p5_code/fig_wake_code.{pdf,png}  3 panels
  outputs/session28/p5_code/README.md        the three results AS MEASURED
  outputs/session28/numbers_parts/p5_code.json  eval_all macros
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import stats_lib  # noqa: E402

# ---- paths -------------------------------------------------------------------
LATENTS_ROOT = REPO / "outputs" / "session28" / "latents"
PF_ROOT = REPO / "outputs" / "session28" / "exp2" / "per_frame_targets"
SPLIT_PATH = REPO / "configs" / "splits" / "split_v2p1.json"
PIPELINE_MANIFEST = REPO / "outputs" / "data_pipeline" / "v2p1" / "manifest.json"
JEPA_CKPT = (
    REPO
    / "outputs"
    / "runs"
    / "session28"
    / "jepa_tf_noc_d64_s42"
    / "encoder"
    / "checkpoint_iter020000.pt"
)
OUT_DIR = REPO / "outputs" / "session28" / "p5_code"
NUMBERS_PART = REPO / "outputs" / "session28" / "numbers_parts" / "p5_code.json"

CACHE_ROOT = (
    Path(
        os.environ.get(
            "VORTEX_JEPA_CACHE",
            str(
                Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
                / "data"
                / "processed"
                / "vortex-jepa"
            ),
        )
    )
    / "v2p1"
)
PREVENT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))

# ---- constants ---------------------------------------------------------------
H = 16  # forecast horizon (primary; matches the closure protocol)
WAKE_OBS = "wake_enstrophy"
STRONG_THR = 0.5  # |single-coordinate Spearman| above which a coordinate is "strong"
KNEE_FRAC = 0.90  # knee = smallest #PCs reaching this fraction of asymptotic skill
N_GROUP = 2  # D163: coordinates group into ~2 physical functions
HW = (192, 96)
MID_Z = 16
DX = 6.0 / 192
DY = 3.0 / 96
TE_X_INDEX = 144  # x/c < 1 boundary in pixels (LE at px 48; matches exp_lev/q machinery)

# (latents tag, canonical family label, figstyle family tag)
FAMILIES = [
    ("jepa_tf_noc_d64_s42", "jepa_tf_noc", "jepa"),
    ("fukami_d64_s42", "fukami", "fukami"),
    ("pod_d64", "pod", "pod"),
]


# =============================================================================
# Loading.
# =============================================================================
def targets_map(split: str) -> dict:
    """(case_id, encounter_index) -> per-frame DNS dict for one split.

    Reads the v2.1 per_frame_targets (DNS-derived, encoder-independent), giving
    wake_enstrophy(t), C_L(t), C_D(t), G(t), and the impact frame.
    """
    d = np.load(PF_ROOT / f"{split}.npz", allow_pickle=True)
    cid = np.asarray(d["case_id"])
    enc = np.asarray(d["encounter_index"], int)
    out = {}
    for i in range(len(cid)):
        out[(str(cid[i]), int(enc[i]))] = {
            "wake": np.asarray(d[WAKE_OBS][i], float),
            "C_L": np.asarray(d["C_L"][i], float),
            "C_D": np.asarray(d["C_D"][i], float),
            "G": np.asarray(d["G"][i], float),
            "impact": int(d["impact_frame"][i]),
        }
    return out


def load_family_latents(tag: str) -> dict | None:
    """Load a family's per-frame latents for train + test_b (+ test_c)."""
    base = LATENTS_ROOT / tag
    if not (base / "train.npz").exists():
        return None
    out: dict = {}
    for split in ("train", "test_b", "test_c"):
        p = base / f"{split}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        cids = d["case_ids"] if "case_ids" in d.files else d["case_id"]
        encs = d["encounter_indices"] if "encounter_indices" in d.files else d["encounter_index"]
        out[split] = {
            "z_full": np.asarray(d["z_full"], float),
            "case_ids": np.asarray([str(c) for c in cids]),
            "enc": np.asarray(encs, int),
            "impact": np.asarray(d["impact_frame"], int),
        }
    return out


def build_post_impact(
    fam_split: dict, tmap: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Post-impact (latent, future-wake, forces, case_id) rows for one split.

    For every encounter and every post-impact frame tt in [impact, T - H), emit
    the latent z[tt] and the future wake enstrophy wake[tt + H]. case_id is
    carried for case-clustered CIs. Force descriptors F = (G, C_L, C_D) at tt are
    the partial-correlation control (forecast-beyond-forces).
    """
    z = fam_split["z_full"]
    cids = fam_split["case_ids"]
    encs = fam_split["enc"]
    imps = fam_split["impact"]
    Tn = z.shape[1]
    X, fw, F, cc = [], [], [], []
    for i in range(len(cids)):
        key = (str(cids[i]), int(encs[i]))
        if key not in tmap:
            continue
        t = tmap[key]
        imp = int(imps[i])
        for tt in range(imp, Tn - H):
            X.append(z[i, tt, :])
            fw.append(t["wake"][tt + H])
            F.append([t["G"][tt], t["C_L"][tt], t["C_D"][tt]])
            cc.append(str(cids[i]))
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(fw, dtype=np.float64),
        np.asarray(F, dtype=np.float64),
        np.asarray(cc),
    )


# =============================================================================
# PART (i): collective code.
# =============================================================================
def _partial_corr(a: np.ndarray, b: np.ndarray, ctrl: np.ndarray) -> float:
    """|rank-partial correlation| of a vs b controlling for ctrl (verbatim port)."""
    ra, rb = rankdata(a), rankdata(b)
    rc = np.column_stack([np.ones(len(a))] + [rankdata(ctrl[:, j]) for j in range(ctrl.shape[1])])

    def res(r):
        return r - rc @ np.linalg.lstsq(rc, r, rcond=None)[0]

    return abs(float(np.corrcoef(res(ra), res(rb))[0, 1]))


def collective_code_skill(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    Fte: np.ndarray | None = None,
) -> dict:
    """Combination vs best-single-coordinate Spearman skill for the wake probe.

    Standardise X per coordinate (fit on train) before ridge so regularisation
    is comparable across families. The combination read-out is the held-out
    projection onto the unit ridge direction; skills are |Spearman| (scale-free).
    The gap = combo - best_single is the collective-code headline.
    """
    sc = StandardScaler().fit(Xtr)
    ytr_r = rankdata(ytr)
    ytr_r = (ytr_r - ytr_r.mean()) / ytr_r.std()
    u = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr_r).coef_
    s = sc.transform(Xte) @ (u / max(np.linalg.norm(u), 1e-12))
    single = np.array([abs(spearmanr(Xte[:, k], yte).statistic) for k in range(Xte.shape[1])])
    combo = abs(spearmanr(s, yte).statistic)
    out = {
        "combo": float(combo),
        "best_single": float(single.max()),
        "gap": float(combo - single.max()),
        "n_strong": int((single > STRONG_THR).sum()),
        "single_skills": single,
        "ridge_dir": u,
    }
    if Fte is not None:
        out["beyond_forces"] = _partial_corr(s, yte, Fte)
    return out


def functional_grouping(
    Xtr: np.ndarray, ytr: np.ndarray, single_skills: np.ndarray, n_group: int = N_GROUP
) -> dict:
    """Group the wake-informative latent coordinates into ~n_group functions (D163).

    The per-coordinate loading profile is its (signed) rank correlation with the
    wake target plus its scale -- a small functional fingerprint. Because a
    COLLECTIVE wake code has no single coordinate above STRONG_THR (that is the
    whole point of part (i)), the "active" set is taken RELATIVELY: the
    wake-informative coordinates are those whose single skill is in the top
    half AND above max(0.15, half the best single skill). They are clustered by
    the fingerprint (KMeans, n_group clusters); for each cluster the leading data
    PC is embedded into the d-dim ambient and the pairwise subspace overlap
    (cos^2) is reported AGAINST the random K/d null. A cos^2 within 0.01 of K/d
    is NOT overlap.
    """
    from sklearn.cluster import KMeans

    d = Xtr.shape[1]
    best = float(single_skills.max())
    rel_thr = max(0.15, 0.5 * best)
    active = np.where(single_skills >= rel_thr)[0]
    if active.size < 2 * n_group:
        # Too few wake-informative coordinates to define n_group subspaces.
        return {
            "n_active": int(active.size),
            "active_coords": active.tolist(),
            "grouped": False,
            "activity_threshold": rel_thr,
            "best_single": best,
            "reason": (
                f"fewer than 2*n_group coordinates above the relative activity "
                f"threshold {rel_thr:.3f} (best single {best:.3f}); no grouping"
            ),
        }
    # functional fingerprint per active coordinate: signed Spearman with wake + a
    # proxy for the within-latent correlation neighbourhood (sign of loading on
    # the top wake-PC direction). Keep it simple and interpretable.
    fp = []
    yr = rankdata(ytr)
    for k in active:
        xr = rankdata(Xtr[:, k])
        wake_r = float(np.corrcoef(xr, yr)[0, 1])
        fp.append([wake_r, float(np.sign(wake_r)), float(np.std(Xtr[:, k]))])
    fp = np.asarray(fp)
    fp = (fp - fp.mean(0)) / (fp.std(0) + 1e-12)
    km = KMeans(n_clusters=n_group, n_init=10, random_state=0).fit(fp)
    labels = km.labels_
    groups = [active[labels == g].tolist() for g in range(n_group)]

    # Subspace per group: the standard-basis subspace spanned by its coordinates.
    # Pairwise overlap = mean squared cosine of principal angles between the
    # group-1 and group-2 axis-aligned subspaces. For disjoint coordinate sets
    # the axis-aligned overlap is 0 by construction; the meaningful overlap test
    # is on the DATA-defined directions (each group's leading PC in z-space).
    bases = []
    for g in groups:
        if len(g) == 0:
            bases.append(None)
            continue
        Zg = Xtr[:, g]
        Zg = Zg - Zg.mean(0)
        # leading PC of the group block, embedded back into the d-dim ambient
        try:
            _, _, vt = np.linalg.svd(Zg, full_matrices=False)
            v = np.zeros(d)
            v[g] = vt[0]
            v = v / max(np.linalg.norm(v), 1e-12)
            bases.append(v)
        except np.linalg.LinAlgError:
            bases.append(None)
    cos2 = float("nan")
    k_dim = 1  # leading-PC direction per group is 1-dimensional
    if bases[0] is not None and bases[1] is not None:
        cos2 = float(np.dot(bases[0], bases[1]) ** 2)
    null = k_dim / d
    return {
        "n_active": int(active.size),
        "active_coords": active.tolist(),
        "grouped": True,
        "activity_threshold": rel_thr,
        "best_single": best,
        "n_group": n_group,
        "group_sizes": [len(g) for g in groups],
        "groups": groups,
        "pairwise_cos2_leadPC": cos2,
        "subspace_null_K_over_d": null,
        "K": k_dim,
        "d": d,
        "overlap_above_null": is_overlap_above_null(cos2, k_dim, d),
        "note": (
            "groups are disjoint coordinate sets, so axis-aligned overlap is 0 by "
            "construction; the reported cos^2 is between each group's leading "
            "data PC embedded in the d-dim ambient, scored against the K/d null."
        ),
    }


def is_overlap_above_null(cos2: float, K: int, d: int, tol: float = 0.01) -> bool:
    """True iff cos^2 exceeds the random K/d null by more than tol (CLAUDE.md)."""
    if not np.isfinite(cos2):
        return False
    return bool(cos2 > (K / d) + tol)


def random_subspace_overlap_null(
    K: int, d: int, n_draws: int = 400, rng: np.random.Generator | None = None
) -> dict:
    """Empirical mean cos^2 between two random K-subspaces in d-dim (~K/d)."""
    if rng is None:
        rng = np.random.default_rng(0)
    vals = []
    for _ in range(n_draws):
        a = np.linalg.qr(rng.standard_normal((d, K)))[0]
        b = np.linalg.qr(rng.standard_normal((d, K)))[0]
        s = np.linalg.svd(a.T @ b, compute_uv=False)
        vals.append(float(np.mean(s**2)))
    return {
        "mean_cos2": float(np.mean(vals)),
        "std_cos2": float(np.std(vals)),
        "expected": K / d,
        "K": K,
        "d": d,
        "n_draws": n_draws,
    }


# =============================================================================
# PART (ii): energy-information curve.
# =============================================================================
def _r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    sse = float(((y_pred - y_true) ** 2).sum())
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-12)


def energy_information_curve(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    ks: list[int],
    energy_axis: bool = False,
) -> dict:
    """Held-out wake-forecast R^2 vs #leading PCs retained (PCA fit on train).

    A ridge probe is fitted on the train PC scores (top-k) and evaluated on the
    test PC scores; R^2 is the protocol held-out R^2 (SST about the test mean).
    The knee is the smallest k reaching KNEE_FRAC of the asymptotic (max-k) skill.
    With energy_axis=True the retained PCA energy fraction is also returned
    (the POD-style energy curve).
    """
    ks = [k for k in ks if k <= Xtr.shape[1]]
    pca = PCA(n_components=max(ks)).fit(Xtr)
    Str = pca.transform(Xtr)
    Ste = pca.transform(Xte)
    ev = pca.explained_variance_ratio_
    skill: dict[int, float] = {}
    energy: dict[int, float] = {}
    for k in ks:
        scaler = StandardScaler().fit(Str[:, :k])
        m = Ridge(alpha=1.0).fit(scaler.transform(Str[:, :k]), ytr)
        yhat = m.predict(scaler.transform(Ste[:, :k]))
        skill[k] = float(_r2(yhat, yte))
        energy[k] = float(ev[:k].sum())
    asym = max(skill.values())
    target = KNEE_FRAC * asym if asym > 0 else float("inf")
    knee_k = max(ks)
    for k in sorted(ks):
        if skill[k] >= target:
            knee_k = k
            break
    out = {"ks": ks, "skill": skill, "knee_k": int(knee_k), "asymptotic_skill": float(asym)}
    if energy_axis:
        out["energy_fraction"] = energy
        out["knee_energy_fraction"] = float(energy[knee_k])
    return out


# =============================================================================
# PART (iii): footprint saliency.
# =============================================================================
def load_pipeline():
    from src.data.omega_pipeline import OmegaPipeline

    return OmegaPipeline.from_manifest(PIPELINE_MANIFEST)


def load_frozen_jepa_encoder(device):
    """Reconstruct the frozen HybridCNNViTEncoder from the s42 checkpoint."""
    import torch

    from src.models.encoder import HybridCNNViTEncoder

    blob = torch.load(JEPA_CKPT, map_location="cpu", weights_only=False)
    targs = blob.get("args", {})
    d = int(targs.get("d", 64))
    proj_norm = str(targs.get("projection_norm", "batchnorm"))
    enc = HybridCNNViTEncoder(latent_dim=d, projection_norm=proj_norm).to(device)
    state = blob.get("jepa_state_dict", blob.get("encoder_state_dict", blob))
    enc_state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
    if not enc_state and "encoder_state_dict" in blob:
        enc_state = blob["encoder_state_dict"]
    enc.load_state_dict(enc_state)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, d


def heldout_impact_inputs(splits: tuple[str, ...]) -> list[dict]:
    """Resolve held-out impact-frame cache inputs (case_id, enc, raw frame)."""
    import h5py

    split_def = json.loads(SPLIT_PATH.read_text())
    rows: list[dict] = []
    case_split = {}
    for s in splits:
        key = "test_b_cases" if s == "test_b" else "test_c_cases"
        for cid in split_def.get(key, []):
            case_split[cid] = s
    for cid, s in case_split.items():
        cdir = CACHE_ROOT / cid
        for ef in sorted(cdir.glob("encounter_*.h5")):
            enc = int(ef.stem.split("_")[1])
            with h5py.File(ef, "r") as f:
                imp = int(f.attrs.get("impact_frame_estimate", 40))
                fstart = int(f.attrs.get("frame_start", 0))
                raw_rel = f.attrs.get("raw_relative_path", "")
            rows.append(
                {
                    "case_id": cid,
                    "enc": enc,
                    "split": s,
                    "impact_local": imp,
                    "frame_start": fstart,
                    "raw_rel": str(raw_rel),
                    "cache_path": str(ef),
                }
            )
    return rows


def compute_saliency_map(
    enc, pipeline, device, rows: list[dict], u_dir: np.ndarray
) -> tuple[np.ndarray, int]:
    """Mean |d (z . u) / d (input omega)| over held-out impact frames.

    For each held-out encounter, normalise the impact-frame omega through the
    pipeline, run a single frame through the frozen encoder, project the latent
    onto the wake-forecast direction u, backprop to the input, and accumulate the
    absolute gradient. Autocast is OFF here (bf16 gradients are too coarse for a
    saliency map); the single-frame forward in fp32 is cheap.
    """
    import h5py
    import torch

    u_t = torch.tensor(
        u_dir / max(np.linalg.norm(u_dir), 1e-12), dtype=torch.float32, device=device
    )
    acc = np.zeros(HW, dtype=np.float64)
    n = 0
    for r in rows:
        with h5py.File(r["cache_path"], "r") as f:
            omega = np.asarray(f["omega_z"][r["impact_local"]], dtype=np.float32)  # (192, 96)
        om = pipeline.preprocess_raw(omega[None], r["case_id"], r["enc"])  # (1, 192, 96)
        om_t = torch.from_numpy(np.ascontiguousarray(om)).to(device)
        om_t = pipeline.normalize(om_t)
        # encoder wants (B, T, C, H, W); single frame.
        x = om_t.view(1, 1, 1, HW[0], HW[1]).clone().requires_grad_(True)
        z = enc(x)  # (1, 1, d)
        proj = (z.view(-1) * u_t).sum()
        enc.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        proj.backward()
        g = x.grad.detach().view(HW).abs().cpu().numpy().astype(np.float64)
        acc += g
        n += 1
    return (acc / max(n, 1)), n


def compute_q_at_frame(raw_path: Path, global_frame: int) -> np.ndarray | None:
    """True 2D Q-criterion on the mid-plane from raw /u (reuses the D126 form)."""
    import h5py

    if not raw_path.exists():
        return None
    with h5py.File(raw_path, "r") as f:
        if "u" not in f:
            return None
        u_mid = np.asarray(f["u"][global_frame, :, :, MID_Z, :], dtype=np.float32)
    u_x = np.nan_to_num(u_mid[..., 0], nan=0.0)
    u_y = np.nan_to_num(u_mid[..., 1], nan=0.0)
    du_dx, du_dy = np.gradient(u_x, DX, DY)
    dv_dx, dv_dy = np.gradient(u_y, DX, DY)
    s_xx, s_yy = du_dx, dv_dy
    s_xy = 0.5 * (du_dy + dv_dx)
    omega_xy = 0.5 * (dv_dx - du_dy)
    s_sq = s_xx**2 + s_yy**2 + 2.0 * s_xy**2
    omega_sq = 2.0 * omega_xy**2
    return (0.5 * (omega_sq - s_sq)).astype(np.float32)


def structure_masks(rows: list[dict], pipeline) -> tuple[np.ndarray, np.ndarray, bool, dict]:
    """Mean |omega| mask and mean Q>tau mask over held-out impact frames.

    Returns (omega_mask_mean, q_mask_mean, q_from_raw, info). Both masks are in
    [0, 1] (fraction of encounters whose pixel is in the structure). The omega
    mask thresholds |omega_norm| at its per-encounter p90; the Q mask thresholds
    Q at its per-encounter p90 of positive Q (vortex-dominated regions). The
    airfoil-adjacent cells are zeroed.
    """
    import h5py

    spatial_mask = pipeline.mask.cpu().numpy().astype(bool)  # True = zero (airfoil)
    divisor = 3.0 * float(pipeline.train_stats.std)
    om_acc = np.zeros(HW, dtype=np.float64)
    q_acc = np.zeros(HW, dtype=np.float64)
    n_om = 0
    n_q = 0
    q_from_raw = False
    for r in rows:
        with h5py.File(r["cache_path"], "r") as f:
            omega = np.asarray(f["omega_z"][r["impact_local"]], dtype=np.float32)
        om = pipeline.preprocess_raw(omega[None], r["case_id"], r["enc"])[0] / divisor
        aom = np.abs(om)
        aom[spatial_mask] = 0.0
        thr = np.percentile(aom[aom > 0], 90) if np.any(aom > 0) else 0.0
        om_acc += (aom > thr).astype(np.float64)
        n_om += 1
        # Q from raw /u
        raw_path = PREVENT / r["raw_rel"] if r["raw_rel"] else None
        if raw_path is not None:
            gframe = r["frame_start"] + r["impact_local"]
            q = compute_q_at_frame(raw_path, gframe)
            if q is not None:
                q_from_raw = True
                q[spatial_mask] = 0.0
                qpos = np.clip(q, 0.0, None)
                qthr = np.percentile(qpos[qpos > 0], 90) if np.any(qpos > 0) else 0.0
                q_acc += (qpos > qthr).astype(np.float64)
                n_q += 1
    om_mask = om_acc / max(n_om, 1)
    if n_q > 0:
        q_mask = q_acc / n_q
    else:
        # fallback: use the omega-magnitude mask as a Q proxy and DOCUMENT it.
        q_mask = om_mask.copy()
    info = {
        "n_omega": n_om,
        "n_q": n_q,
        "q_from_raw_u": bool(q_from_raw),
        "omega_thresh_pct": 90,
        "q_thresh_pct": 90,
        "q_fallback_used": bool(n_q == 0),
    }
    return om_mask, q_mask, q_from_raw, info


def saliency_mask_overlap(saliency: np.ndarray, mask: np.ndarray) -> float:
    """|saliency|^2-energy-weighted mean of the structure indicator (D126 form)."""
    energy = saliency.astype(np.float64) ** 2
    tot = energy.sum()
    if tot <= 0:
        return float("nan")
    energy = energy / tot
    return float((energy * mask.astype(np.float64)).sum())


def overlap_vs_chance(
    saliency: np.ndarray,
    mask: np.ndarray,
    n_perm: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Saliency-mask overlap vs a spatial-permutation chance baseline.

    The mask is shuffled spatially (its values permuted across pixels, preserving
    the structure area fraction) n_perm times; the null overlap distribution is
    formed against the FIXED saliency map. localized = overlap exceeds the 97.5th
    percentile of the null (one-sided). z = (overlap - null_mean) / null_sd.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    obs = saliency_mask_overlap(saliency, mask)
    flat_mask = mask.astype(np.float64).ravel()
    energy = (saliency.astype(np.float64) ** 2).ravel()
    tot = energy.sum()
    energy = energy / max(tot, 1e-12)
    null = np.empty(n_perm)
    for b in range(n_perm):
        null[b] = float((energy * rng.permutation(flat_mask)).sum())
    lo, hi = float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))
    mu, sd = float(null.mean()), float(null.std())
    z = (obs - mu) / sd if sd > 0 else float("nan")
    return {
        "overlap": float(obs),
        "chance_mean": mu,
        "chance_lo": lo,
        "chance_hi": hi,
        "chance_sd": sd,
        "z": float(z),
        "localized": bool(np.isfinite(obs) and obs > hi),
        "n_perm": n_perm,
    }


# =============================================================================
# Driver helpers.
# =============================================================================
def run_collective(fams: list[tuple[str, str, str]], tmap_tr: dict, tmap_te: dict) -> dict:
    """Part (i) over all available families + a seed-robustness check on jepa."""
    out: dict = {"per_family": {}}
    for tag, fam, _figtag in fams:
        blob = load_family_latents(tag)
        if blob is None or "test_b" not in blob:
            continue
        Xtr, ytr, _, _ = build_post_impact(blob["train"], tmap_tr)
        Xte, yte, Fte, _ = build_post_impact(blob["test_b"], tmap_te)
        res = collective_code_skill(Xtr, ytr, Xte, yte, Fte)
        grp = functional_grouping(Xtr, ytr, res["single_skills"])
        out["per_family"][fam] = {
            "combo": res["combo"],
            "best_single": res["best_single"],
            "gap": res["gap"],
            "n_strong": res["n_strong"],
            "beyond_forces": res.get("beyond_forces", float("nan")),
            "grouping": grp,
            "n_rows_train": int(Xtr.shape[0]),
            "n_rows_test_b": int(Xte.shape[0]),
        }
    # jepa seed robustness on the gap (s42, s0, s1, s2 where present)
    gaps = []
    for s in (42, 0, 1, 2):
        blob = load_family_latents(f"jepa_tf_noc_d64_s{s}")
        if blob is None or "test_b" not in blob:
            continue
        Xtr, ytr, _, _ = build_post_impact(blob["train"], tmap_tr)
        Xte, yte, _, _ = build_post_impact(blob["test_b"], tmap_te)
        gaps.append(collective_code_skill(Xtr, ytr, Xte, yte)["gap"])
    if gaps:
        out["jepa_seed_gap_mean"] = float(np.mean(gaps))
        out["jepa_seed_gap_sd"] = float(np.std(gaps))
        out["jepa_seed_n"] = len(gaps)

    # Paired family comparison (jepa vs fukami): per held-out encounter, the
    # combo-readout absolute prediction error; sign test that jepa errs less +
    # case-clustered CI on the per-encounter error difference (NOT
    # case_permutation_p; the B6 paired-location degeneracy applies here).
    out["paired_jepa_vs_fukami"] = _paired_family_skill(
        "jepa_tf_noc_d64_s42", "fukami_d64_s42", tmap_tr, tmap_te
    )
    return out


def _per_encounter_readout_error(
    tag: str, tmap_tr: dict, tmap_te: dict
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-row STANDARDISED |combo-readout - true future wake|, with case ids.

    A ridge probe is fitted directly on the standardised train latents toward
    the train-standardised wake target (zero-mean, unit-variance on train), so
    the readout error is in train-sigma units and comparable across families
    without an unstable affine rescaling of a rank direction. The comparison
    unit is the (frame-level) row; the case-clustered CI groups rows by case.
    """
    blob = load_family_latents(tag)
    if blob is None or "test_b" not in blob:
        return None
    Xtr, ytr, _, _ = build_post_impact(blob["train"], tmap_tr)
    Xte, yte, _, cc = build_post_impact(blob["test_b"], tmap_te)
    sc = StandardScaler().fit(Xtr)
    y_mu, y_sd = float(ytr.mean()), float(ytr.std() + 1e-12)
    m = Ridge(alpha=1.0).fit(sc.transform(Xtr), (ytr - y_mu) / y_sd)
    pred = m.predict(sc.transform(Xte))
    abserr = np.abs(pred - (yte - y_mu) / y_sd)  # train-sigma units
    return abserr, np.asarray(cc)


def _paired_family_skill(tag_a: str, tag_b: str, tmap_tr: dict, tmap_te: dict) -> dict:
    """Sign test + case-clustered CI that family A forecasts the wake better than B.

    Operates on the shared held-out (frame-level) rows: A's absolute readout
    error vs B's. delta = err_b - err_a (>0 means A better). Case-clustered
    bootstrap CI over the test_b cases; one-sided sign test that A errs less.
    """
    ea = _per_encounter_readout_error(tag_a, tmap_tr, tmap_te)
    eb = _per_encounter_readout_error(tag_b, tmap_tr, tmap_te)
    if ea is None or eb is None:
        return {"n": 0, "reason": "a family latent missing"}
    err_a, cc_a = ea
    err_b, cc_b = eb
    # the two families share the same test_b rows in the same order (same join);
    # guard the alignment by case-id sequence equality.
    if err_a.shape != err_b.shape or not np.array_equal(cc_a, cc_b):
        return {"n": 0, "reason": "row misalignment between families"}
    delta = err_b - err_a  # >0 => A (jepa) better
    k, n_eff, p = stats_lib.sign_test_one_sided(err_a, err_b)
    rng = np.random.default_rng(2026613)
    cci = stats_lib.case_cluster_bootstrap(delta, cc_a, n_boot=10000, rng=rng)
    return {
        "fam_a": tag_a,
        "fam_b": tag_b,
        "n": int(err_a.size),
        "n_cases": cci["n_cases"],
        "a_better_count": int(k),
        "n_effective": int(n_eff),
        "sign_p_one_sided": float(p),
        "mean_delta_abserr": float(delta.mean()),
        "case_cluster_enc_mean_ci": cci["enc_mean_ci"],
        "case_cluster_case_mean_ci": cci["case_mean_ci"],
        "note": (
            "delta = err_fukami - err_jepa (>0 => jepa forecasts the wake "
            "better); sign test + case-clustered CI, NOT case_permutation_p (B6)."
        ),
    }


def run_energy_info(fams: list[tuple[str, str, str]], tmap_tr: dict, tmap_te: dict) -> dict:
    """Part (ii) over all families."""
    ks = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    out: dict = {"ks": ks, "per_family": {}}
    for tag, fam, _figtag in fams:
        blob = load_family_latents(tag)
        if blob is None or "test_b" not in blob:
            continue
        Xtr, ytr, _, _ = build_post_impact(blob["train"], tmap_tr)
        Xte, yte, _, _ = build_post_impact(blob["test_b"], tmap_te)
        energy_axis = fam == "pod"
        curve = energy_information_curve(Xtr, ytr, Xte, yte, ks, energy_axis=energy_axis)
        out["per_family"][fam] = curve
    return out


def run_footprint(splits: tuple[str, ...], u_dir: np.ndarray, device_str: str | None) -> dict:
    """Part (iii): frozen-encoder saliency overlap vs structure masks + chance."""
    import torch

    pipeline = load_pipeline()
    if device_str == "cpu":
        device = torch.device("cpu")
        device_name = "cpu"
    else:
        from src.utils.device import require_rtx6000

        device = require_rtx6000(gpu_index=1)
        device_name = torch.cuda.get_device_name(device.index)
    enc, _d = load_frozen_jepa_encoder(device)
    rows = heldout_impact_inputs(splits)
    saliency, n_sal = compute_saliency_map(enc, pipeline, device, rows, u_dir)
    om_mask, q_mask, q_from_raw, info = structure_masks(rows, pipeline)
    rng = np.random.default_rng(20260613)
    om_res = overlap_vs_chance(saliency, om_mask, n_perm=2000, rng=rng)
    q_res = overlap_vs_chance(saliency, q_mask, n_perm=2000, rng=rng)
    return {
        "device": device_name,
        "n_impact_frames": int(n_sal),
        "q_from_raw_u": bool(q_from_raw),
        "mask_info": info,
        "omega_overlap": om_res,
        "q_overlap": q_res,
        "saliency_map": saliency,
        "omega_mask_mean": om_mask,
        "q_mask_mean": q_mask,
    }


# =============================================================================
# Figure.
# =============================================================================
def make_figure(collective: dict, energy: dict, footprint: dict | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(REPO / "scripts" / "session21"))
    import figstyle as fs  # noqa: E402

    fs.use_style()
    n_panels = 3 if footprint is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(fs.TEXTWIDTH_IN, fs.TEXTWIDTH_IN * 0.34))
    if n_panels == 2:
        axes = list(axes) + [None]

    fam_order = [("jepa_tf_noc", "jepa"), ("fukami", "fukami"), ("pod", "pod")]

    # Panel (a): collective-code bars (combo, best-single, gap) per family.
    ax = axes[0]
    cf = collective["per_family"]
    present = [(f, t) for f, t in fam_order if f in cf]
    xpos = np.arange(len(present))
    w = 0.27
    combo = [cf[f]["combo"] for f, _ in present]
    best = [cf[f]["best_single"] for f, _ in present]
    gap = [cf[f]["gap"] for f, _ in present]
    ax.bar(xpos - w, combo, w, label="combo", color="0.3")
    ax.bar(xpos, best, w, label="best single", color="0.65")
    ax.bar(xpos + w, gap, w, label="gap", color="#1b7837")
    ax.set_xticks(xpos)
    ax.set_xticklabels(
        [fs.FAMILY_LABEL[t] for _, t in present], fontsize=5.5, rotation=12, ha="right"
    )
    ax.set_ylabel(r"wake-forecast $|\rho_s|$ / gap")
    ax.set_title("(a) collective vs single code", fontsize=7.5)
    ax.legend(fontsize=5.5, loc="upper right")
    ax.axhline(0, color="0.5", lw=0.5)

    # Panel (b): energy-information curve (skill vs #PCs) per family.
    ax = axes[1]
    ks = energy["ks"]
    for f, t in fam_order:
        if f not in energy["per_family"]:
            continue
        cur = energy["per_family"][f]
        sk = [cur["skill"][k] for k in ks]
        ax.plot(
            ks,
            sk,
            marker=fs.FAMILY_MARKER[t],
            ms=2.5,
            lw=1.0,
            color=fs.FAMILY_COLOR[t],
            label=fs.FAMILY_LABEL[t],
        )
        kk = cur["knee_k"]
        ax.axvline(kk, color=fs.FAMILY_COLOR[t], lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("# leading latent PCs")
    ax.set_ylabel(r"held-out wake $R^2$ ($H=16$)")
    ax.set_title("(b) energy-information curve", fontsize=7.5)
    ax.legend(fontsize=5.5, loc="lower right")

    # Panel (c): footprint overlap vs chance (omega + Q).
    if footprint is not None and axes[2] is not None:
        ax = axes[2]
        labels = [r"$|\omega|$", "Q"]
        ovs = [footprint["omega_overlap"], footprint["q_overlap"]]
        xs = np.arange(2)
        vals = [o["overlap"] for o in ovs]
        chance = [o["chance_mean"] for o in ovs]
        chi = [o["chance_hi"] for o in ovs]
        ax.bar(xs, vals, 0.5, color="#1b7837", label="saliency overlap")
        ax.errorbar(
            xs,
            chance,
            yerr=[np.array(chi) - np.array(chance)] * 1,
            fmt="none",
            ecolor="0.3",
            capsize=3,
            label="chance 97.5%",
        )
        ax.scatter(xs, chance, color="0.3", s=12, zorder=5)
        for i, o in enumerate(ovs):
            tag = "loc." if o["localized"] else "n.s."
            ax.text(
                i, vals[i] + 0.01, f"z={o['z']:.1f}\n{tag}", ha="center", va="bottom", fontsize=5.0
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylabel("energy-wtd overlap")
        ax.set_title("(c) wake-forecast footprint", fontsize=7.5)
        ax.legend(fontsize=5.0, loc="upper right")

    fig.tight_layout(pad=0.5, w_pad=1.3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_wake_code.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[p5] wrote figure {OUT_DIR / 'fig_wake_code.png'}")


# =============================================================================
# Serialisation.
# =============================================================================
def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_outputs(collective: dict, energy: dict, footprint: dict | None, config: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- npz ----
    npz: dict = {}
    npz["energy_ks"] = np.array(energy["ks"])
    for fam, cur in energy["per_family"].items():
        npz[f"energy_skill_{fam}"] = np.array([cur["skill"][k] for k in energy["ks"]])
        if "energy_fraction" in cur:
            npz[f"energy_frac_{fam}"] = np.array([cur["energy_fraction"][k] for k in energy["ks"]])
    if footprint is not None:
        npz["saliency_map"] = footprint["saliency_map"]
        npz["omega_mask_mean"] = footprint["omega_mask_mean"]
        npz["q_mask_mean"] = footprint["q_mask_mean"]
    if npz:
        np.savez(OUT_DIR / "p5_results.npz", **npz)
        print(f"[p5] wrote {OUT_DIR / 'p5_results.npz'}")

    # ---- json ----
    fp_serial = None
    if footprint is not None:
        fp_serial = {
            k: v
            for k, v in footprint.items()
            if k not in ("saliency_map", "omega_mask_mean", "q_mask_mean")
        }
    serial = {
        "config": config,
        "collective_code": _clean(collective),
        "energy_information": _clean(
            {
                "ks": energy["ks"],
                "per_family": {
                    f: {k: v for k, v in cur.items() if k != "skill" or True}
                    for f, cur in energy["per_family"].items()
                },
            }
        ),
        "footprint": _clean(fp_serial) if fp_serial is not None else None,
        "subspace_null_note": (
            "Random mean cos^2 for two K-dim subspaces in d-dim ambient is K/d "
            "(~0.047 for K=3, d=64). Any pairwise cos^2 within ~0.01 of K/d is NOT "
            "overlap; reported via is_overlap_above_null (CLAUDE.md caveat)."
        ),
    }
    (OUT_DIR / "p5_results.json").write_text(json.dumps(serial, indent=2))
    print(f"[p5] wrote {OUT_DIR / 'p5_results.json'}")

    write_numbers_part(collective, energy, footprint)
    write_readme(collective, energy, footprint, config)


def write_numbers_part(collective: dict, energy: dict, footprint: dict | None) -> None:
    cf = collective["per_family"]
    numbers: dict = {}
    if "jepa_tf_noc" in cf:
        j = cf["jepa_tf_noc"]
        numbers["p5_collective_gap_jepa"] = {
            "macro": "NumPfiveCollectiveGapJepa",
            "value": j["gap"],
            "fmt": "%.2f",
            "n": j["n_rows_test_b"],
            "split": "test_b",
            "endpoint": "representational",
            "probe": "ridge",
            "observable": "wake_enstrophy@H16",
            "note": (
                f"combo {j['combo']:.2f} minus best-single {j['best_single']:.2f}; "
                f"n_strong(>{STRONG_THR})={j['n_strong']}; "
                + (
                    f"seed mean {collective.get('jepa_seed_gap_mean', float('nan')):.2f}"
                    f"+-{collective.get('jepa_seed_gap_sd', 0.0):.2f} "
                    f"over {collective.get('jepa_seed_n', 1)} seeds; "
                    if "jepa_seed_gap_mean" in collective
                    else ""
                )
                + "D163 collective-code gap regenerated on v2.1 unconditioned latents."
            ),
        }
    for fam, macro in (("fukami", "Fukami"), ("pod", "Pod")):
        if fam in cf:
            numbers[f"p5_collective_gap_{fam}"] = {
                "macro": f"NumPfiveCollectiveGap{macro}",
                "value": cf[fam]["gap"],
                "fmt": "%.2f",
                "n": cf[fam]["n_rows_test_b"],
                "split": "test_b",
                "observable": "wake_enstrophy@H16",
                "note": f"combo {cf[fam]['combo']:.2f} - best-single {cf[fam]['best_single']:.2f}",
            }
    for fam, macro in (("jepa_tf_noc", "Jepa"), ("fukami", "Fukami"), ("pod", "Pod")):
        if fam in energy["per_family"]:
            cur = energy["per_family"][fam]
            note = f"knee at {KNEE_FRAC:.0%} of asymptotic R^2 {cur['asymptotic_skill']:.2f}"
            if "knee_energy_fraction" in cur:
                note += f"; retained energy {cur['knee_energy_fraction']:.2%}"
            numbers[f"p5_energy_knee_{fam}"] = {
                "macro": f"NumPfiveEnergyKnee{macro}",
                "value": cur["knee_k"],
                "fmt": "%d",
                "unit": "PCs",
                "split": "test_b",
                "observable": "wake_enstrophy@H16",
                "note": note,
            }
    pair = collective.get("paired_jepa_vs_fukami", {})
    if pair.get("n", 0):
        numbers["p5_paired_jepa_vs_fukami_wake"] = {
            "macro": "NumPfivePairedJepaWake",
            "value": pair["mean_delta_abserr"],
            "fmt": "%.3f",
            "ci_lo": pair["case_cluster_enc_mean_ci"][0],
            "ci_hi": pair["case_cluster_enc_mean_ci"][1],
            "n": pair["n"],
            "split": "test_b",
            "observable": "wake_enstrophy@H16 readout |error| (fukami - jepa)",
            "note": (
                f"jepa errs less in {pair['a_better_count']}/{pair['n_effective']} rows, "
                f"one-sided sign p = {pair['sign_p_one_sided']:.3g}; "
                f"case-clustered CI over {pair['n_cases']} cases; sign test + "
                "case-clustered CI, NOT case_permutation_p (B6)."
            ),
        }
    if footprint is not None:
        for kind, macro, key in (
            ("omega", "FootprintOmega", "omega_overlap"),
            ("q", "FootprintQ", "q_overlap"),
        ):
            o = footprint[key]
            if o["localized"]:
                status = "STRUCTURE-LOCALISED"
            elif o["overlap"] < o["chance_lo"]:
                status = "BELOW chance (saliency AVOIDS this structure)"
            else:
                status = "NOT localised (within null)"
            numbers[f"p5_footprint_{kind}_overlap"] = {
                "macro": f"NumPfive{macro}Overlap",
                "value": o["overlap"],
                "fmt": "%.3f",
                "ci_lo": o["chance_lo"],
                "ci_hi": o["chance_hi"],
                "n": footprint["n_impact_frames"],
                "split": "test_b+test_c",
                "observable": f"wake-forecast saliency overlap with {kind} mask",
                "note": (
                    f"chance mean {o['chance_mean']:.3f} "
                    f"(95% null [{o['chance_lo']:.3f}, {o['chance_hi']:.3f}]); "
                    f"z={o['z']:.2f}; {status}; Q from raw /u: {footprint['q_from_raw_u']}."
                ),
            }
    NUMBERS_PART.parent.mkdir(parents=True, exist_ok=True)
    NUMBERS_PART.write_text(json.dumps({"part": "p5_code", "numbers": numbers}, indent=2))
    print(f"[p5] wrote {NUMBERS_PART}")


def write_readme(collective: dict, energy: dict, footprint: dict | None, config: dict) -> None:
    cf = collective["per_family"]
    L: list[str] = []
    L.append("# Physics Track P5: where the wake code lives (v2.1)\n")
    L.append(
        "Three honest re-measurements on the v2.1 UNCONDITIONED latents "
        "(jepa_tf_noc / fukami / pod at d=64). D162/D163 are NOT assumed; every "
        "number below is AS MEASURED on v2.1. Where a result weakens vs the v2 "
        "(D162/D163) story it is flagged.\n"
    )
    L.append("## Subspace-overlap null (CLAUDE.md)\n")
    L.append(
        "The random-baseline mean cos^2 for two K-dim subspaces in d-dim ambient "
        "is K/d (~0.047 for K=3, d=64). Any pairwise cos^2 within ~0.01 of K/d is "
        "NOT overlap and must not be read as functional sharing. The grouping "
        "result below scores its leading-PC subspace overlap against this null.\n"
    )

    L.append("## (i) Collective vs redundant code (D163 regenerated)\n")
    L.append(
        "Per family, toward future wake enstrophy at H=16 on held-out test_b "
        "(targets joined by (case_id, encounter)): combination skill (ridge over "
        "all 64 coordinates, standardised on train), best single-coordinate skill, "
        "the GAP (combo - best single), forecast-beyond-forces partial correlation, "
        "and #coordinates with single |Spearman| > "
        f"{STRONG_THR}. A LARGE positive gap means the wake forecast is a "
        "COLLECTIVE (distributed) code, not carried by one coordinate.\n"
    )
    for fam in ("jepa_tf_noc", "fukami", "pod"):
        if fam not in cf:
            continue
        c = cf[fam]
        L.append(
            f"- {fam}: combo {c['combo']:.3f}, best-single {c['best_single']:.3f}, "
            f"GAP {c['gap']:+.3f}, n_strong {c['n_strong']}, "
            f"beyond-forces {c['beyond_forces']:.3f} "
            f"(n_test_b={c['n_rows_test_b']})."
        )
    if "jepa_seed_gap_mean" in collective:
        L.append(
            f"\nJEPA seed robustness: gap "
            f"{collective['jepa_seed_gap_mean']:+.3f} +- "
            f"{collective['jepa_seed_gap_sd']:.3f} over "
            f"{collective['jepa_seed_n']} seeds.\n"
        )
    if "jepa_tf_noc" in cf and cf["jepa_tf_noc"].get("grouping", {}).get("grouped"):
        g = cf["jepa_tf_noc"]["grouping"]
        L.append(
            f"\nFunctional grouping (jepa, D163 ~{N_GROUP} functions): "
            f"{g['n_active']} wake-active coordinates split into groups of sizes "
            f"{g['group_sizes']}; leading-PC subspace cos^2 = "
            f"{g['pairwise_cos2_leadPC']:.3f} vs K/d null {g['subspace_null_K_over_d']:.3f} "
            f"-> overlap above null: {g['overlap_above_null']} "
            + (
                "(functional SHARING between the two groups)."
                if g["overlap_above_null"]
                else (
                    "(the two groups' leading directions are at or below the K/d "
                    "null, i.e. functionally distinct / not sharing a subspace -- "
                    "consistent with D163's ~2 separate functions)."
                )
            )
            + "\n"
        )
    elif "jepa_tf_noc" in cf:
        g = cf["jepa_tf_noc"].get("grouping", {})
        L.append(f"\nFunctional grouping (jepa): {g.get('reason', 'n/a')}.\n")
    pair = collective.get("paired_jepa_vs_fukami", {})
    if pair.get("n", 0):
        L.append(
            f"\nPaired jepa vs fukami wake forecast (sign test + case-clustered CI, "
            f"NOT case-permutation): jepa errs less in {pair['a_better_count']}/"
            f"{pair['n_effective']} held-out rows (one-sided sign p = "
            f"{pair['sign_p_one_sided']:.3g}); mean |error| advantage "
            f"{pair['mean_delta_abserr']:+.3f} (case-clustered 95% CI "
            f"[{pair['case_cluster_enc_mean_ci'][0]:.3f}, "
            f"{pair['case_cluster_enc_mean_ci'][1]:.3f}] over {pair['n_cases']} cases).\n"
        )
    # honest reproduce-or-weaken note for (i)
    jgap = cf.get("jepa_tf_noc", {}).get("gap")
    if jgap is not None:
        L.append(
            "\n**D163 status (i):** the v2 manuscript reported the JEPA wake "
            "forecast as a collective code (large gap). On v2.1 the JEPA gap is "
            f"{jgap:+.3f}: "
            + (
                "the collective-code result REPRODUCES."
                if jgap > 0.1
                else "the collective-code gap is WEAK/absent on v2.1; the claim "
                "must be softened in the manuscript."
            )
            + "\n"
        )

    L.append("## (ii) Energy-information curve (new)\n")
    L.append(
        "Held-out wake-forecast R^2 (ridge on train, evaluated on test_b) vs the "
        "number of leading latent PCs retained (PCA fit on train post-impact "
        "latents). The KNEE is the smallest #PCs reaching "
        f"{KNEE_FRAC:.0%} of the asymptotic (max-PC) skill. This quantifies how "
        "many latent PCs each family needs to CARRY the wake -- the "
        "participation-ratio-vs-distributed-code tension made numeric.\n"
    )
    for fam in ("jepa_tf_noc", "fukami", "pod"):
        if fam not in energy["per_family"]:
            continue
        cur = energy["per_family"][fam]
        extra = ""
        if "knee_energy_fraction" in cur:
            extra = f", retained energy {cur['knee_energy_fraction']:.1%}"
        L.append(
            f"- {fam}: knee at {cur['knee_k']} PCs "
            f"(asymptotic R^2 {cur['asymptotic_skill']:.3f}{extra})."
        )
    L.append("")

    L.append("## (iii) Footprint of the wake-forecast direction (new; supersedes D163 decode)\n")
    if footprint is None:
        L.append(
            "Footprint part SKIPPED (no encoder pass requested). Run without "
            "--skip-footprint to compute the frozen-encoder saliency overlap.\n"
        )
    else:
        L.append(
            "The wake-forecast DIRECTION (the standardised ridge weight vector of "
            "the H=16 wake probe) is back-propagated through the FROZEN jepa "
            "encoder: saliency = |d (z . u) / d (input omega)|, averaged over "
            f"{footprint['n_impact_frames']} held-out impact-frame inputs "
            f"(device: {footprint['device']}). The mean saliency map is "
            "energy-weighted overlap-scored against the thresholded |omega| mask "
            "and the Q-criterion vortex mask, vs a spatial-permutation chance "
            "baseline.\n"
        )
        L.append(
            "Q-criterion source: "
            + (
                "TRUE 2D Q from raw /u velocity (PREVENT_ROOT), reusing the "
                "_oneoff_pod_q_overlap / exp4_structures_shap form."
                if footprint["q_from_raw_u"]
                else "RAW /u UNAVAILABLE; the Q mask FELL BACK to the |omega| "
                "magnitude mask (documented; the two columns are then identical)."
            )
            + "\n"
        )
        for kind, key in (("|omega|", "omega_overlap"), ("Q", "q_overlap")):
            o = footprint[key]
            below = o["overlap"] < o["chance_lo"]
            if o["localized"]:
                verdict = "**STRUCTURE-LOCALISED** (overlap clears the upper null)."
            elif below:
                verdict = (
                    "**BELOW chance**: the footprint overlap sits below the lower "
                    "null bound, so the wake-forecast saliency actively AVOIDS this "
                    "structure (it lives off the high-magnitude cores / vortices, "
                    "e.g. on shear-layer edges and lower-amplitude regions)."
                )
            else:
                verdict = (
                    "**NOT localised**: overlap is within the chance band; the wake "
                    "code does not preferentially read this structure."
                )
            L.append(
                f"- {kind}: overlap {o['overlap']:.3f} vs chance mean "
                f"{o['chance_mean']:.3f} (95% null [{o['chance_lo']:.3f}, "
                f"{o['chance_hi']:.3f}]), z = {o['z']:.2f} -> " + verdict
            )
        L.append(
            "\n**D163 status (iii):** D163 claimed qualitatively that the code "
            "reads the LEV / shear layer. The number above is the honest test: "
            + (
                "at least one structure mask clears the chance null, so the "
                "footprint IS structure-localised."
                if (footprint["omega_overlap"]["localized"] or footprint["q_overlap"]["localized"])
                else "NEITHER mask clears the chance null on v2.1 (both overlaps "
                "sit AT or BELOW chance), so the 'reads the LEV / shear layer' "
                "sentence is NOT supported by the saliency footprint and must be "
                "dropped. If anything the wake-forecast direction reads the input "
                "OFF the high-|omega| cores and Q vortices, consistent with the "
                "day's meta-pattern that latent-image / decode claims do not "
                "survive fair scrutiny on v2.1."
            )
            + "\n"
        )

    L.append("## Statistics + provenance\n")
    L.append(
        "Held-out test_b (and test_b+test_c for the footprint); probes fitted on "
        "train post-impact frames only. Case-clustered conventions via stats_lib. "
        "The footprint overlap uses a spatial-permutation null (not a paired "
        "location test). Energy-information knee is a trend measure. "
        f"Pipeline divisor 3*train_std = {config['norm_divisor']:.4f} (split_v2p1). "
        f"JEPA encoder checkpoint: {config['jepa_ckpt']}.\n"
    )
    (OUT_DIR / "README.md").write_text("\n".join(L))
    print(f"[p5] wrote {OUT_DIR / 'README.md'}")


# =============================================================================
# Driver.
# =============================================================================
def run(
    splits=("test_b", "test_c"), device_str: str | None = None, skip_footprint: bool = False
) -> dict:
    pipeline_divisor = 3.0 * float(json.loads(PIPELINE_MANIFEST.read_text())["train_stats"]["std"])
    tmap_tr = targets_map("train")
    tmap_te = targets_map("test_b")

    print("[p5] PART (i) collective code")
    collective = run_collective(FAMILIES, tmap_tr, tmap_te)
    for fam, c in collective["per_family"].items():
        print(
            f"[p5]   {fam}: combo {c['combo']:.3f} best {c['best_single']:.3f} "
            f"gap {c['gap']:+.3f} n_strong {c['n_strong']}"
        )

    print("[p5] PART (ii) energy-information curve")
    energy = run_energy_info(FAMILIES, tmap_tr, tmap_te)
    for fam, cur in energy["per_family"].items():
        print(f"[p5]   {fam}: knee {cur['knee_k']} PCs (asym R^2 {cur['asymptotic_skill']:.3f})")

    footprint = None
    if not skip_footprint:
        print("[p5] PART (iii) footprint saliency (frozen encoder autograd)")
        # the wake-forecast direction is the jepa ridge direction (standardised).
        jblob = load_family_latents("jepa_tf_noc_d64_s42")
        Xtr, ytr, _, _ = build_post_impact(jblob["train"], tmap_tr)
        Xte, yte, Fte, _ = build_post_impact(jblob["test_b"], tmap_te)
        u_dir = collective_code_skill(Xtr, ytr, Xte, yte, Fte)["ridge_dir"]
        footprint = run_footprint(splits, np.asarray(u_dir), device_str)
        print(
            f"[p5]   footprint device {footprint['device']}, "
            f"{footprint['n_impact_frames']} frames; "
            f"omega overlap {footprint['omega_overlap']['overlap']:.3f} "
            f"(z={footprint['omega_overlap']['z']:.2f}, "
            f"loc={footprint['omega_overlap']['localized']}); "
            f"Q overlap {footprint['q_overlap']['overlap']:.3f} "
            f"(z={footprint['q_overlap']['z']:.2f}, "
            f"loc={footprint['q_overlap']['localized']}, raw_u={footprint['q_from_raw_u']})"
        )

    config = {
        "horizon": H,
        "wake_observable": WAKE_OBS,
        "strong_threshold": STRONG_THR,
        "knee_fraction": KNEE_FRAC,
        "n_group": N_GROUP,
        "norm_divisor": pipeline_divisor,
        "jepa_ckpt": str(JEPA_CKPT.relative_to(REPO)),
        "splits_footprint": list(splits),
        "device_requested": device_str or "rtx6000:1",
        "families": [f for _, f, _ in FAMILIES],
    }
    make_figure(collective, energy, footprint)
    write_outputs(collective, energy, footprint, config)
    return {"collective": collective, "energy": energy, "footprint": footprint}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--splits",
        nargs="+",
        default=["test_b", "test_c"],
        help="held-out splits for the footprint (default test_b test_c)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="force a torch device for the saliency autograd (e.g. cpu); default RTX 6000 gpu 1",
    )
    p.add_argument(
        "--skip-footprint",
        action="store_true",
        help="skip Part (iii) (no encoder pass; CPU-only parts (i)+(ii))",
    )
    args = p.parse_args()
    run(splits=tuple(args.splits), device_str=args.device, skip_footprint=args.skip_footprint)


if __name__ == "__main__":
    main()

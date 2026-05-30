"""Session 20 Track D: optimal-transport field dissimilarity and OT-latent alignment.

Reference: Tran, Yeh, Taira, J. Fluid Mech. 1027, A24 (2026).

Two sub-analyses, one script.

D-i  OT field dissimilarity replacing the misleading SSIM
---------------------------------------------------------
A reconstructive autoencoder (Fukami) can score a high SSIM by agreeing on the
bulk-zero background while collapsing the actual leading-edge vortex. Wang-SSIM
rewards that because most of the field is quiescent. Unbalanced optimal transport
exposes the collapse: it costs a great deal of transport "work" to move a nearly
empty reconstruction onto the true vortex, and the marginal-mass mismatch (a
collapsed recon has far less vorticity mass than the truth) is penalised by the
KL marginal-relaxation term.

We implement the Tran et al. eq (2.5) signed-vorticity split::

    d_field(V1, V2) = S_eps(m+_1, m+_2) + S_eps(m-_1, m-_2)

with m+ = max(omega, 0) and m- = max(-omega, 0) the positive / negative vorticity
treated as (unnormalised) mass distributions, transported SEPARATELY and summed.
S_eps is the entropic unbalanced-OT transport cost (POT
ot.unbalanced.sinkhorn_unbalanced, KL marginal divergence). The ground cost is the
squared Euclidean distance between pixel coordinates in CHORD units; a
characteristic radius rho = 1 chord sets the cost scale (a unit of mass moved one
chord costs 1).

Tractability: a full 192x96 = 18432-point Sinkhorn is intractable. We
average-pool each field by a factor of 4 to 48x24 = 1152 points BEFORE building
the 1152x1152 cost matrix. This preserves 8 pixels/chord, more than enough to
resolve the leading-edge vortex (~1 chord across) and the transport geometry.

We also report Wang-SSIM (L = 8.31) on the same decoded fields for continuity.

D-i GATE: the OT field distance must rank the collapsed Fukami reconstruction
WORSE (larger distance) than JEPA on test_b. If it does, OT replaces SSIM as the
headline reconstruction metric. If it does NOT reverse the SSIM ranking, we report
descriptively.

D-ii  OT-geodesic vs latent-distance alignment (drift mechanism, geometric)
---------------------------------------------------------------------------
Claim: the JEPA latent metric tracks the physical transport geometry better than
the Fukami latent metric, so a single predictor step is a transport-consistent
move and iterating it stays on-manifold.

Per test_b encounter we build (i) the matrix of OT distances between DNS vorticity
FRAMES and (ii) the matrix of latent Euclidean distances between the same frames,
then report the Spearman correlation between the two, for JEPA d=64, JEPA d=32, and
Fukami d=64. A higher Spearman means the latent geometry is a better isometry of
the physical (OT) geometry.

Tractability choice: option (a), full frame-frame OT matrix on a UNIFORMLY
SUBSAMPLED set of frames (stride FRAME_STRIDE, ~30 of the 120 frames spanning the
whole encounter). A full 120x120 matrix is 7140 pairs/encounter (~16 min each,
~11 h total) which is infeasible; stride-4 gives a 30x30 matrix (435 pairs) per
encounter at ~1 min, and the off-diagonal entries give a clean Shepard/Spearman.
The OT frame-frame matrix depends only on the DNS fields, so it is computed ONCE
per encounter and correlated against all three latent matrices.

The reported headline is the PER-ENCOUNTER mean Spearman (correlate the off-diagonal
OT and latent distance vectors within each encounter, then average across
encounters). This is the geometrically correct statistic for the claim, which is
about whether the latent metric tracks transport geometry *along a single
trajectory*. A naive POOLED Spearman (concatenate all off-diagonal pairs across
encounters before correlating) is also reported but is misleading: it mixes
within-encounter structure with between-encounter latent-norm scale differences,
and the drift-prone Fukami latents have large per-encounter norm spread that
correlates with the between-encounter OT spread, inflating Fukami's pooled value
above JEPA's. The gate uses the per-encounter mean.

D-ii GATE: JEPA OT-latent Spearman must exceed Fukami's by a clear margin
(target > 0.15 absolute) on test_b. If marginal, report descriptively.

Pure CPU numpy + POT. No GPU, no training. READ-ONLY.
"""

from __future__ import annotations

import os

# Limit BLAS/OMP threads to 1 so the process pool (one worker per encounter) does
# not oversubscribe the 96 cores. Must be set before numpy/scipy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import ot  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DECODED_DIR = REPO / "outputs" / "session20" / "decoded"
OUT_DIR = REPO / "outputs" / "session20" / "ot"
MANIFEST = REPO / "outputs" / "data_pipeline" / "v1" / "manifest.json"

# Latent sources (z_full = per-frame latent trajectory, (n, 120, d)).
LATENTS = {
    "jepa_d64": (REPO / "outputs" / "session14" / "latents" / "S12_E_d64" / "test_b.npz", "z_full"),
    "jepa_d32": (REPO / "outputs" / "session14" / "latents" / "S12_E_d32" / "test_b.npz", "z_full"),
    "fukami": (
        REPO / "outputs" / "session18" / "exp_b1" / "latents_fukami_d64_noBN" / "test_b.npz",
        "z_full",
    ),
}

# Decoded-field method keys for D-i.
METHODS_DI = {
    "jepa_d64": "jepa_norm",
    "jepa_d32": "jepa_d32_norm",
    "fukami": "fukami_norm",
    "pod": "pod_norm",
}

# Grid geometry (locked, see CLAUDE.md): physical extent x in (-1.5, 4.5) over 192
# px and y in (-1.5, 1.5) over 96 px -> 32 px/chord isotropic.
EXTENT_X = (-1.5, 4.5)
EXTENT_Y = (-1.5, 1.5)
POOL = 4  # average-pool factor: 192x96 -> 48x24 (8 px/chord).
NX_POOL = 192 // POOL  # 48 along the x (chordwise) axis
NY_POOL = 96 // POOL  # 24 along the y (cross-stream) axis

# Unbalanced-OT hyperparameters.
#   reg    : entropic regularisation of the transport plan (Sinkhorn).
#   reg_m  : KL marginal-relaxation weight; with the cost in chord^2 and reg_m ~ 1
#            the cost of NOT matching a unit of mass is comparable to moving it ~1
#            chord, so a collapsed (low-mass) field is penalised both for the
#            transport it does do and for the mass it fails to match. rho = 1 chord.
OT_REG = 0.05
OT_REG_M = 1.0
OT_NUMITERMAX = 1000
OT_STOPTHR = 1e-6
MASS_EPS = 1e-12  # numerical floor so all-zero half-fields do not divide by zero

# SSIM constants: Wang convention on pipeline-normalised omega, L = 8.31
# (= 2 * global p99.9 of |target_norm|; see CLAUDE.md SSIM convention).
SSIM_L = 8.31
SSIM_C1 = (0.01 * SSIM_L) ** 2
SSIM_C2 = (0.03 * SSIM_L) ** 2

IMPACT_IDX = 1  # offsets = [-8, 0, 8, 16, 24, 32, 40]; index 1 = impact (offset 0)
IMPACT16_IDX = 3  # offset +16

FRAME_STRIDE = 4  # D-ii frame subsample: 120 -> 30 frames spanning the encounter


# ----------------------------------------------------------------------------
# Geometry / cost matrix
# ----------------------------------------------------------------------------
def pool4(field: np.ndarray) -> np.ndarray:
    """Average-pool a (192, 96) field by factor POOL to (48, 24)."""
    h, w = field.shape
    return field.reshape(h // POOL, POOL, w // POOL, POOL).mean(axis=(1, 3))


def build_cost_matrix() -> np.ndarray:
    """Squared-Euclidean ground cost between pooled pixel centres, in chord^2.

    The H (size-48) axis maps to physical x in EXTENT_X, the W (size-24) axis to
    physical y in EXTENT_Y. Pixel centres are taken at cell midpoints.
    """
    dx = (EXTENT_X[1] - EXTENT_X[0]) / NX_POOL
    dy = (EXTENT_Y[1] - EXTENT_Y[0]) / NY_POOL
    xs = EXTENT_X[0] + (np.arange(NX_POOL) + 0.5) * dx  # chord units
    ys = EXTENT_Y[0] + (np.arange(NY_POOL) + 0.5) * dy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")  # (48, 24)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (1152, 2)
    return ot.dist(coords, coords, metric="sqeuclidean")  # (1152, 1152), chord^2


# Module-global cost matrix. Built once in main() and inherited by Pool workers
# through fork (copy-on-write), so it is not pickled per task.
_COST: np.ndarray | None = None


def split_pos_neg_pooled(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool then split into positive / negative vorticity mass vectors (1152,)."""
    mp = pool4(np.maximum(field, 0.0)).ravel()
    mn = pool4(np.maximum(-field, 0.0)).ravel()
    return mp, mn


# ----------------------------------------------------------------------------
# Unbalanced OT transport cost S_eps and the signed d_field
# ----------------------------------------------------------------------------
def s_eps(a: np.ndarray, b: np.ndarray, cost: np.ndarray) -> float:
    """Entropic unbalanced-OT transport cost <T, M> between mass vectors a, b.

    a and b are UNNORMALISED (their sums may differ); the KL marginal relaxation
    absorbs the mass mismatch. If both half-fields are empty the cost is 0.
    """
    sa, sb = a.sum(), b.sum()
    if sa < 1e-9 and sb < 1e-9:
        return 0.0
    aa = a.astype(np.float64) + MASS_EPS
    bb = b.astype(np.float64) + MASS_EPS
    plan = ot.unbalanced.sinkhorn_unbalanced(
        aa,
        bb,
        cost,
        OT_REG,
        OT_REG_M,
        reg_type="kl",
        numItermax=OT_NUMITERMAX,
        stopThr=OT_STOPTHR,
    )
    return float((plan * cost).sum())


def d_field(f1: np.ndarray, f2: np.ndarray, cost: np.ndarray) -> float:
    """Tran et al. eq (2.5) signed-vorticity OT field distance between two (192,96)."""
    p1, n1 = split_pos_neg_pooled(f1)
    p2, n2 = split_pos_neg_pooled(f2)
    return s_eps(p1, p2, cost) + s_eps(n1, n2, cost)


# ----------------------------------------------------------------------------
# SSIM (Wang convention, global single-window form on full-resolution fields)
# ----------------------------------------------------------------------------
def ssim_global(x: np.ndarray, y: np.ndarray) -> float:
    """Global Wang-SSIM between two fields with L = 8.31 constants.

    Single-window (whole-field) form: uses global means, variances, covariance.
    This matches the project SSIM-continuity convention used elsewhere in
    Session 20 for decoded fields (one scalar per field pair).
    """
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cxy = ((x - mx) * (y - my)).mean()
    num = (2 * mx * my + SSIM_C1) * (2 * cxy + SSIM_C2)
    den = (mx * mx + my * my + SSIM_C1) * (vx + vy + SSIM_C2)
    return float(num / den)


# ----------------------------------------------------------------------------
# D-i: per-method OT field distance + SSIM, per split and frame offset
# ----------------------------------------------------------------------------
def _d_i_one_encounter(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Worker: one encounter -> (ot (4 methods, 7 offsets), ssim (4, 7)).

    Receives the target (7,192,96) and a dict of recon arrays so only this
    encounter's slices are pickled to the worker, not the whole split.
    """
    tgt_enc, recon_enc = args  # tgt_enc (7,H,W); recon_enc dict mname->(7,H,W)
    n_off = tgt_enc.shape[0]
    methods = list(METHODS_DI.keys())
    ot_out = np.zeros((len(methods), n_off))
    ssim_out = np.zeros((len(methods), n_off))
    # Precompute pooled pos/neg masses of the target per offset (reused across methods).
    tgt_masses = [split_pos_neg_pooled(tgt_enc[oi]) for oi in range(n_off)]
    for mi, mname in enumerate(methods):
        rec = recon_enc[mname]
        for oi in range(n_off):
            tp, tn = tgt_masses[oi]
            rp, rn = split_pos_neg_pooled(rec[oi])
            ot_out[mi, oi] = s_eps(tp, rp, _COST) + s_eps(tn, rn, _COST)
            ssim_out[mi, oi] = ssim_global(tgt_enc[oi], rec[oi])
    return ot_out, ssim_out


def run_d_i(splits: list[str], max_enc: int | None, workers: int) -> dict:
    """Compute mean OT d_field and mean SSIM per (method, split, frame offset)."""
    results: dict = {}
    methods = list(METHODS_DI.keys())
    for split in splits:
        d = np.load(DECODED_DIR / f"{split}.npz", allow_pickle=True)
        target = d["target_norm"]  # (n, 7, 192, 96)
        offsets = [int(o) for o in d["offsets"]]
        n_all = target.shape[0]
        n = n_all if max_enc is None else min(max_enc, n_all)
        recons = {m: d[METHODS_DI[m]] for m in methods}
        tasks = [(target[ei], {m: recons[m][ei] for m in methods}) for ei in range(n)]
        t0 = time.time()
        if workers > 1:
            with Pool(workers) as pool:
                out = pool.map(_d_i_one_encounter, tasks, chunksize=1)
        else:
            out = [_d_i_one_encounter(t) for t in tasks]
        ot_stack = np.stack([o[0] for o in out])  # (n, 4, 7)
        ssim_stack = np.stack([o[1] for o in out])  # (n, 4, 7)
        ot_mean = ot_stack.mean(axis=0)  # (4, 7)
        ssim_mean = ssim_stack.mean(axis=0)  # (4, 7)
        results[split] = {"offsets": offsets, "n_encounters": int(n), "methods": {}}
        for mi, mname in enumerate(methods):
            results[split]["methods"][mname] = {
                "ot_field_by_offset": ot_mean[mi].tolist(),
                "ssim_by_offset": ssim_mean[mi].tolist(),
                "ot_field_impact": float(ot_mean[mi, IMPACT_IDX]),
                "ot_field_impact16": float(ot_mean[mi, IMPACT16_IDX]),
                "ssim_impact": float(ssim_mean[mi, IMPACT_IDX]),
                "ssim_impact16": float(ssim_mean[mi, IMPACT16_IDX]),
            }
            print(
                f"  [D-i] {split:7s} {mname:9s} "
                f"OT_impact={ot_mean[mi, IMPACT_IDX]:.4f} "
                f"OT_imp16={ot_mean[mi, IMPACT16_IDX]:.4f} "
                f"SSIM_impact={ssim_mean[mi, IMPACT_IDX]:.4f}",
                flush=True,
            )
        print(f"    ({split}: {n} enc, {time.time()-t0:.1f}s)", flush=True)
    return results


# ----------------------------------------------------------------------------
# D-ii: OT frame-frame matrix vs latent-distance matrix, Spearman per method
# ----------------------------------------------------------------------------
def _load_normalised_dns(pipe: OmegaPipeline, case_id: str, enc: int) -> np.ndarray:
    """Load raw DNS omega_z for one encounter and apply the decode pipeline.

    Returns (120, 192, 96) in 3-sigma normalised space, identical to the
    target_norm produced for the decoded fields.
    """
    default_cache = Path(os.path.expanduser("~/PREVENT")) / "data" / "processed" / "vortex-jepa"
    cache = Path(os.environ.get("VORTEX_JEPA_CACHE", str(default_cache)))
    path = cache / "v1" / case_id / f"encounter_{enc:02d}.h5"
    import h5py

    with h5py.File(path, "r") as f:
        omega = f["omega_z"][:]  # (120, 192, 96), raw
    pre = pipe.preprocess_raw(omega, case_id, enc)
    return pipe.normalize(pre)


def upper_offdiag(mat: np.ndarray) -> np.ndarray:
    """Flatten the strict upper triangle of a square matrix."""
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]


def _d_ii_ot_one_encounter(args: tuple) -> np.ndarray:
    """Worker: compute the OT frame-frame off-diagonal vector for one encounter.

    Heavy part (only depends on the DNS fields). Loads raw omega, applies the
    decode pipeline, computes the nf x nf OT matrix on subsampled frames, returns
    its strict-upper-triangle as a 1-D vector.
    """
    case_id, enc, frames, manifest_str = args
    pipe = OmegaPipeline.from_manifest(manifest_str)
    omega_norm = _load_normalised_dns(pipe, case_id, enc)  # (120, 192, 96)
    masses = [split_pos_neg_pooled(omega_norm[fr]) for fr in frames]
    nf = len(frames)
    ot_mat = np.zeros((nf, nf))
    for a in range(nf):
        mpa, mna = masses[a]
        for b in range(a + 1, nf):
            mpb, mnb = masses[b]
            val = s_eps(mpa, mpb, _COST) + s_eps(mna, mnb, _COST)
            ot_mat[a, b] = ot_mat[b, a] = val
    return upper_offdiag(ot_mat)


def run_d_ii(max_enc: int | None, workers: int) -> dict:
    """Spearman(OT frame-frame, latent frame-frame) per method on test_b."""
    # Reference encounter ordering: the JEPA d64 latent file.
    ref = np.load(LATENTS["jepa_d64"][0], allow_pickle=True)
    case_ids = [str(c) for c in ref["case_id"]]
    enc_idx = [int(e) for e in ref["encounter_index"]]
    n_enc_all = len(case_ids)
    sel = list(range(n_enc_all if max_enc is None else min(max_enc, n_enc_all)))

    # Load all three latent stacks; verify their (case_id, encounter) order matches.
    lat_stacks: dict[str, np.ndarray] = {}
    for mname, (path, key) in LATENTS.items():
        dd = np.load(path, allow_pickle=True)
        lat_stacks[mname] = dd[key]  # (n, 120, d)
        # Fukami uses 'case_ids'/'encounter_indices'; jepa uses 'case_id'/'encounter_index'.
        cid_key = "case_id" if "case_id" in dd.files else "case_ids"
        eix_key = "encounter_index" if "encounter_index" in dd.files else "encounter_indices"
        these_cids = [str(c) for c in dd[cid_key]]
        these_eix = [int(e) for e in dd[eix_key]]
        assert (
            these_cids == case_ids and these_eix == enc_idx
        ), f"latent ordering mismatch for {mname}"

    frames = np.arange(0, 120, FRAME_STRIDE)  # uniformly subsampled frame indices
    nf = len(frames)

    per_method_spearman: dict[str, list[float]] = {m: [] for m in LATENTS}
    # Shepard-scatter accumulators (off-diagonal OT vs latent distances, all enc).
    shepard: dict[str, dict[str, list]] = {m: {"ot": [], "lat": []} for m in LATENTS}

    # Heavy OT frame-frame off-diagonal vectors, one per encounter, in parallel.
    tasks = [(case_ids[k], enc_idx[k], frames, str(MANIFEST)) for k in sel]
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as pool:
            ot_offs = pool.map(_d_ii_ot_one_encounter, tasks, chunksize=1)
    else:
        ot_offs = [_d_ii_ot_one_encounter(t) for t in tasks]
    print(f"  [D-ii] OT matrices: {len(sel)} enc in {time.time()-t0:.1f}s", flush=True)

    # Cheap latent correlations in the main process.
    for ki, k in enumerate(sel):
        ot_off = ot_offs[ki]
        cid, enc = case_ids[k], enc_idx[k]
        line = [f"  [D-ii] enc {k:2d} {cid} e{enc}"]
        for mname in LATENTS:
            z = lat_stacks[mname][k]  # (120, d)
            zf = z[frames]  # (nf, d)
            diff = zf[:, None, :] - zf[None, :, :]
            lat_mat = np.sqrt((diff * diff).sum(-1))  # (nf, nf) Euclidean
            lat_off = upper_offdiag(lat_mat)
            rho = spearmanr(ot_off, lat_off).correlation
            per_method_spearman[mname].append(float(rho))
            shepard[mname]["ot"].append(ot_off)
            shepard[mname]["lat"].append(lat_off)
            line.append(f"{mname}={rho:.3f}")
        print(" ".join(line), flush=True)

    summary: dict = {
        "frame_stride": FRAME_STRIDE,
        "n_frames_subsampled": int(nf),
        "n_encounters": int(len(sel)),
        "per_method": {},
    }
    for mname in LATENTS:
        arr = np.array(per_method_spearman[mname])
        summary["per_method"][mname] = {
            "spearman_mean": float(arr.mean()),
            "spearman_std": float(arr.std()),
            "spearman_per_encounter": arr.tolist(),
        }
    # Pooled Spearman across all off-diagonal pairs (all encounters concatenated).
    for mname in LATENTS:
        ot_all = np.concatenate(shepard[mname]["ot"])
        lat_all = np.concatenate(shepard[mname]["lat"])
        summary["per_method"][mname]["spearman_pooled"] = float(
            spearmanr(ot_all, lat_all).correlation
        )
    summary["_shepard"] = shepard  # not JSON-serialised; used for the figure only
    return summary


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def make_figure(d_i: dict, d_ii: dict, fig_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 5))

    # Panel 1: D-i bar chart, OT field distance by method at impact on test_b,
    # with SSIM annotated. Lead split = test_b (the gate split).
    ax1 = fig.add_subplot(1, 2, 1)
    split = "test_b"
    methods = list(METHODS_DI.keys())
    ot_imp = [d_i[split]["methods"][m]["ot_field_impact"] for m in methods]
    ssim_imp = [d_i[split]["methods"][m]["ssim_impact"] for m in methods]
    xpos = np.arange(len(methods))
    colors = ["#2c7fb8", "#7fcdbb", "#d95f0e", "#999999"]
    bars = ax1.bar(xpos, ot_imp, color=colors, edgecolor="k", linewidth=0.6)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(methods, rotation=15)
    ax1.set_ylabel("OT field distance $d_{field}$ (impact, chord$^2$)")
    ax1.set_title("D-i: OT field distance vs SSIM (test_b, impact)")
    for xi, (b, s) in enumerate(zip(bars, ssim_imp)):
        ax1.text(
            xi,
            b.get_height(),
            f"SSIM\n{s:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax1.margins(y=0.18)

    # Panel 2: D-ii Shepard scatter, OT vs latent distances, JEPA d64 vs Fukami.
    # The headline metric is the PER-ENCOUNTER mean Spearman (the gate metric);
    # the scatter pools all encounters after standardising both axes WITHIN each
    # encounter (z-score), so the cloud reflects the within-encounter geometry the
    # claim is about and does not conflate between-encounter scale differences
    # (which inflate a naive pooled Spearman, especially for the drift-prone Fukami
    # latents). The legend reports the per-encounter mean +/- std.
    ax2 = fig.add_subplot(1, 2, 2)
    shp = d_ii["_shepard"]

    def standardise(v):
        v = np.asarray(v, dtype=float)
        s = v.std()
        return (v - v.mean()) / s if s > 0 else v - v.mean()

    def pooled_zscored(ot_list, lat_list, n=4000, seed=0):
        ox = np.concatenate([standardise(o) for o in ot_list])
        ly = np.concatenate([standardise(latent) for latent in lat_list])
        rng = np.random.default_rng(seed)
        if ox.size > n:
            sel = rng.choice(ox.size, n, replace=False)
            return ox[sel], ly[sel]
        return ox, ly

    for mname, col, lab in [
        ("jepa_d64", "#2c7fb8", "JEPA d=64"),
        ("fukami", "#d95f0e", "Fukami d=64"),
    ]:
        ox, ly = pooled_zscored(shp[mname]["ot"], shp[mname]["lat"])
        pm = d_ii["per_method"][mname]
        lbl = "{} ($\\overline{{r_s}}$={:.3f}$\\pm${:.3f})".format(
            lab, pm["spearman_mean"], pm["spearman_std"]
        )
        ax2.scatter(ox, ly, s=4, alpha=0.22, color=col, label=lbl)
    ax2.set_xlabel("OT distance between DNS frames (per-encounter z-score)")
    ax2.set_ylabel("latent Euclidean distance (per-encounter z-score)")
    ax2.set_title("D-ii: OT-vs-latent Shepard (test_b, within-encounter)")
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(fig_path.with_suffix(".png"), dpi=150)
    fig.savefig(fig_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  figure -> {fig_path.with_suffix('.png')} / .pdf", flush=True)


# ----------------------------------------------------------------------------
# Gates + main
# ----------------------------------------------------------------------------
def evaluate_gates(d_i: dict, d_ii: dict) -> dict:
    # D-i gate: OT ranks Fukami WORSE (larger d_field) than JEPA on test_b impact.
    tb = d_i["test_b"]["methods"]
    ot_fuk = tb["fukami"]["ot_field_impact"]
    ot_jep = tb["jepa_d64"]["ot_field_impact"]
    ssim_fuk = tb["fukami"]["ssim_impact"]
    ssim_jep = tb["jepa_d64"]["ssim_impact"]
    # Does SSIM (mistakenly) rank Fukami >= JEPA? Higher SSIM = "better".
    ssim_ranks_fukami_better = ssim_fuk >= ssim_jep
    ot_ranks_fukami_worse = ot_fuk > ot_jep
    d_i_gate = bool(ot_ranks_fukami_worse)

    # D-ii gate: JEPA Spearman exceeds Fukami's by > 0.15 absolute (test_b).
    pm = d_ii["per_method"]
    jep64 = pm["jepa_d64"]["spearman_mean"]
    jep32 = pm["jepa_d32"]["spearman_mean"]
    fuk = pm["fukami"]["spearman_mean"]
    margin64 = jep64 - fuk
    margin32 = jep32 - fuk
    d_ii_gate = bool(margin64 > 0.15)

    return {
        "d_i": {
            "passed": d_i_gate,
            "ot_fukami_impact": ot_fuk,
            "ot_jepa_d64_impact": ot_jep,
            "ot_margin_fukami_minus_jepa": ot_fuk - ot_jep,
            "ssim_fukami_impact": ssim_fuk,
            "ssim_jepa_d64_impact": ssim_jep,
            "ssim_ranks_fukami_at_least_as_good": bool(ssim_ranks_fukami_better),
            "ot_reverses_ssim_ranking": bool(d_i_gate and ssim_ranks_fukami_better),
        },
        "d_ii": {
            "passed": d_ii_gate,
            "spearman_jepa_d64": jep64,
            "spearman_jepa_d32": jep32,
            "spearman_fukami": fuk,
            "margin_d64_minus_fukami": margin64,
            "margin_d32_minus_fukami": margin32,
            "target_margin": 0.15,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-enc",
        type=int,
        default=None,
        help="cap encounters per split (smoke test); default = all",
    )
    ap.add_argument("--skip-d-i", action="store_true")
    ap.add_argument("--skip-d-ii", action="store_true")
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["test_a", "test_b", "test_c"],
        help="D-i splits to evaluate",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=32,
        help="encounter-level process-pool size (1 = serial)",
    )
    ap.add_argument(
        "--figure-only",
        action="store_true",
        help="rebuild ot.png/pdf from cached ot_results.json + shepard_data.npz",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.figure_only:
        with open(OUT_DIR / "ot_results.json") as f:
            cached = json.load(f)
        sh = np.load(OUT_DIR / "shepard_data.npz")
        shepard = {}
        for m in cached["d_ii"]["per_method"]:
            split_at = np.cumsum(sh[f"{m}__sizes"])[:-1]
            shepard[m] = {
                "ot": np.split(sh[f"{m}__ot"], split_at),
                "lat": np.split(sh[f"{m}__lat"], split_at),
            }
        make_figure(cached["d_i"], {**cached["d_ii"], "_shepard": shepard}, OUT_DIR / "ot")
        print("figure-only rebuild complete", flush=True)
        return

    global _COST
    _COST = build_cost_matrix()
    print(
        f"cost matrix {_COST.shape}, max {_COST.max():.2f} chord^2 "
        f"(pooled {NX_POOL}x{NY_POOL} = {NX_POOL*NY_POOL} pts, factor {POOL}); "
        f"OT reg={OT_REG} reg_m={OT_REG_M} kl; workers={args.workers}",
        flush=True,
    )

    out: dict = {
        "reference": "Tran, Yeh, Taira, J. Fluid Mech. 1027, A24 (2026)",
        "config": {
            "pool_factor": POOL,
            "pooled_grid": [NX_POOL, NY_POOL],
            "extent_x_chord": list(EXTENT_X),
            "extent_y_chord": list(EXTENT_Y),
            "ot_reg": OT_REG,
            "ot_reg_m": OT_REG_M,
            "ot_reg_type": "kl",
            "rho_chord": 1.0,
            "ssim_L": SSIM_L,
            "ssim_c1": SSIM_C1,
            "ssim_c2": SSIM_C2,
            "frame_stride_d_ii": FRAME_STRIDE,
            "offsets": [-8, 0, 8, 16, 24, 32, 40],
            "impact_index": IMPACT_IDX,
            "impact16_index": IMPACT16_IDX,
        },
    }

    if not args.skip_d_i:
        print("=== D-i: OT field dissimilarity ===", flush=True)
        out["d_i"] = run_d_i(args.splits, args.max_enc, args.workers)
    if not args.skip_d_ii:
        print("=== D-ii: OT-latent alignment (test_b) ===", flush=True)
        d_ii = run_d_ii(args.max_enc, args.workers)
        shepard = d_ii.pop("_shepard")
        out["d_ii"] = d_ii

    if not args.skip_d_i and not args.skip_d_ii:
        out["gates"] = evaluate_gates(out["d_i"], {**out["d_ii"], "_shepard": shepard})
        make_figure(out["d_i"], {**out["d_ii"], "_shepard": shepard}, OUT_DIR / "ot")
        # Persist the Shepard arrays so the figure can be rebuilt without re-running
        # the expensive OT (used by --figure-only).
        flat: dict[str, np.ndarray] = {}
        for m in shepard:
            flat[f"{m}__ot"] = np.concatenate(shepard[m]["ot"])
            flat[f"{m}__lat"] = np.concatenate(shepard[m]["lat"])
            # encounter boundaries to allow per-encounter z-scoring on reload
            flat[f"{m}__sizes"] = np.array([len(a) for a in shepard[m]["ot"]])
        np.savez_compressed(OUT_DIR / "shepard_data.npz", **flat)

    json_path = OUT_DIR / "ot_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON -> {json_path}", flush=True)

    if "gates" in out:
        g = out["gates"]
        print("\n=== GATES ===")
        print(
            f"D-i  {'PASS' if g['d_i']['passed'] else 'FAIL'}: "
            f"OT Fukami {g['d_i']['ot_fukami_impact']:.4f} vs JEPA "
            f"{g['d_i']['ot_jepa_d64_impact']:.4f} "
            f"(margin {g['d_i']['ot_margin_fukami_minus_jepa']:+.4f}); "
            f"SSIM Fukami {g['d_i']['ssim_fukami_impact']:.4f} vs JEPA "
            f"{g['d_i']['ssim_jepa_d64_impact']:.4f}; "
            f"OT reverses SSIM ranking = {g['d_i']['ot_reverses_ssim_ranking']}"
        )
        print(
            f"D-ii {'PASS' if g['d_ii']['passed'] else 'FAIL'}: "
            f"Spearman JEPA d64 {g['d_ii']['spearman_jepa_d64']:.3f}, "
            f"d32 {g['d_ii']['spearman_jepa_d32']:.3f}, "
            f"Fukami {g['d_ii']['spearman_fukami']:.3f}; "
            f"margin d64 {g['d_ii']['margin_d64_minus_fukami']:+.3f}, "
            f"d32 {g['d_ii']['margin_d32_minus_fukami']:+.3f} (target > 0.15)"
        )


if __name__ == "__main__":
    main()

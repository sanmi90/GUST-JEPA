"""Sparse wall-pressure FLOW-FIELD recovery comparison (author reframe 2026-06-13).

PROBE REGIME DECLARATION (CLAUDE.md probe methodology): the pressure -> latent map
reused here is the STATE-RECOVERY map of scripts/session28/sensing_cf.py. It reads a
family's PER-encounter latent at the readout frame impact + READOUT_H from a causal
window of K wall-pressure taps ending AT that frame; it is NOT a (G, D, Y) parameter
probe. This script adds the PHYSICAL-SPACE companion to that already-done STATE
recovery: it pushes the pressure-estimated latent through each family's frozen
visualization decoder to a recovered omega field and scores the FIELD against DNS
truth at impact + READOUT_H.

The sensing track is now framed as a cross-method comparison of which representation,
estimated from a few wall-pressure taps, best recovers the STATE and the FLOW FIELD.
The earlier pressure -> wake-enstrophy claim is dropped (author decision 2026-06-13);
only state and field recovery are reported.

THE CHAIN (per family, per encounter on test_b / test_c):
    sparse wall pressure (K qDEIM taps over the pre-impact window W)
      --KRR(RBF)-->  estimated latent z_hat(impact + READOUT_H)
      --family decoder-->  recovered omega field
    scored against the DNS truth field at impact + READOUT_H.

DECODE-CEILING REFERENCE (the decoder-quality confound, made explicit): for each
family we ALSO score the field decoded from the TRUE encoded latent z_dns(impact +
READOUT_H) (no pressure step). The gap "pressure-recovered vs decode-ceiling" isolates
how much of the field error is the pressure->state recovery versus the decoder itself.
POD's linear decoder has the best raw ceiling SSIM (0.676 on test_a) while Fukami's is
0.499, so a field-from-pressure ranking can be partly decoder quality, not state
recovery. We never let the ceiling silently flip the ranking: both numbers are
reported side by side and the headline states the gap.

REUSE (do not reinvent):
  scripts/session28/sensing_cf.py           pressure->z KRR map + qDEIM tap picks,
                                             WINDOW, READOUT_H, feats/fit_krr/krr_pred,
                                             5-fold case folds (make_case_folds).
  scripts/session28/decode_operating_points.py  decoder build/forward + operating
                                             iters; recon_*.npz decode-ceiling fields.
  scripts/session9_train_decoder.py          decoder loaders (build_decoder, encoder
                                             load, LatentStore).
  scripts/session21/figstyle.py              Fig-3 vort_panel (RdBu_r, vmin=-3/vmax=+3,
                                             black airfoil).
  scripts/session28/stats_lib.py             case-clustered bootstrap + sign test.
SSIM = Wang K1=0.01 K2=0.03 on PIPELINE-NORMALIZED omega, data range L from the
registry (split_v2p1 = 8.45); the decode-ceiling and DNS-truth fields are both
re-normalized into the pipeline-normalized space before SSIM, since L is in those
units (configs/ssim_data_range.json).

Statistics: 5-fold CASE-level CV on the pressure->z map (sensing_cf folds); a
case-clustered bootstrap CI on the headline field SSIM per family; paired family
differences via one-sided sign test + a case-clustered CI on the per-encounter SSIM
difference (NOT case_permutation_p, the B6 lesson).

GPU: the decoder forward targets the RTX 6000 at require_rtx6000(gpu_index=1) ONLY
(GPU 0 may be busy; the L40S cards are FORBIDDEN). If gpu_index=1 is unavailable the
decoder forward falls back to CPU (the decoder is small); the chosen device is logged
and recorded in results.json. The KRR work is always CPU.

Usage:
    nice -n 10 .venv/bin/python scripts/session28/sensing_field_cf.py
    nice -n 10 .venv/bin/python scripts/session28/sensing_field_cf.py --quick   # K=8 only
    nice -n 10 .venv/bin/python scripts/session28/sensing_field_cf.py --cpu     # force CPU decode
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "session28"))

# Reuse the sensing_cf pressure->z machinery verbatim (loaders, KRR, picks, folds).
import sensing_cf as SC  # noqa: E402
from sensing_cf import feats, fit_krr, krr_pred  # noqa: E402
import stats_lib  # noqa: E402
from closure_matrix import make_case_folds  # noqa: E402

# --------------------------------------------------------------------- constants

PREVENT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
CACHE = (
    Path(os.environ.get("VORTEX_JEPA_CACHE", str(PREVENT / "data/processed/vortex-jepa"))) / "v2p1"
)
LAT_ROOT = REPO / "outputs" / "session28" / "latents"
DNS_PATH = REPO / "outputs" / "session28" / "exp2" / "dns_physical_metrics.npz"
DECODE_DIR = REPO / "outputs" / "session28" / "decode"
OP_PATH = DECODE_DIR / "operating_points.json"
RUNS = REPO / "outputs" / "runs" / "session28"
OMEGA_MANIFEST = REPO / "outputs" / "data_pipeline" / "v2p1" / "manifest.json"

OUT = REPO / "outputs" / "session28" / "sensing" / "field_recovery"
PARTS = REPO / "outputs" / "session28" / "numbers_parts"

K_LIST = [2, 4, 8, 16]
HEADLINE_K = 8
WINDOW = SC.WINDOW  # 30 frames, pressure window (causal, ends at readout)
READOUT_H = SC.READOUT_H  # 16: readout frame is impact + 16, matches decode horizon
N_PRESSURE = SC.N_PRESSURE  # 192 surface taps
SPLITS_EVAL = ["test_b", "test_c"]


@dataclass
class FieldFamily:
    name: str  # display / key name (must match a sensing_cf family)
    macro: str  # LaTeX macro suffix
    color: str  # figure colour (from sensing_cf)
    marker: str  # figure marker
    lat_tag: str  # latents dir (lead seed) for the pressure->z map
    decode_key: str  # decode/<decode_key>/ holding recon_*.npz (ceiling fields)
    op_family: str  # operating_points.json family key
    encoder_ckpt: Path | None  # frozen encoder (JEPA path) or None (post-hoc npz)
    latents_dir: Path | None  # LatentStore dir for post-hoc decoders, else None


# Families with a TRAINED decoder (so a field can be recovered at all). bvae has NO
# decoder under outputs/session28/decode (not in operating_points.json), so it cannot
# enter the FIELD comparison; it is reported as absent in the README. The faithful
# pure-reconstructive comparison is fukami + pod against the predictive jepa_tf_noc.
FAMILIES: list[FieldFamily] = [
    FieldFamily(
        "jepa_tf_noc",
        "JepaTf",
        "#1b7837",
        "o",
        "jepa_tf_noc_d64_s42",
        "tf_noc",
        "dec_jepa_tf_noc_d64_s42",
        RUNS / "jepa_tf_noc_d64_s42" / "encoder" / "checkpoint_iter020000.pt",
        None,
    ),
    FieldFamily(
        "fukami",
        "Fukami",
        "#c0392b",
        "s",
        "fukami_d64_s42",
        "fukami",
        "dec_posthoc_fukami_d64",
        None,
        LAT_ROOT / "fukami_d64_s42",
    ),
    FieldFamily(
        "pod",
        "Pod",
        "#2166ac",
        "^",
        "pod_d64",
        "pod",
        "dec_posthoc_pod_d64",
        None,
        LAT_ROOT / "pod_d64",
    ),
]


# --------------------------------------------------------------------- SSIM + metrics


def ssim_global(x: np.ndarray, y: np.ndarray, L: float) -> float:
    """Global single-window Wang SSIM between two fields (project convention).

    C1 = (0.01 * L)^2, C2 = (0.03 * L)^2 with L the dataset SSIM data range
    (split_v2p1 = 8.45), exactly the form used in
    scripts/session27_exp_ot_field_and_alignment_uncond.py::ssim_global. Inputs
    must be in the SAME (pipeline-normalized) units as L.
    """
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cxy = ((x - mx) * (y - my)).mean()
    num = (2 * mx * my + c1) * (2 * cxy + c2)
    den = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(num / max(den, 1e-18))


def field_mse(x: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between two fields (same units)."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    return float(((x - y) ** 2).mean())


def eps_volume(rec: np.ndarray, dns: np.ndarray, eps: float = 1.0) -> float:
    """Fukami volume L2 relative error ||dns - rec||_2 / ||dns||_2 on one field.

    Matches scripts/session9_train_decoder.py::_l2_relative_error (eps floor 1.0
    on the DENOMINATOR; on pipeline-normalized fields the floor is effectively
    inert because ||dns|| >> 1). rec/dns are flattened internally.
    """
    rec = rec.astype(np.float64).ravel()
    dns = dns.astype(np.float64).ravel()
    num = float(np.sqrt(((dns - rec) ** 2).sum()))
    den = float(np.sqrt((dns**2).sum()))
    return num / max(den, eps)


def peak_vorticity_retention(rec: np.ndarray, dns: np.ndarray) -> float:
    """Recovered peak |omega| over DNS peak |omega| (1.0 = peak fully retained).

    Captures the well-known smoothing of decoded fields: a value < 1 means the
    recovered field under-resolves the vortex-core peak vorticity. Sign is taken
    by magnitude so a correctly-signed but attenuated core still scores < 1.
    """
    dns_peak = float(np.abs(dns).max())
    rec_peak = float(np.abs(rec).max())
    return rec_peak / max(dns_peak, 1e-12)


def enc_keys(case_ids: np.ndarray, enc_idx: np.ndarray) -> np.ndarray:
    """Per-encounter join keys '<case_id>|<enc_idx>' for exact family pairing."""
    return np.array([f"{c}|{int(e)}" for c, e in zip(case_ids, enc_idx)])


def boot_metric_ci(
    values: np.ndarray, case_ids: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> list[float]:
    """Case-clustered percentile bootstrap CI of the MEAN of a per-encounter metric.

    Resamples whole CASES with replacement (no case straddles the resample), then
    takes the mean over the pooled encounters; returns the [2.5, 97.5] percentile
    CI. This is the headline field-SSIM CI per family.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    uc = np.array(sorted(set(case_ids.tolist())))
    by_case = {c: np.where(case_ids == c)[0] for c in uc}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = uc[rng.integers(0, len(uc), size=len(uc))]
        sel = np.concatenate([by_case[c] for c in pick])
        boots[b] = values[sel].mean()
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


# --------------------------------------------------------------------- decoder I/O


def _resolve_device(force_cpu: bool):
    """Return (torch.device, gpu_name|None). RTX 6000 gpu1 only, else CPU."""
    import torch

    if force_cpu:
        return torch.device("cpu"), None
    try:
        from src.utils.device import require_rtx6000

        dev = require_rtx6000(gpu_index=1)
        name = torch.cuda.get_device_name(dev.index)
        if "RTX" not in name or "6000" not in name:
            raise RuntimeError(f"refusing non-RTX6000 device: {name}")
        return dev, name
    except Exception as exc:  # GPU 1 busy / absent -> CPU fallback (decoder is small)
        print(f"[field_cf] RTX 6000 gpu1 unavailable ({exc}); decoder forward on CPU", flush=True)
        return torch.device("cpu"), None


def _load_decoder(fam: FieldFamily, op_iter: int, latent_dim: int, device):
    """Build the family's frozen decoder at its operating iter (decode_operating_points)."""
    import decode_operating_points as DOP

    ckpt = RUNS / fam.op_family / f"decoder_iter{op_iter:06d}.pt"
    return DOP.build_decoder_from_ckpt(ckpt, latent_dim=latent_dim, device=device)


def _decode_latents(dec, z_rows: np.ndarray, pipe, device) -> np.ndarray:
    """(n, d) latents -> (n, 192, 96) RAW-scale omega via the family decoder.

    Reuses decode_operating_points.decode_latents (autocast bf16 on CUDA, fp32 on
    CPU; unnormalizes each frame to raw scale). The whole batch is decoded as one
    pseudo-sequence so the decoder's temporal positions are irrelevant per-frame.
    """
    import decode_operating_points as DOP

    return DOP.decode_latents(dec, z_rows.astype(np.float32), pipe, device)


# --------------------------------------------------------------------- DNS truth


def load_dns_truth_field(case_id: str, k: int, frame: int, pipe) -> np.ndarray:
    """DNS truth omega at one absolute frame, pipeline-preprocessed (mask + clip).

    Returned on the RAW scale (units of 1/t_c) like the decoder outputs, so the
    raw-scale metrics (MSE, eps_volume, peak retention) are directly comparable; SSIM
    normalizes both sides into pipeline-normalized units (see score_fields).
    """
    p = CACHE / case_id / f"encounter_{int(k):02d}.h5"
    with h5py.File(p, "r") as f:
        omega = np.asarray(f["omega_z"], dtype=np.float32)
    frame = int(np.clip(frame, 0, omega.shape[0] - 1))
    return pipe.preprocess_raw(omega[frame], case_id, int(k))


# --------------------------------------------------------------------- decode ceiling


def load_ceiling_fields(fam: FieldFamily, split: str) -> dict:
    """Decode-ceiling omega fields at impact+READOUT_H from the recon_*.npz.

    recon_*.npz holds omega_decoded_horizon (n, len(HORIZONS), 192, 96) decoded from
    the TRUE encoded latent z_dns. We pull the impact+READOUT_H slice and the per-
    encounter (case_id, enc_idx, impact, abs_frame) bookkeeping.
    """
    npz = np.load(DECODE_DIR / fam.decode_key / f"recon_{split}.npz", allow_pickle=True)
    offs = [int(o) for o in npz["horizon_offsets"]]
    if READOUT_H not in offs:
        raise KeyError(f"{fam.decode_key} recon_{split}: horizon {READOUT_H} not in {offs}")
    hidx = offs.index(READOUT_H)
    fields = npz["omega_decoded_horizon"][:, hidx]  # (n, 192, 96) raw scale
    cids = np.array([str(c) for c in npz["case_ids"]])
    eis = npz["encounter_indices"].astype(int)
    impact = npz["impact_frame"].astype(int)
    abs_frames = npz["horizon_abs_frames"][:, hidx].astype(int)
    return {
        "fields": fields,
        "case_ids": cids,
        "enc_idx": eis,
        "impact": impact,
        "abs_frame": abs_frames,
        "keys": enc_keys(cids, eis),
    }


# --------------------------------------------------------------------- scoring


def score_fields(rec: np.ndarray, dns: np.ndarray, pipe, L: float) -> dict:
    """All field metrics for one (recovered, dns) raw-scale pair.

    SSIM is computed in pipeline-NORMALIZED units (L is in those units); MSE,
    eps_volume and peak retention are on the raw scale (interpretable in 1/t_c).
    """
    rec_n = pipe.normalize(rec)
    dns_n = pipe.normalize(dns)
    return {
        "ssim": ssim_global(rec_n, dns_n, L),
        "mse": field_mse(rec, dns),
        "eps_volume": eps_volume(rec, dns),
        "peak_retention": peak_vorticity_retention(rec, dns),
    }


# --------------------------------------------------------------------- main eval


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="K=8 only (smoke).")
    ap.add_argument("--cpu", action="store_true", help="Force the decoder forward onto CPU.")
    ap.add_argument(
        "--cv-boot", type=int, default=2000, help="Case-clustered bootstrap resamples (CI)."
    )
    args = ap.parse_args()
    k_list = [HEADLINE_K] if args.quick else K_LIST

    import torch  # noqa: F401  (decode path imports torch lazily inside helpers)
    from src.data.omega_pipeline import OmegaPipeline, ssim_data_range

    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    pipe = OmegaPipeline.from_manifest(OMEGA_MANIFEST)
    try:
        L = float(ssim_data_range(str(OMEGA_MANIFEST)))
    except Exception:
        L = 8.45
    op = json.loads(OP_PATH.read_text())
    dns_metrics = np.load(DNS_PATH, allow_pickle=True)

    device, gpu_name = _resolve_device(args.cpu)
    decode_on = "RTX6000_gpu1" if gpu_name is not None else "CPU"
    print(
        f"[field_cf] device={device} gpu={gpu_name} decode_on={decode_on} ssim_L={L:.2f} "
        f"K={k_list} readout=impact+{READOUT_H} W={WINDOW}",
        flush=True,
    )

    # ---- placement: reuse sensing_cf's target-blind qDEIM picks (primary) ----
    # The qDEIM snapshot is the TRAIN window-mean pressure (family-independent), so we
    # build it once from the jepa family's train bundle, exactly as sensing_cf does.
    ref_train = SC.load_family_split(FAMILIES[0].lat_tag, "train", dns_metrics)
    snapshot = ref_train.X.mean(axis=1)  # (n_train, 192)
    qdeim = {K: SC.selector_qdeim(snapshot, K) for K in K_LIST}
    print(f"[field_cf] qDEIM picks (target-blind): { {K: qdeim[K] for K in K_LIST} }", flush=True)

    rows: list[dict] = []
    headline: dict[str, dict] = {}  # fam -> per-encounter SSIM arrays at HEADLINE_K
    per_enc_store: dict[str, dict] = {}  # fam -> {split: dict of per-enc metrics + keys}

    for fam in FAMILIES:
        op_iter = int(op["families"][fam.op_family]["best_iter"])
        ceiling_ssim_testa = float(op["families"][fam.op_family]["best_val_ssim"])
        print(f"\n[field_cf] === {fam.name} (decoder iter {op_iter}) ===", flush=True)

        # pressure->z bundles (train for the map; eval splits for deployment)
        tr = SC.load_family_split(fam.lat_tag, "train", dns_metrics)
        eval_bundles = {
            sp: SC.load_family_split(fam.lat_tag, sp, dns_metrics) for sp in SPLITS_EVAL
        }
        d = tr.z.shape[1]

        dec = _load_decoder(fam, op_iter, d, device)
        per_enc_store[fam.name] = {}
        headline[fam.name] = {}

        for K in k_list:
            taps = qdeim[K]
            Xtr = feats(tr.X, taps)

            # ---- 5-fold CASE-level CV of the pressure->z map (sensing_cf folds) ----
            # Out-of-fold z_hat on TRAIN, decoded and scored vs DNS, gives a leakage-
            # free in-distribution field-recovery estimate (reported alongside the
            # held-out test_b / test_c numbers).
            cv_ssim = _cv_field_ssim(tr, Xtr, dec, pipe, L, device)

            # fit the deployment map once on full TRAIN, eval on held-out splits
            pz = fit_krr(Xtr, tr.z)

            for sp in SPLITS_EVAL:
                te = eval_bundles[sp]
                Xte = feats(te.X, taps)
                zhat = krr_pred(pz, Xte)  # (n, d) estimated impact+H latent

                # pressure-recovered fields (decode z_hat)
                rec_fields = _decode_latents(dec, zhat, pipe, device)  # (n, 192, 96) raw

                # decode-ceiling fields (decode TRUE z_dns at impact+H), aligned by key
                ceil = load_ceiling_fields(fam, sp)
                ceil_pos = {k: i for i, k in enumerate(ceil["keys"])}
                te_keys = enc_keys(te.case_ids, te.enc_idx)

                pr_metrics = {m: [] for m in ("ssim", "mse", "eps_volume", "peak_retention")}
                cl_metrics = {m: [] for m in ("ssim", "mse", "eps_volume", "peak_retention")}
                kept_keys, kept_cids = [], []
                for i in range(len(te.case_ids)):
                    key = te_keys[i]
                    if key not in ceil_pos:
                        continue
                    frame = int(te.readout[i])  # absolute readout frame = impact + H
                    dns_field = load_dns_truth_field(
                        te.case_ids[i], int(te.enc_idx[i]), frame, pipe
                    )
                    pr = score_fields(rec_fields[i], dns_field, pipe, L)
                    cl = score_fields(ceil["fields"][ceil_pos[key]], dns_field, pipe, L)
                    for m in pr_metrics:
                        pr_metrics[m].append(pr[m])
                        cl_metrics[m].append(cl[m])
                    kept_keys.append(key)
                    kept_cids.append(te.case_ids[i])

                kept_cids = np.array(kept_cids)
                pr_arr = {m: np.asarray(v) for m, v in pr_metrics.items()}
                cl_arr = {m: np.asarray(v) for m, v in cl_metrics.items()}
                n_enc = len(kept_keys)

                row = dict(
                    family=fam.name,
                    K=K,
                    split=sp,
                    placement="qDEIM",
                    n_enc=n_enc,
                    decoder_iter=op_iter,
                    ceiling_val_ssim_testa=ceiling_ssim_testa,
                    cv_field_ssim_train=cv_ssim if sp == SPLITS_EVAL[0] else None,
                )
                for m in pr_arr:
                    row[f"pr_{m}_mean"] = float(pr_arr[m].mean())
                    row[f"ceil_{m}_mean"] = float(cl_arr[m].mean())
                # the headline gap: pressure-recovered SSIM vs decode-ceiling SSIM
                row["ssim_gap_pr_vs_ceiling"] = float(cl_arr["ssim"].mean() - pr_arr["ssim"].mean())
                rows.append(row)

                per_enc_store[fam.name][f"{sp}_K{K}"] = dict(
                    keys=np.array(kept_keys),
                    case_ids=kept_cids,
                    pr_ssim=pr_arr["ssim"],
                    ceil_ssim=cl_arr["ssim"],
                    pr_mse=pr_arr["mse"],
                    pr_eps=pr_arr["eps_volume"],
                    pr_peak=pr_arr["peak_retention"],
                )
                if sp == "test_b" and K == (HEADLINE_K if HEADLINE_K in k_list else k_list[0]):
                    headline[fam.name] = dict(
                        keys=np.array(kept_keys),
                        case_ids=kept_cids,
                        pr_ssim=pr_arr["ssim"],
                        ceil_ssim=cl_arr["ssim"],
                        K=K,
                    )
                print(
                    f"    K={K:<2d} {sp}: pr_ssim={pr_arr['ssim'].mean():.3f} "
                    f"ceil_ssim={cl_arr['ssim'].mean():.3f} "
                    f"gap={row['ssim_gap_pr_vs_ceiling']:+.3f} "
                    f"pr_eps={pr_arr['eps_volume'].mean():.3f} "
                    f"pr_peak={pr_arr['peak_retention'].mean():.3f} (n={n_enc})",
                    flush=True,
                )
        del dec
        try:
            import torch as _t

            _t.cuda.empty_cache()
        except Exception:
            pass

    # ---- case-clustered bootstrap CI on the headline field SSIM per family ----
    headline_ci = {}
    for fam in FAMILIES:
        h = headline[fam.name]
        pr_ci = boot_metric_ci(h["pr_ssim"], h["case_ids"], n_boot=args.cv_boot, seed=0)
        ceil_ci = boot_metric_ci(h["ceil_ssim"], h["case_ids"], n_boot=args.cv_boot, seed=0)
        headline_ci[fam.name] = dict(
            K=int(h["K"]),
            pr_ssim=float(h["pr_ssim"].mean()),
            pr_ssim_ci=pr_ci,
            ceil_ssim=float(h["ceil_ssim"].mean()),
            ceil_ssim_ci=ceil_ci,
            ssim_gap=float(h["ceil_ssim"].mean() - h["pr_ssim"].mean()),
            n_enc=int(h["pr_ssim"].size),
        )

    # ---- paired family differences on the pressure-recovered field SSIM ----
    # JEPA vs each reconstructive family at headline K: one-sided sign test that
    # JEPA SSIM is HIGHER (better) + a case-clustered CI on the per-encounter SSIM
    # difference (JEPA - other). NOT case_permutation_p (the B6 lesson).
    paired = {}
    jep = FAMILIES[0].name
    hj = headline[jep]
    jpos = {k: i for i, k in enumerate(hj["keys"])}
    for fam in FAMILIES[1:]:
        ho = headline[fam.name]
        common = [k for k in ho["keys"] if k in jpos]
        ji = np.array([jpos[k] for k in common])
        oi = np.array([np.where(ho["keys"] == k)[0][0] for k in common])
        ssim_j = hj["pr_ssim"][ji]
        ssim_o = ho["pr_ssim"][oi]
        case_aligned = hj["case_ids"][ji]
        # sign test that JEPA SSIM > other (i.e. -ssim_j < -ssim_o => jepa better)
        k_better, n_eff, sign_p = stats_lib.sign_test_one_sided(-ssim_j, -ssim_o)
        delta = ssim_j - ssim_o  # >0 = jepa better field recovery
        cc = stats_lib.case_cluster_bootstrap(
            delta, case_aligned, n_boot=10000, rng=np.random.default_rng(0)
        )
        paired[f"{jep}_vs_{fam.name}"] = dict(
            K=int(hj["K"]),
            n_paired=len(common),
            sign_k=k_better,
            sign_n=n_eff,
            sign_p=sign_p,
            enc_mean_delta=cc["enc_mean"],
            enc_mean_ci=cc["enc_mean_ci"],
            case_mean_delta=cc["case_mean"],
            case_mean_ci=cc["case_mean_ci"],
        )

    # ---- honest ranking verdict: does the STATE winner also win the FIELD? ----
    Kh = HEADLINE_K if HEADLINE_K in k_list else k_list[0]
    pr_rank = sorted(
        ((fam.name, headline_ci[fam.name]["pr_ssim"]) for fam in FAMILIES),
        key=lambda kv: -kv[1],
    )
    ceil_rank = sorted(
        ((fam.name, headline_ci[fam.name]["ceil_ssim"]) for fam in FAMILIES),
        key=lambda kv: -kv[1],
    )
    state_order = ["jepa_tf_noc", "fukami", "pod"]  # known STATE-recovery ordering
    field_order = [n for n, _ in pr_rank]
    verdict = dict(
        state_order=state_order,
        field_pr_order=field_order,
        field_ceiling_order=[n for n, _ in ceil_rank],
        field_winner=field_order[0],
        state_winner_wins_field=bool(field_order[0] == state_order[0]),
        note=(
            "If field_pr_order != state_order the decoder-quality confound (visible "
            "in field_ceiling_order) has changed the ranking; the headline reports "
            "the field result AS MEASURED, with the ceiling gap, not the state order."
        ),
    )

    # ---------------------------------------------------------------- write results
    wall = time.time() - t_start
    results_json = dict(
        chain=(
            "sparse wall pressure (K qDEIM taps, W=%d) --KRR--> z_hat(impact+%d) "
            "--family decoder--> recovered omega field; scored vs DNS truth at "
            "impact+%d" % (WINDOW, READOUT_H, READOUT_H)
        ),
        ssim_convention="Wang K1=0.01 K2=0.03 on pipeline-normalized omega; "
        "L=%.2f (registry split_v2p1)" % L,
        decode_device=decode_on,
        gpu_name=gpu_name,
        K_list=k_list,
        headline_K=Kh,
        window_W=WINDOW,
        readout_H=READOUT_H,
        qdeim_picks={str(K): qdeim[K] for K in K_LIST},
        operating_iters={f.name: int(op["families"][f.op_family]["best_iter"]) for f in FAMILIES},
        bvae_absent="bvae has no trained decoder under outputs/session28/decode "
        "(absent from operating_points.json); it cannot enter the FIELD comparison. "
        "Its STATE recovery is reported in sensing_cf.json.",
        rows=rows,
        headline_ci=headline_ci,
        paired=paired,
        verdict=verdict,
        wall_seconds=wall,
    )

    def _jd(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    (OUT / "results.json").write_text(json.dumps(results_json, indent=2, default=_jd))

    save = dict(
        K_list=np.asarray(k_list, dtype=np.int64),
        families=np.asarray([f.name for f in FAMILIES]),
        colors=np.asarray([f.color for f in FAMILIES]),
        markers=np.asarray([f.marker for f in FAMILIES]),
        qdeim_headline=np.asarray(qdeim[Kh], dtype=np.int64),
        ssim_L=np.float64(L),
    )
    for fam in FAMILIES:
        for sp in SPLITS_EVAL:
            for K in k_list:
                d = per_enc_store[fam.name].get(f"{sp}_K{K}")
                if d is None:
                    continue
                pre = f"{fam.name}_{sp}_K{K}"
                save[f"{pre}_keys"] = d["keys"]
                save[f"{pre}_case_ids"] = d["case_ids"]
                save[f"{pre}_pr_ssim"] = d["pr_ssim"]
                save[f"{pre}_ceil_ssim"] = d["ceil_ssim"]
                save[f"{pre}_pr_mse"] = d["pr_mse"]
                save[f"{pre}_pr_eps"] = d["pr_eps"]
                save[f"{pre}_pr_peak"] = d["pr_peak"]
    np.savez(OUT / "results.npz", **save)

    emit_numbers_part(headline_ci, paired, rows, verdict, qdeim, L, k_list)

    try:
        make_field_figure(pipe, L, qdeim, op, device, k_list)
    except Exception as exc:  # figure is non-load-bearing; never fail the run
        print(f"[field_cf] WARNING figure skipped: {exc}", flush=True)

    write_readme(headline_ci, paired, verdict, L, decode_on, k_list)

    print(
        f"\n[field_cf] wrote {OUT/'results.json'}, .npz ({len(rows)} rows); "
        f"decode_on={decode_on}; wall {wall:.1f}s",
        flush=True,
    )
    print_report(headline_ci, paired, verdict, k_list)


def _cv_field_ssim(tr, Xtr, dec, pipe, L, device) -> float:
    """5-fold CASE-level CV field SSIM of the pressure->z->decode chain on TRAIN.

    Out-of-fold z_hat is decoded and scored against the DNS truth at the readout for
    the held-out TRAIN encounters. Decoding every train encounter at every K is the
    expensive part, so the CV is computed once per (family, K) and the mean over folds
    of the per-encounter SSIM is returned.
    """
    folds = make_case_folds(tr.case_ids, n_folds=5, seed=0)
    ssims = []
    for f in range(5):
        te = folds == f
        trn = ~te
        if te.sum() == 0 or trn.sum() == 0:
            continue
        pz = fit_krr(Xtr[trn], tr.z[trn])
        zhat = krr_pred(pz, Xtr[te])
        rec = _decode_latents(dec, zhat, pipe, device)
        idx = np.where(te)[0]
        for j, i in enumerate(idx):
            frame = int(tr.readout[i])
            dns_field = load_dns_truth_field(tr.case_ids[i], int(tr.enc_idx[i]), frame, pipe)
            ssims.append(ssim_global(pipe.normalize(rec[j]), pipe.normalize(dns_field), L))
    return float(np.mean(ssims)) if ssims else float("nan")


# --------------------------------------------------------------------- figure


def make_field_figure(pipe, L, qdeim, op, device, k_list) -> None:
    """Representative low-error encounter: DNS vs each family's pressure-recovered
    field vs each family's decode-ceiling field at impact+READOUT_H (Fig-3 convention).

    Per the project's representative-not-worst rule, the showcase encounter is the
    one whose JEPA pressure-recovered SSIM is closest to the JEPA test_b MEDIAN (a
    typical case, not the best and not the worst).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(REPO / "scripts" / "session21"))
    import figstyle as fs  # noqa: E402

    fs.use_style()
    Kh = HEADLINE_K if HEADLINE_K in k_list else k_list[0]
    taps = qdeim[Kh]

    # Re-derive the JEPA test_b pressure-recovered SSIM per encounter to pick the
    # representative (median) encounter, then render all families on that encounter.
    res = np.load(OUT / "results.npz", allow_pickle=True)
    jep = FAMILIES[0]
    pr = res[f"{jep.name}_test_b_K{Kh}_pr_ssim"]
    keys = res[f"{jep.name}_test_b_K{Kh}_keys"]
    rep = int(np.argsort(pr)[len(pr) // 2])  # median-SSIM encounter
    rep_key = str(keys[rep])
    rep_cid, rep_ei = rep_key.split("|")
    rep_ei = int(rep_ei)

    dns_metrics = np.load(DNS_PATH, allow_pickle=True)
    # readout frame for this encounter from the JEPA bundle
    jbundle = SC.load_family_split(jep.lat_tag, "test_b", dns_metrics)
    jkeys = enc_keys(jbundle.case_ids, jbundle.enc_idx)
    jidx = int(np.where(jkeys == rep_key)[0][0])
    frame = int(jbundle.readout[jidx])

    dns_field = load_dns_truth_field(rep_cid, rep_ei, frame, pipe)
    dns_n = pipe.normalize(dns_field)

    # one row per family: [DNS, pressure-recovered, decode-ceiling]; plus a header
    # DNS column shared. We use 3 columns x n_fam rows: col0 pressure-recovered,
    # col1 decode-ceiling, col2 DNS (repeated for visual reference).
    nfam = len(FAMILIES)
    fig, axes = plt.subplots(nfam, 3, figsize=fs.figure_size(1.0, aspect=0.30 * nfam))
    if nfam == 1:
        axes = axes[None, :]
    im = None
    for r, fam in enumerate(FAMILIES):
        op_iter = int(op["families"][fam.op_family]["best_iter"])
        tr = SC.load_family_split(fam.lat_tag, "train", dns_metrics)
        te = SC.load_family_split(fam.lat_tag, "test_b", dns_metrics)
        tkeys = enc_keys(te.case_ids, te.enc_idx)
        ti = int(np.where(tkeys == rep_key)[0][0])
        d = tr.z.shape[1]
        dec = _load_decoder(fam, op_iter, d, device)
        pz = fit_krr(feats(tr.X, taps), tr.z)
        zhat = krr_pred(pz, feats(te.X[ti : ti + 1], taps))
        rec_field = _decode_latents(dec, zhat, pipe, device)[0]
        ceil = load_ceiling_fields(fam, "test_b")
        cpos = {k: i for i, k in enumerate(ceil["keys"])}
        ceil_field = ceil["fields"][cpos[rep_key]]
        del dec

        rec_n = pipe.normalize(rec_field)
        ceil_n = pipe.normalize(ceil_field)
        ssim_pr = ssim_global(rec_n, dns_n, L)
        ssim_cl = ssim_global(ceil_n, dns_n, L)

        im = fs.vort_panel(axes[r, 0], rec_n)
        fs.vort_panel(axes[r, 1], ceil_n)
        fs.vort_panel(axes[r, 2], dns_n)
        axes[r, 0].set_ylabel(fam.name.replace("_", " "), fontsize=7)
        axes[r, 0].set_title(f"pressure-recovered (SSIM {ssim_pr:.2f})", fontsize=6.5)
        axes[r, 1].set_title(f"decode-ceiling (SSIM {ssim_cl:.2f})", fontsize=6.5)
        axes[r, 2].set_title("DNS truth", fontsize=6.5)

    fig.suptitle(
        f"flow-field recovery, K={Kh} qDEIM taps, impact+{READOUT_H}; "
        f"representative case {rep_cid} enc {rep_ei}",
        fontsize=7.5,
    )
    # Manual layout (the dedicated colorbar axes is not tight_layout-compatible).
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.04, top=0.90, wspace=0.08, hspace=0.30)
    if im is not None:
        cax = fig.add_axes([0.92, 0.25, 0.012, 0.5])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(r"$\omega_z$ (normalised)", fontsize=6.5)
    fig.savefig(OUT / "fig_field_recovery.pdf")
    fig.savefig(OUT / "fig_field_recovery.png", dpi=200)
    plt.close(fig)
    print(f"[field_cf] wrote {OUT/'fig_field_recovery.pdf'} (rep {rep_key})", flush=True)


# --------------------------------------------------------------------- numbers part


def emit_numbers_part(headline_ci, paired, rows, verdict, qdeim, L, k_list) -> None:
    """Write outputs/session28/numbers_parts/sensing_field_cf.json (eval_all format)."""
    Kh = HEADLINE_K if HEADLINE_K in k_list else k_list[0]
    macro = {f.name: f.macro for f in FAMILIES}
    nums: dict[str, dict] = {}

    def row_at(fam, sp, K, col):
        for r in rows:
            if r["family"] == fam and r["split"] == sp and r["K"] == K:
                return r[col]
        return None

    for fam in FAMILIES:
        h = headline_ci[fam.name]
        # pressure-recovered field SSIM at K=8 test_b (headline)
        nums[f"field_pr_ssim_K{Kh}_testb_{fam.name}"] = dict(
            macro=f"FieldPrSsim{macro[fam.name]}",
            value=float(h["pr_ssim"]),
            fmt="%.2f",
            ci_lo=float(h["pr_ssim_ci"][0]),
            ci_hi=float(h["pr_ssim_ci"][1]),
            n=int(h["n_enc"]),
            split="test_b",
            endpoint="deployment",
            probe="krr_rbf->decoder",
            observable="field_ssim",
            note=f"pressure({Kh} qDEIM taps,W={WINDOW})->z_hat(impact+{READOUT_H})->"
            f"decoder->omega; Wang SSIM L={L:.2f}; case-clustered CI; d=64 lead seed",
        )
        # decode-ceiling field SSIM at K=8 test_b (the confound reference)
        nums[f"field_ceiling_ssim_K{Kh}_testb_{fam.name}"] = dict(
            macro=f"FieldCeilSsim{macro[fam.name]}",
            value=float(h["ceil_ssim"]),
            fmt="%.2f",
            ci_lo=float(h["ceil_ssim_ci"][0]),
            ci_hi=float(h["ceil_ssim_ci"][1]),
            n=int(h["n_enc"]),
            split="test_b",
            endpoint="representational",
            observable="field_ssim",
            note=f"decode-ceiling: field from TRUE encoded z_dns(impact+{READOUT_H}) "
            f"(no pressure step); Wang SSIM L={L:.2f}; exposes the decoder-quality "
            f"confound vs the pressure-recovered SSIM",
        )
        # the explicit gap pressure-recovered vs decode-ceiling
        nums[f"field_ssim_gap_K{Kh}_testb_{fam.name}"] = dict(
            macro=f"FieldSsimGap{macro[fam.name]}",
            value=float(h["ssim_gap"]),
            fmt="%.2f",
            split="test_b",
            endpoint="deployment",
            observable="field_ssim_gap",
            note=f"decode-ceiling SSIM minus pressure-recovered SSIM at K={Kh}; the "
            f"part of the field error attributable to imperfect pressure->state "
            f"recovery (small gap => decoder-limited, large gap => state-limited)",
        )
        # raw-scale eps_volume + peak retention of the pressure-recovered field
        nums[f"field_pr_eps_K{Kh}_testb_{fam.name}"] = dict(
            macro=f"FieldPrEps{macro[fam.name]}",
            value=float(row_at(fam.name, "test_b", Kh, "pr_eps_volume_mean")),
            fmt="%.2f",
            split="test_b",
            endpoint="deployment",
            observable="field_eps_volume",
            note=f"volume L2 relative error of the pressure-recovered field at K={Kh}",
        )
        nums[f"field_pr_peak_K{Kh}_testb_{fam.name}"] = dict(
            macro=f"FieldPrPeak{macro[fam.name]}",
            value=float(row_at(fam.name, "test_b", Kh, "pr_peak_retention_mean")),
            fmt="%.2f",
            split="test_b",
            endpoint="deployment",
            observable="peak_vorticity_retention",
            note=f"recovered/DNS peak |omega| of the pressure-recovered field at K={Kh}",
        )

    # paired JEPA-vs-recon sign test (worst-case = max p over recon families)
    if paired:
        worst = max(paired.items(), key=lambda kv: kv[1]["sign_p"])
        nums["field_pr_ssim_jepa_vs_recon_signp"] = dict(
            macro="FieldPrSsimSignP",
            value=float(worst[1]["sign_p"]),
            fmt="%.3f",
            split="test_b",
            endpoint="deployment",
            observable="field_ssim",
            note=f"max over recon families of one-sided sign-test p (JEPA "
            f"pressure-recovered field SSIM > recon), {worst[0]}, K={Kh}",
        )
    # verdict: does the field winner match the state winner (1) or not (0)?
    nums["field_state_winner_matches"] = dict(
        macro="FieldStateWinnerMatches",
        value=1 if verdict["state_winner_wins_field"] else 0,
        fmt="%d",
        split="test_b",
        endpoint="deployment",
        observable="ranking_agreement",
        note=f"1 iff the STATE-recovery winner ({verdict['state_order'][0]}) is also "
        f"the FIELD-recovery winner; field order={verdict['field_pr_order']}, "
        f"ceiling order={verdict['field_ceiling_order']}",
    )
    (PARTS / "sensing_field_cf.json").write_text(
        json.dumps({"part": "sensing_field_cf", "numbers": nums}, indent=2)
    )
    print(f"[field_cf] wrote {PARTS/'sensing_field_cf.json'} ({len(nums)} numbers)", flush=True)


# --------------------------------------------------------------------- README + report


def write_readme(headline_ci, paired, verdict, L, decode_on, k_list) -> None:
    Kh = HEADLINE_K if HEADLINE_K in k_list else k_list[0]
    lines = []
    lines.append("# Sparse wall-pressure FLOW-FIELD recovery comparison (v2.1)\n")
    lines.append(
        "Author reframe 2026-06-13: the sensing track is a cross-method comparison of\n"
        "which representation, estimated from sparse wall pressure, best recovers the\n"
        "STATE and the FLOW FIELD. The earlier pressure -> wake-enstrophy claim is\n"
        "dropped. This file is the PHYSICAL-SPACE companion to the STATE recovery in\n"
        "`outputs/session28/sensing/` (state R^2 at K=8 test_b: jepa 0.78 > fukami 0.66\n"
        "> bvae 0.51 > pod 0.34).\n"
    )
    lines.append("## The chain\n")
    lines.append(
        f"Per family, per encounter on test_b and test_c:\n\n"
        f"    sparse wall pressure (K qDEIM taps over the pre-impact window W={WINDOW})\n"
        f"      --KRR(RBF)-->  estimated latent z_hat(impact+{READOUT_H})\n"
        f"      --family decoder (frozen operating point)-->  recovered omega field\n"
        f"    scored against the DNS truth field at impact+{READOUT_H}.\n\n"
        f"The KRR map, qDEIM tap picks, pressure window W and readout horizon are reused\n"
        f"verbatim from `scripts/session28/sensing_cf.py`. The decoders are loaded at\n"
        f"their frozen operating iters from `decode_operating_points.py`.\n"
    )
    lines.append("## Metrics\n")
    lines.append(
        f"At K=2/4/8/16 on test_b and test_c, per family: field SSIM (Wang K1=0.01\n"
        f"K2=0.03 on pipeline-normalized omega, data range L={L:.2f} from the\n"
        f"`configs/ssim_data_range.json` registry), volume L2 relative error\n"
        f"(eps_volume), and peak-vorticity retention (recovered/DNS peak |omega|), of\n"
        f"the pressure-recovered field vs DNS. MSE/eps/peak are on the raw scale; SSIM\n"
        f"normalizes both sides into the pipeline-normalized units of L.\n"
    )
    lines.append("## Decode-ceiling reference (the decoder-quality confound)\n")
    lines.append(
        "For every family we ALSO score the field decoded from the TRUE encoded latent\n"
        "z_dns(impact+%d) (no pressure step). The gap `decode-ceiling SSIM minus\n"
        "pressure-recovered SSIM` isolates how much field error is the pressure->state\n"
        "recovery versus the decoder itself. POD's LINEAR decoder has the best raw\n"
        "ceiling (test_a SSIM ~0.68) while Fukami's is ~0.50, so a field-from-pressure\n"
        "ranking can be partly decoder quality, not state recovery. Both numbers are\n"
        "reported side by side; the ceiling is never allowed to silently flip the\n"
        "ranking.\n" % READOUT_H
    )
    lines.append("## Headline (test_b, K=%d)\n" % Kh)
    lines.append(
        "| family | pressure-recovered SSIM [95% CI] | decode-ceiling SSIM [95% CI] | gap |"
    )
    lines.append("| --- | --- | --- | --- |")
    for fam in FAMILIES:
        h = headline_ci[fam.name]
        lines.append(
            "| %s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %+.3f |"
            % (
                fam.name,
                h["pr_ssim"],
                h["pr_ssim_ci"][0],
                h["pr_ssim_ci"][1],
                h["ceil_ssim"],
                h["ceil_ssim_ci"][0],
                h["ceil_ssim_ci"][1],
                h["ssim_gap"],
            )
        )
    lines.append("")
    lines.append("## Which method recovers the FIELD best (with the ceiling gap)\n")
    fw = verdict["field_pr_order"][0]
    lines.append(
        "Pressure-recovered field SSIM ranking (best first): %s.\n"
        "Decode-ceiling SSIM ranking (best first): %s.\n"
        "STATE-recovery ranking (sensing_cf, for reference): %s.\n\n"
        % (
            ", ".join(verdict["field_pr_order"]),
            ", ".join(verdict["field_ceiling_order"]),
            ", ".join(verdict["state_order"]),
        )
    )
    if verdict["state_winner_wins_field"]:
        lines.append(
            "VERDICT: the predictive latent that wins STATE recovery (%s) ALSO wins\n"
            "FIELD recovery from sparse pressure (%s). The decoder-quality confound\n"
            "(visible in the ceiling order) does NOT flip the headline ranking.\n"
            % (verdict["state_order"][0], fw)
        )
    else:
        lines.append(
            "VERDICT (adversarial, reported AS MEASURED): the FIELD winner is %s, which\n"
            "is NOT the STATE winner %s. The decode-ceiling order (%s) shows this is\n"
            "driven at least in part by decoder quality, not state recovery: the family\n"
            "with the stronger raw decoder recovers a more SSIM-similar field even when\n"
            "its pressure->state recovery is weaker. We report the field result as\n"
            "measured and do NOT force the state ordering onto it; the gap column makes\n"
            "the confound explicit.\n"
            % (fw, verdict["state_order"][0], ", ".join(verdict["field_ceiling_order"]))
        )
    lines.append("")
    lines.append("## Statistics\n")
    lines.append(
        "5-fold CASE-level CV on the pressure->z map (sensing_cf folds; no case\n"
        "straddles a fold); case-clustered bootstrap CI on the headline field SSIM per\n"
        "family; paired JEPA-vs-reconstructive differences via a one-sided sign test +\n"
        "a case-clustered CI on the per-encounter SSIM difference (NOT\n"
        "case_permutation_p, the B6 lesson).\n"
    )
    lines.append("## Notes\n")
    lines.append(
        "- bvae has no trained decoder under `outputs/session28/decode`, so it cannot\n"
        "  enter the FIELD comparison; its STATE recovery is in `sensing_cf.json`.\n"
        "- Decoder forward ran on: %s (RTX 6000 gpu1 if available, else CPU; the L40S\n"
        "  cards are forbidden).\n"
        "- Outputs: `results.{npz,json}`, `fig_field_recovery.{pdf,png}`, and the\n"
        "  numbers part `outputs/session28/numbers_parts/sensing_field_cf.json`.\n" % decode_on
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"[field_cf] wrote {OUT/'README.md'}", flush=True)


def print_report(headline_ci, paired, verdict, k_list) -> None:
    Kh = HEADLINE_K if HEADLINE_K in k_list else k_list[0]
    print(f"\n===== FIELD RECOVERY HEADLINE (test_b, K={Kh}) =====")
    print(f"{'family':14s} {'pr_SSIM':>9s} {'ceil_SSIM':>10s} {'gap':>7s}   CI(pr)")
    for fam in FAMILIES:
        h = headline_ci[fam.name]
        print(
            f"{fam.name:14s} {h['pr_ssim']:9.3f} {h['ceil_ssim']:10.3f} "
            f"{h['ssim_gap']:+7.3f}   [{h['pr_ssim_ci'][0]:.3f},{h['pr_ssim_ci'][1]:.3f}]"
        )
    print(f"\nfield (pressure-recovered) order: {verdict['field_pr_order']}")
    print(f"field (decode-ceiling) order    : {verdict['field_ceiling_order']}")
    print(f"state-recovery order (reference): {verdict['state_order']}")
    print(
        f"=> STATE winner also wins FIELD: {verdict['state_winner_wins_field']} "
        f"(field winner = {verdict['field_pr_order'][0]})"
    )
    print("\npaired JEPA vs reconstructive (sign test + case-clustered CI on SSIM delta):")
    for k, v in paired.items():
        print(
            f"  {k:28s} sign {v['sign_k']}/{v['sign_n']} p={v['sign_p']:.3f} "
            f"case-delta={v['case_mean_delta']:+.3f} "
            f"CI[{v['case_mean_ci'][0]:+.3f},{v['case_mean_ci'][1]:+.3f}]"
        )


if __name__ == "__main__":
    main()

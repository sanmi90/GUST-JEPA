"""Precompute per-encounter near-body lift-element targets (Session 34, Track C).

For every encounter in the v2p2 cache, reads the RAW mid-span velocity and
vorticity (``/u[..., 16, 0:2]``, ``/curlU[..., 16, 2]``) over the encounter's
frame window, forms the Chang lift-element field

    e = omega_z_std * (-v dphi_L/dx + u dphi_L/dy),   omega_z_std = -curlU_z,

band-weights and scales it (``field = band * e / E_SCALE``), and writes the
80-D ``nearbody_lift_element`` target per frame::

    ${VORTEX_JEPA_CACHE}/v2p2/nearbody_observables/{case_id}/encounter_{k:02d}.h5
    ${VORTEX_JEPA_CACHE}/v2p2/nearbody_observables/_train_stats.json
    ${VORTEX_JEPA_CACHE}/v2p2/nearbody_observables/_manifest.json

Standardization stats are train-pool only (split == 'train' AND encounter in
train_encounter_indices), exactly like the wake observables cache.

QC gate (D254): per encounter, the best lagged correlation (|lag| <= 25 frames)
between the band-integrated element and the stored C_L(t). The campaign gate
requires the median over gust train encounters (|G| > 0) to clear
--qc-corr-threshold (default 0.4); on failure the script exits nonzero and the
training queue must NOT be launched. Session-34 pre-study on 4 train
encounters: corr 0.56-0.79 at |lag| <= 7 for gusts; the no-gust Baseline
shedding cycle is phase-ambiguous (anti-phase corr) and is excluded from the
gate by the |G| > 0 filter.

D254 evidence: for --d254-cases, also computes the |omega|-proxy target
(band * pipeline-normalized omega from the encounter cache) and records the
per-frame cosine similarity between the proxy and Chang 64-D patch blocks.

Usage::

    taskset -c 0-15 python scripts/session34/precompute_nearbody_observables.py \\
        --partition v2p2 --split configs/splits/split_v2p2.json \\
        --omega-pipeline-manifest outputs/data_pipeline/v2p2/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.lift_element import (  # noqa: E402
    ALPHA_DEG,
    DEFAULT_ADJACENT_MASK_PATH,
    DEFAULT_PHI_L_PATH,
    DX,
    DY,
    E_SCALE,
    LIFT_DIR,
    MID_SPAN_INDEX,
    OMEGA_STORED_SIGN,
    build_nearbody_band,
    lift_element_field,
    load_midspan_solid,
    load_phi_L,
    save_phi_L,
    solve_phi_L,
)
from src.data.nearbody_observables import (  # noqa: E402
    compute_standardization_from_targets,
    nearbody_lift_element_target,
    nearbody_patch_signed_target,
)

PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))

MODE = "nearbody_lift_element"
D254_DEFAULT_CASES = ("Baseline", "G+3.00_D1.50_Y+0.20", "G-2.00_D1.00_Y+0.40")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute near-body lift-element observables")
    p.add_argument("--partition", default="v2p2")
    p.add_argument("--split", default="configs/splits/split_v2p2.json")
    p.add_argument("--omega-pipeline-manifest",
                   default="outputs/data_pipeline/v2p2/manifest.json",
                   help="Used only for the D254 proxy comparison.")
    p.add_argument("--delta-n", type=float, default=0.3)
    p.add_argument("--qc-corr-threshold", type=float, default=0.4,
                   help="Gate on median lagged corr(band e integral, C_L) over gust "
                        "train encounters.")
    p.add_argument("--d254-cases", nargs="+", default=list(D254_DEFAULT_CASES))
    p.add_argument("--force", action="store_true")
    p.add_argument("--cases", nargs="+", default=None)
    return p.parse_args()


def gather_cases(split_path: Path, cases_filter: list[str] | None) -> dict:
    with open(split_path) as f:
        manifest = json.load(f)
    cases = manifest["cases"]
    if cases_filter is not None:
        cases = {cid: c for cid, c in cases.items() if cid in cases_filter}
    return cases


def lagged_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 25) -> tuple[float, int]:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    best, best_lag = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = float(np.dot(a[lag:], b[: len(b) - lag]) / den)
        else:
            c = float(np.dot(a[:lag], b[-lag:]) / den)
        if abs(c) > abs(best):
            best, best_lag = c, lag
    return best, best_lag


def main() -> None:
    args = parse_args()
    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = REPO / split_path

    # --- geometry + phi_L (solve once, persist with provenance) ---
    solid = load_midspan_solid(PREVENT / "data/raw/periodic/Baseline.h5")
    adjacent_path = REPO / DEFAULT_ADJACENT_MASK_PATH
    adjacent = np.load(adjacent_path)
    band = build_nearbody_band(adjacent, delta_n=args.delta_n)
    phi_path = REPO / DEFAULT_PHI_L_PATH
    if phi_path.exists() and not args.force:
        phi = load_phi_L(phi_path)
        print(f"[nearbody] loaded phi_L from {phi_path} "
              f"(residual={phi['residual_linf']:.1e})", flush=True)
    else:
        res = solve_phi_L(solid, lift_dir=LIFT_DIR)
        save_phi_L(res, solid, path=phi_path, lift_dir=LIFT_DIR)
        phi = load_phi_L(phi_path)
        print(f"[nearbody] solved+saved phi_L (residual={phi['residual_linf']:.1e})",
              flush=True)
    grad_x = phi["grad_x"].astype(np.float32)
    grad_y = phi["grad_y"].astype(np.float32)

    cases = gather_cases(split_path, args.cases)
    out_root = CACHE / args.partition / "nearbody_observables"
    out_root.mkdir(parents=True, exist_ok=True)

    # D254 proxy comparison needs the omega pipeline (cached omega, normalized).
    omega_pipeline = None
    if args.d254_cases:
        from src.data.omega_pipeline import OmegaPipeline
        mpath = Path(args.omega_pipeline_manifest)
        if not mpath.is_absolute():
            mpath = REPO / mpath
        omega_pipeline = OmegaPipeline.from_manifest(mpath)

    train_targets: list[np.ndarray] = []
    qc_records: list[dict] = []
    d254_records: list[dict] = []
    n_written = 0
    t0 = time.time()

    for ci, (cid, case) in enumerate(sorted(cases.items())):
        raw_path = PREVENT / case["relative_path"]
        n_enc = int(case["n_encounters_full"])
        enc_meta = []
        for k in range(n_enc):
            src = CACHE / args.partition / cid / f"encounter_{k:02d}.h5"
            if not src.exists():
                continue
            with h5py.File(src, "r") as g:
                enc_meta.append({
                    "k": k,
                    "frame_start": int(g.attrs["frame_start"]),
                    "n_frames": int(g.attrs["n_frames"]),
                    "cl": np.asarray(g["C_L"], dtype=np.float64),
                })
        if not enc_meta:
            continue
        out_dir = out_root / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(raw_path, "r") as rf:
            for e in enc_meta:
                k, f0, nfr = e["k"], e["frame_start"], e["n_frames"]
                in_train = (case["split"] == "train"
                            and k in case["train_encounter_indices"])
                uv = rf["u"][f0:f0 + nfr, :, :, MID_SPAN_INDEX, 0:2]
                om_stored = rf["curlU"][f0:f0 + nfr, :, :, MID_SPAN_INDEX, 2]
                om_std = OMEGA_STORED_SIGN * np.nan_to_num(
                    np.asarray(om_stored, dtype=np.float32), nan=0.0)
                el = lift_element_field(om_std, uv[..., 0], uv[..., 1], grad_x, grad_y)
                field = el * band / E_SCALE
                tgt = nearbody_lift_element_target(torch.from_numpy(field)).numpy()
                e_int = (el * band).sum(axis=(1, 2)) * DX * DY
                corr, lag = lagged_corr(e_int, e["cl"])
                qc_records.append({
                    "case_id": cid, "encounter_index": k, "split": case["split"],
                    "G": float(case["G"]), "in_train_pool": bool(in_train),
                    "lagged_corr": corr, "lag_frames": lag,
                })
                if in_train:
                    train_targets.append(tgt.astype(np.float32))
                if cid in (args.d254_cases or []) and omega_pipeline is not None:
                    with h5py.File(CACHE / args.partition / cid /
                                   f"encounter_{k:02d}.h5", "r") as g:
                        omega_raw = np.asarray(g["omega_z"], dtype=np.float32)
                    omega_clean = omega_pipeline.preprocess_raw(omega_raw, cid, k)
                    omega_norm = omega_pipeline.normalize(
                        torch.from_numpy(omega_clean)).numpy()
                    proxy_field = torch.from_numpy(omega_norm * band)
                    p64 = nearbody_patch_signed_target(proxy_field).numpy()
                    c64 = tgt[:, :64]
                    num = (p64 * c64).sum(axis=1)
                    den = (np.linalg.norm(p64, axis=1)
                           * np.linalg.norm(c64, axis=1) + 1e-12)
                    d254_records.append({
                        "case_id": cid, "encounter_index": k,
                        "cosine_mean": float((num / den).mean()),
                        "cosine_min": float((num / den).min()),
                    })
                out_path = out_dir / f"encounter_{k:02d}.h5"
                if out_path.exists() and not args.force:
                    continue
                with h5py.File(out_path, "w") as out_g:
                    out_g.attrs["case_id"] = cid
                    out_g.attrs["encounter_index"] = int(k)
                    out_g.attrs["split"] = case["split"]
                    out_g.attrs["in_train_pool"] = bool(in_train)
                    out_g.attrs["preprocessing_version"] = "nearbody_observables_v1"
                    out_g.create_dataset(MODE, data=tgt, dtype="float32")
                    out_g.create_dataset("band_e_integral", data=e_int, dtype="float32")
                n_written += 1
        if (ci + 1) % 10 == 0 or (ci + 1) == len(cases):
            print(f"[nearbody] {ci + 1}/{len(cases)} cases "
                  f"({n_written} encounters) in {time.time() - t0:.1f}s", flush=True)

    # --- train-pool standardization stats ---
    n_train = sum(t.shape[0] for t in train_targets)
    print(f"[nearbody] pooling {n_train} train frames "
          f"({len(train_targets)} encounters) for standardization", flush=True)
    stats = compute_standardization_from_targets(train_targets, mode=MODE)
    live_dims = int((stats.std > 1e-6).sum())
    stats_path = out_root / "_train_stats.json"
    with open(stats_path, "w") as f:
        json.dump({MODE: stats.to_dict()}, f, indent=2)
    print(f"[nearbody] wrote train stats to {stats_path} "
          f"({live_dims}/80 dims with std > 1e-6)", flush=True)

    # --- QC gate ---
    gust_train = [r for r in qc_records if r["in_train_pool"] and abs(r["G"]) > 0]
    med = float(np.median([abs(r["lagged_corr"]) for r in gust_train]))
    gate_pass = med >= args.qc_corr_threshold
    print(f"[nearbody] QC gate: median |lagged corr| over {len(gust_train)} gust "
          f"train encounters = {med:.3f} (threshold {args.qc_corr_threshold}) "
          f"-> {'PASS' if gate_pass else 'FAIL'}", flush=True)
    if d254_records:
        cmean = float(np.mean([r["cosine_mean"] for r in d254_records]))
        print(f"[nearbody] D254 proxy-vs-Chang 64D patch cosine: mean={cmean:.3f} "
              f"over {len(d254_records)} encounters", flush=True)

    solid_sha = hashlib.sha256(np.ascontiguousarray(solid.astype(np.uint8))).hexdigest()
    adj_sha = hashlib.sha256(np.ascontiguousarray(adjacent.astype(np.uint8))).hexdigest()
    manifest_out = out_root / "_manifest.json"
    with open(manifest_out, "w") as f:
        json.dump({
            "partition": args.partition,
            "mode": MODE,
            "mode_dim": 80,
            "delta_n": args.delta_n,
            "e_scale": E_SCALE,
            "alpha_deg": ALPHA_DEG,
            "lift_dir": list(LIFT_DIR),
            "omega_stored_sign": OMEGA_STORED_SIGN,
            "mid_span_index": MID_SPAN_INDEX,
            "phi_L_path": str(REPO / DEFAULT_PHI_L_PATH),
            "phi_L_residual_linf": phi["residual_linf"],
            "solid_sha256": solid_sha,
            "adjacent_mask_sha256": adj_sha,
            "n_encounters_written": n_written,
            "n_train_frames_pooled": n_train,
            "live_std_dims": live_dims,
            "qc": {
                "threshold": args.qc_corr_threshold,
                "median_abs_lagged_corr_gust_train": med,
                "gate_pass": gate_pass,
                "records": qc_records,
            },
            "d254_proxy_vs_chang": d254_records,
            "split": str(split_path),
        }, f, indent=2)
    print(f"[nearbody] wrote manifest to {manifest_out}", flush=True)
    print(f"[nearbody] done: {n_written} encounters in {time.time() - t0:.1f}s", flush=True)
    if not gate_pass:
        print("[nearbody] QC GATE FAILED -- do NOT launch the training queue.",
              flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()

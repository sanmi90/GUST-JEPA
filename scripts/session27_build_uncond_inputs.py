#!/usr/bin/env python3
"""
session27_build_uncond_inputs.py
================================

Build the unconditioned-encoder analogues of the two input artifacts the
session25 coordinate suite consumes, by swapping ONLY the latent tensor while
keeping every DNS-derived physical descriptor identical:

  (1) per_frame_targets/{split}.npz  ->  outputs/session27/causal_noc_<model>/
        per_frame_targets/{split}.npz
      Same keys as the production file, but z_full is replaced by the
      unconditioned latents (outputs/session27/latents_own_jepa_noc_<model>/),
      aligned by (case_id, encounter). All physical columns (C_L, C_D,
      wake_enstrophy, circulation_*, centroid_*, etc.) are copied verbatim from
      the production per_frame_targets file because they are computed from the
      DNS field and are independent of the encoder.

  (2) exp_b1/latents_jepa_d64/{split}.npz  ->  outputs/session27/causal_noc_<model>/
        latents_jepa_noc/{split}.npz
      Same keys as the production B1 export, z + z_full replaced by the
      unconditioned latents (z = z_full at the impact frame).

This is a faithful repoint: the session25 scripts are copied verbatim except
for the input path constants, so they operate on the unconditioned latents with
the paper's exact methodology.

NOTHING outside outputs/session27/ is written. Production artifacts are read
read-only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PF_PROD = REPO / "outputs/session16/exp2/per_frame_targets/{}.npz"
B1_PROD = REPO / "outputs/session18/exp_b1/latents_jepa_d64/{}.npz"
UNC_LAT = REPO / "outputs/session27/latents_own_jepa_noc_{model}/{split}.npz"

SPLITS = ("train", "test_b", "test_c")


def _keys(cids, encs):
    return [(str(c), int(e)) for c, e in zip(cids, encs)]


def load_uncond(model: str, split: str):
    d = np.load(str(UNC_LAT).format(model=model, split=split), allow_pickle=True)
    cids = d["case_ids"] if "case_ids" in d.files else d["case_id"]
    encs = d["encounter_indices"] if "encounter_indices" in d.files else d["encounter_index"]
    zf = np.asarray(d["z_full"], np.float32)
    imp = np.asarray(d["impact_frame"], int)
    return _keys(cids, encs), zf, imp


def swap_z_full(prod_blob, prod_keys, unc_keys, unc_zfull):
    """Return a z_full array reordered to match prod row order, sourced from uncond."""
    row_of = {k: i for i, k in enumerate(unc_keys)}
    T = prod_blob["z_full"].shape[1]
    d = unc_zfull.shape[2]
    out = np.full((len(prod_keys), T, d), np.nan, np.float32)
    n_missing = 0
    for i, k in enumerate(prod_keys):
        if k in row_of:
            out[i] = unc_zfull[row_of[k]]
        else:
            n_missing += 1
    return out, n_missing


def build_per_frame(model: str, out_root: Path):
    pf_dir = out_root / "per_frame_targets"
    pf_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        prod = np.load(str(PF_PROD).format(split), allow_pickle=True)
        prod_keys = _keys(prod["case_id"], prod["encounter_index"])
        try:
            unc_keys, unc_zf, _ = load_uncond(model, split)
        except FileNotFoundError:
            print(f"[per_frame] SKIP {model} {split}: uncond latents missing")
            continue
        # the uncond latent dim may differ from prod (both 64 here)
        new_zfull, nmiss = swap_z_full(prod, prod_keys, unc_keys, unc_zf)
        d_out = {k: prod[k] for k in prod.files}
        d_out["z_full"] = new_zfull
        # sanity: keys/impact identical so descriptors stay aligned to z_full
        np.savez(pf_dir / f"{split}.npz", **d_out)
        print(f"[per_frame] {model} {split}: wrote {len(prod_keys)} rows "
              f"(d={new_zfull.shape[2]}, {nmiss} uncond-missing)")


def build_b1(model: str, out_root: Path):
    b1_dir = out_root / "latents_jepa_noc"
    b1_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        prod = np.load(str(B1_PROD).format(split), allow_pickle=True)
        prod_keys = _keys(prod["case_id"], prod["encounter_index"])
        try:
            unc_keys, unc_zf, _ = load_uncond(model, split)
        except FileNotFoundError:
            print(f"[b1] SKIP {model} {split}: uncond latents missing")
            continue
        new_zfull, nmiss = swap_z_full(prod, prod_keys, unc_keys, unc_zf)
        imp = np.asarray(prod["impact_frame"], int)
        new_z = np.stack([new_zfull[i, imp[i]] for i in range(len(prod_keys))]).astype(np.float32)
        d_out = {k: prod[k] for k in prod.files}
        d_out["z_full"] = new_zfull
        d_out["z"] = new_z
        np.savez(b1_dir / f"{split}.npz", **d_out)
        print(f"[b1] {model} {split}: wrote {len(prod_keys)} rows "
              f"(d={new_zfull.shape[2]}, {nmiss} uncond-missing)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tf", choices=["tf", "lstm"])
    args = ap.parse_args()
    out_root = REPO / f"outputs/session27/causal_noc_{args.model}"
    out_root.mkdir(parents=True, exist_ok=True)
    build_per_frame(args.model, out_root)
    build_b1(args.model, out_root)
    print(f"[done] uncond inputs for model={args.model} under {out_root}")


if __name__ == "__main__":
    main()

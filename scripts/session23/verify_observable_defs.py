"""Session 23 Track B: verify the six closure-observable equations against the code.

Re-implements the wake enstrophy, signed circulations, and wall-normal impulse from the
written Section 2.2 equations, INDEPENDENTLY of the production code, and checks that they
reproduce the stored targets in outputs/session17/exp2/dns_physical_metrics.npz (the same
file the JEPA forward-closure numbers are computed against) to float tolerance on sampled
frames. Also reconciles the 84-case / 320-encounter counts against split_v2.json.

The field is prepared exactly as the production code does (OmegaPipeline.preprocess_raw:
solid + adjacent-cell mask, per-encounter p99.99 clip, raw scale); only the integral
formulas under test are re-implemented here. CPU-only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

REPO = Path("/home/carlos/GUST-JEPA")
sys.path.insert(0, str(REPO))

# Load the production observable module by file path (scripts/ is not a package).
_spec = importlib.util.spec_from_file_location(
    "exp2_dns", REPO / "scripts" / "session17" / "exp2_dns_physical_metrics.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402

# Geometry and wake window EXACTLY as written in Section 2.2 (Eq. block).
DX, DY = 6.0 / 192, 3.0 / 96
XG = np.linspace(-1.5, 4.5, 192).astype(np.float32)
YG = np.linspace(-1.5, 1.5, 96).astype(np.float32)
WAKE_X = (0.5, 4.0)
WAKE_YMAX = 1.0
THR = 1.0  # |omega| > 1 threshold for the signed circulations


def observables_from_equations(omega: np.ndarray) -> dict:
    """Independent re-implementation of the Section 2.2 equations.

    omega: (T, 192, 96) masked, clipped, raw-scale spanwise vorticity.
    """
    omega = omega.astype(np.float32)
    xx = XG[:, None]
    yy = YG[None, :]
    wake = ((xx >= WAKE_X[0]) & (xx <= WAKE_X[1]) & (np.abs(yy) <= WAKE_YMAX))[None]
    ow = omega * wake
    enstrophy = (ow**2).sum(axis=(1, 2)) * DX * DY
    gamma_pos = np.where(ow > THR, ow, np.float32(0.0)).sum(axis=(1, 2)) * DX * DY
    gamma_neg = np.where(ow < -THR, ow, np.float32(0.0)).sum(axis=(1, 2)) * DX * DY
    impulse_y = np.einsum("tij,i->t", omega, XG) * DX * DY  # full field, +int x*omega dA
    return {
        "wake_enstrophy": enstrophy,
        "circulation_pos": gamma_pos,
        "circulation_neg": gamma_neg,
        "I_y": impulse_y,
    }


def main() -> None:
    pipe = OmegaPipeline.from_manifest(mod.OMEGA_MANIFEST)
    tgt = np.load(mod.OUT / "dns_physical_metrics.npz", allow_pickle=True)

    obs_names = ["wake_enstrophy", "circulation_pos", "circulation_neg", "I_y"]
    print("=" * 70)
    print("Track B: verify Section 2.2 observable equations vs stored targets")
    print("=" * 70)

    n_sample = 6
    for split in ("test_b", "train"):
        encs = mod.gather_split_encounters(split)
        cids = tgt[f"{split}_case_id"].astype(str)
        ks = tgt[f"{split}_encounter_index"].astype(int)
        worst = {o: 0.0 for o in obs_names}
        rel = {o: 0.0 for o in obs_names}
        n_checked = 0
        for e in encs[:n_sample]:
            with h5py.File(e["path"], "r") as f:
                om = np.asarray(f["omega_z"], dtype=np.float32)
            clean = pipe.preprocess_raw(om, e["case_id"], e["k"])
            mine = observables_from_equations(clean)
            idx = np.where((cids == e["case_id"]) & (ks == e["k"]))[0]
            if len(idx) == 0:
                continue
            j = int(idx[0])
            n_checked += 1
            for o in obs_names:
                stored = np.asarray(tgt[f"{split}_{o}"][j], dtype=np.float64)
                T = min(len(stored), len(mine[o]))
                a = stored[:T]
                b = mine[o][:T].astype(np.float64)
                d = float(np.nanmax(np.abs(a - b)))
                scale = float(np.nanmax(np.abs(a))) + 1e-12
                worst[o] = max(worst[o], d)
                rel[o] = max(rel[o], d / scale)
        print(f"\n[{split}] checked {n_checked} encounters x 120 frames")
        for o in obs_names:
            print(f"  {o:18s} max|abs diff| = {worst[o]:.3e}   max rel = {rel[o]:.3e}")
        ok = all(rel[o] < 1e-4 for o in obs_names)
        print(f"  -> {'PASS' if ok else 'CHECK'} (relative tolerance 1e-4, float32 field)")

    # ---- count reconciliation: 84 cases -> 320 encounter windows ----
    manifest = json.load(open(mod.SPLIT_MANIFEST))
    cases = manifest["cases"]
    split_cases = Counter(c["split"] for c in cases.values())
    per_split_enc = {s: len(mod.gather_split_encounters(s)) for s in ("train", "test_a", "test_b", "test_c")}
    total_enc = sum(per_split_enc.values())
    print("\n" + "=" * 70)
    print("Count reconciliation (split_v2.json)")
    print("=" * 70)
    print(f"  distinct (G,D,Y) cases : {len(cases)}")
    print(f"  cases per split        : {dict(split_cases)}")
    print(f"  encounters per split   : {per_split_enc}")
    print(f"  total encounter windows: {total_enc}  "
          f"(= {' + '.join(str(per_split_enc[s]) for s in ('train','test_a','test_b','test_c'))})")


if __name__ == "__main__":
    main()

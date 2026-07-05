"""d-parameterized 3-seed forecast band: JEPA-own vs regAE-matched, wake-enstrophy
forecast R^2 vs horizon. Usage: m_seed_forecast_band.py [D]  (default 64).
Auto-detects which seeds have rollouts present. Reuses m_lowd_forecast's probe.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m_lowd_forecast as M  # noqa: E402

ROLL = M.ROLL
LAT = M.LAT
D = int(sys.argv[1]) if len(sys.argv) > 1 else 64
OBS = "wake_enstrophy"
HS = (1, 2, 4, 8, 12, 16)
SEEDS = [0, 1, 2, 42]


def jepa_tags(s):
    return f"jepa_tf_noc_d{D}_s{s}", f"jepa_own_d{D}_s{s}"


def regae_tags(s):
    if D == 64:
        lat = f"regae/cnn_vit_s{s}"
        rk = "regae_d64" if s == 0 and not (ROLL / f"regae_d64_s{s}" / "test_b.npz").exists() else f"regae_d64_s{s}"
    else:
        lat = f"regae/cnn_vit_d{D}_s{s}"
        rk = f"regae_d{D}_s{s}"
    return lat, rk


def band(name, tagfn, otr, itr, otb, itb):
    curves = []
    used = []
    for s in SEEDS:
        lat, rk = tagfn(s)
        if not (ROLL / rk / "test_b.npz").exists() or not (LAT / lat / "train.npz").exists():
            continue
        gs = M.fit_probe(lat, OBS, otr, itr)
        c = M.forecast_curve(rk, lat, OBS, gs, otb, itb)
        if c is not None:
            curves.append(c); used.append(s)
    if not curves:
        print(f"  {name:13s}: (no rollouts yet)")
        return
    arr = {h: np.array([c[h] for c in curves]) for h in HS}
    cells = [f"{arr[h].mean():+.2f}[{arr[h].min():+.2f},{arr[h].max():+.2f}]" for h in HS]
    print(f"  {name:13s}:" + "  ".join(f"{c:>15s}" for c in cells) + f"   (seeds {used})")


def main():
    otr, itr, _ = M.load_obs("train")
    otb, itb, _ = M.load_obs("test_b")
    print(f"\n=== d={D} forecast band, {OBS}, R^2 mean[min,max] ===")
    print("    h:           " + "  ".join(f"{h:>15d}" for h in HS))
    band("JEPA-own", jepa_tags, otr, itr, otb, itb)
    band("regAE-matched", regae_tags, otr, itr, otb, itb)


if __name__ == "__main__":
    main()

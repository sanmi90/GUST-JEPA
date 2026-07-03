"""Track O1 recovery for the D250 native-pooled-pipeline flagship (jepa_pool_vec).

Same frozen protocol as scripts/session32/track_o1_recovery.py (W=30 causal
windows, per-model TCSI OSP taps, GroupKFold-CV mapping selection, case
bootstrap): the ONLY change is the family under test, the vec retrain whose
training predictor is the vector AutoregressivePredictor instead of the tiled
ResUNet (HANDOFF D250). A fresh OSP staircase is built for jepa_pool_vec with
the identical criterion into a session33 taps file; the frozen session32 taps
are merged in unmodified so the jepa/fukami/POD baselines stay bit-identical.

Requires the Q1 latent cache for jepa_pool_vec at --cache-dir
(latents_jepa_pool_vec_{train,test_b}.npz + pressure_{train,test_b}.npz).

Run (RTX 6000):
    taskset -c 0-15 python -m scripts.session33.vec_o1_recovery --gpu 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import scripts.session32.track_o1_recovery as o1  # noqa: E402

VEC_FAMILY = {
    # pooled == spatial name (the vec model has no spatial tier), mirroring the
    # fukami/POD single-tier entries in the frozen FAMILIES table.
    "jepa_vec": {"pooled": "jepa_pool_vec", "spatial": "jepa_pool_vec", "role": "predictive"},
}


def build_vec_taps(cache_dir: Path, out_path: Path, seed: int) -> Path:
    """Merge the frozen OSP taps with a fresh jepa_pool_vec staircase."""
    from scripts.session32.osp_select import build_osp_taps
    from src.evaluation.rom_eval import load_windows

    osp = json.loads((REPO_ROOT / "outputs/session32/osp_taps_v2p2.json").read_text())
    if "jepa_pool_vec" not in osp:
        print("[vec-o1] building OSP staircase for jepa_pool_vec", flush=True)
        windows = load_windows(REPO_ROOT / "outputs/session31/windows_v2p2.json")
        qdeim = json.loads(
            (REPO_ROOT / "outputs/session32/qdeim_taps_v2p2.json").read_text()
        )
        caches = {"jepa_pool_vec": o1.load_cache(cache_dir, "jepa_pool_vec", "train")}
        p_train = o1.load_pressure(cache_dir, "train")["p_wall"]
        payload = build_osp_taps(caches, windows, p_train, w=30, qdeim_taps=qdeim, seed=seed)
        osp["jepa_pool_vec"] = payload["jepa_pool_vec"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(osp, indent=2))
    print(f"[vec-o1] taps -> {out_path}", flush=True)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Track O1 recovery for jepa_pool_vec (D250)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="outputs/session33/q1_vec_latents")
    ap.add_argument("--out", default="outputs/session33/track_o1_recovery_vec.json")
    args = ap.parse_args(argv)

    cache_dir = REPO_ROOT / args.cache_dir
    taps_path = build_vec_taps(
        cache_dir, REPO_ROOT / "outputs/session33/osp_taps_vec.json", args.seed
    )

    o1.FAMILIES.update(VEC_FAMILY)
    return o1.main(
        [
            "--gpu", str(args.gpu),
            "--seed", str(args.seed),
            "--cache-dir", str(cache_dir),
            "--osp-taps", str(taps_path),
            "--families", "jepa_vec",
            "--out", args.out,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

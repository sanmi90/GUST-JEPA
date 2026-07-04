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


def build_vec_taps(cache_dir: Path, out_path: Path, seed: int, model: str) -> Path:
    """Merge the frozen OSP taps with a fresh staircase for ``model``."""
    from scripts.session32.osp_select import build_osp_taps
    from src.evaluation.rom_eval import load_windows

    osp = json.loads((REPO_ROOT / "outputs/session32/osp_taps_v2p2.json").read_text())
    if model not in osp:
        print(f"[vec-o1] building OSP staircase for {model}", flush=True)
        windows = load_windows(REPO_ROOT / "outputs/session31/windows_v2p2.json")
        qdeim = json.loads(
            (REPO_ROOT / "outputs/session32/qdeim_taps_v2p2.json").read_text()
        )
        caches = {model: o1.load_cache(cache_dir, model, "train")}
        p_train = o1.load_pressure(cache_dir, "train")["p_wall"]
        payload = build_osp_taps(caches, windows, p_train, w=30, qdeim_taps=qdeim, seed=seed)
        osp[model] = payload[model]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(osp, indent=2))
    print(f"[vec-o1] taps -> {out_path}", flush=True)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Track O1 recovery for a vec model (D250)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="jepa_pool_vec")
    ap.add_argument("--mapping", choices=["krr", "mlp", "lstm", "lstm_tuned"], default=None,
                    help="Force a single recovery estimator (e.g. lstm).")
    ap.add_argument("--cache-dir", default="outputs/session33/q1_vec_latents")
    ap.add_argument("--out", default="outputs/session33/track_o1_recovery_vec.json")
    args = ap.parse_args(argv)

    cache_dir = REPO_ROOT / args.cache_dir
    taps_stem = args.model.replace("jepa_pool_vec", "vec")
    taps_path = build_vec_taps(
        cache_dir,
        REPO_ROOT / f"outputs/session33/osp_taps_{taps_stem}.json",
        args.seed,
        args.model,
    )

    fam = "jepa_vec" if args.model == "jepa_pool_vec" else args.model
    o1.FAMILIES.update(
        {fam: {"pooled": args.model, "spatial": args.model, "role": "predictive"}}
    )
    o1_argv = [
        "--gpu", str(args.gpu),
        "--seed", str(args.seed),
        "--cache-dir", str(cache_dir),
        "--osp-taps", str(taps_path),
        "--families", fam,
        "--out", args.out,
    ]
    if args.mapping is not None:
        o1_argv += ["--mapping", args.mapping]
    return o1.main(o1_argv)


if __name__ == "__main__":
    raise SystemExit(main())

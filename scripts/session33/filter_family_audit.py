"""Filter-family audit: run the FROZEN filter envelope on the reconstructive family.

Session 33 audit (user challenge, 2026-07-03): the paper's central question is
"which reduced coefficient state can a wall-pressure filter track", but the
frozen envelope ran only jepa_pool and pod. This runs the SAME frozen protocol
(D220: rho=1.0, K=8, osp_per_model, N=64, stochastic, field-free init) on the
reconstructive family: fukami, fukami_wake, and the matched kit control
ae_wake_pool. Missing OSP staircases (fukami_wake, ae_wake_pool) are built with
the identical TCSI criterion into a session33 taps file; the frozen session32
taps are copied in unmodified.

Characterisation only; nothing is tuned. Output:
outputs/session33/envelope_family_audit.json (same schema as envelope_by_gust).

Run (RTX 6000):
    taskset -c 0-15 python -m scripts.session33.filter_family_audit --gpu 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import scripts.session32.envelope_by_gust as env  # noqa: E402
from scripts.session32.track_o1_recovery import load_cache, load_pressure  # noqa: E402

NEW_MODELS = {
    "fukami": (
        "outputs/runs/session31/fukami",
        True,
        "outputs/session31/q1_latents/latents_%s_train.npz",
    ),
    "fukami_wake": (
        "outputs/runs/session31/fukami_wake",
        True,
        "outputs/session31/q1_latents/latents_%s_train.npz",
    ),
    "ae_wake_pool": (
        "outputs/runs/session32/ae_wake_pool",
        False,
        "outputs/session32/q1_pool_latents/latents_%s_train.npz",
    ),
}
CACHE_DIR = {
    "fukami": "outputs/session31/q1_latents",
    "fukami_wake": "outputs/session31/q1_latents",
    "ae_wake_pool": "outputs/session32/q1_pool_latents",
}


def build_missing_taps(models, osp_src, qdeim_src, out_path, seed):
    """Merge the frozen taps with freshly built staircases for missing models."""
    from src.evaluation.rom_eval import load_windows
    from scripts.session32.osp_select import build_osp_taps

    osp = json.loads((REPO_ROOT / osp_src).read_text())
    missing = [m for m in models if m not in osp]
    if missing:
        print(f"[fam-audit] building OSP staircases for {missing}", flush=True)
        windows = load_windows(REPO_ROOT / "outputs/session31/windows_v2p2.json")
        qdeim = json.loads((REPO_ROOT / qdeim_src).read_text())
        caches = {
            m: load_cache(REPO_ROOT / CACHE_DIR[m], m, "train") for m in missing
        }
        p_train = load_pressure(REPO_ROOT / CACHE_DIR[missing[0]], "train")["p_wall"]
        payload = build_osp_taps(caches, windows, p_train, w=30, qdeim_taps=qdeim,
                                 seed=seed)
        for m in missing:
            osp[m] = payload[m]
    out = REPO_ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(osp, indent=2))
    print(f"[fam-audit] taps -> {out}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="frozen-filter envelope on the recon family")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=list(NEW_MODELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/session33/envelope_family_audit.json")
    args = ap.parse_args(argv)

    taps_path = build_missing_taps(
        args.models,
        "outputs/session32/osp_taps_v2p2.json",
        "outputs/session32/qdeim_taps_v2p2.json",
        "outputs/session33/osp_taps_family_audit.json",
        args.seed,
    )

    # extend the envelope harness with the reconstructive family and run it
    env.MODEL_RUN.update(NEW_MODELS)
    argv2 = [
        "--gpu", str(args.gpu),
        "--models", *args.models,
        "--osp-taps", str(taps_path),
        "--out", args.out,
        "--seed", str(args.seed),
    ]
    if args.limit:
        argv2 += ["--limit", str(args.limit)]
    return env.main(argv2)


if __name__ == "__main__":
    raise SystemExit(main())

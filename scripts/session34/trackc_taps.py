"""Track C OSP tap staircases for every cell (C5 step 1).

Builds a fresh OSP staircase (identical criterion to the frozen session32
protocol: W=30 causal windows, TCSI selection, seed 0) for each Track C cell's
s0 run, merged with the frozen ``outputs/session32/osp_taps_v2p2.json`` so any
overlapping baseline stays bit-identical. Also symlinks the model-independent
pressure caches (frozen session31 artifacts) into the Track C latent cache dir,
which ``build_osp_taps`` requires.

Output: ``outputs/session34/osp_taps_trackc.json``.

Run (CPU-heavy, after the C1 latent sweep):
    taskset -c 0-15 python -m scripts.session34.trackc_taps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import scripts.session32.track_o1_recovery as o1  # noqa: E402
from scripts.session32.osp_select import build_osp_taps  # noqa: E402
from scripts.session34.trackc_cells import CELLS  # noqa: E402
from src.evaluation.rom_eval import load_windows  # noqa: E402


def ensure_pressure_symlinks(cache_dir: Path) -> None:
    for s in ("train", "test_b"):
        dst = cache_dir / f"pressure_{s}.npz"
        if not dst.exists():
            src = REPO_ROOT / f"outputs/session31/q1_latents/pressure_{s}.npz"
            if not src.exists():
                raise FileNotFoundError(f"frozen pressure cache missing: {src}")
            dst.symlink_to(src)
            print(f"[trackc-taps] symlinked {dst.name}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track C OSP taps")
    ap.add_argument("--cache-dir", default="outputs/session34/trackc_latents")
    ap.add_argument("--out", default="outputs/session34/osp_taps_trackc.json")
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cache_dir = REPO_ROOT / args.cache_dir
    ensure_pressure_symlinks(cache_dir)

    osp = json.loads((REPO_ROOT / "outputs/session32/osp_taps_v2p2.json").read_text())
    qdeim = json.loads((REPO_ROOT / "outputs/session32/qdeim_taps_v2p2.json").read_text())
    windows = load_windows(REPO_ROOT / "outputs/session31/windows_v2p2.json")
    p_train = o1.load_pressure(cache_dir, "train")["p_wall"]

    t0 = time.time()
    for cell in args.cells:
        run_name = CELLS[cell][0]
        if run_name in osp:
            print(f"[trackc-taps] {cell} ({run_name}): frozen staircase reused", flush=True)
            continue
        print(f"[trackc-taps] {cell} ({run_name}): building OSP staircase", flush=True)
        caches = {run_name: o1.load_cache(cache_dir, run_name, "train")}
        payload = build_osp_taps(caches, windows, p_train, w=30, qdeim_taps=qdeim,
                                 seed=args.seed)
        osp[run_name] = payload[run_name]

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(osp, indent=2))
    print(f"[trackc-taps] wrote {out_path} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

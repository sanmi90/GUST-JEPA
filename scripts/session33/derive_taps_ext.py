"""Derive the extended taps file with K1 prefixes (Track T2b; HANDOFF D239).

The frozen Session 32 taps files are NEVER edited. This script copies
``osp_taps_v2p2.json`` (per-model nested TCSI staircases + qdeim_shared) into
``outputs/session33/taps_v2p2_ext.json`` and adds a ``K1`` entry per model as the
prefix of that model's ``K2`` staircase (the staircase is warm-started, so K1 is
the greedy first pick by construction). ``src.estimation.obs_operator.load_osp_taps``
requires an exact ``K{k}`` key, which is why K=1 needs this derived file.

Run:
    python -m scripts.session33.derive_taps_ext
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def main(argv=None):
    ap = argparse.ArgumentParser(description="derive K1-extended taps file")
    ap.add_argument("--osp-taps", default="outputs/session32/osp_taps_v2p2.json")
    ap.add_argument("--qdeim-taps", default="outputs/session32/qdeim_taps_v2p2.json")
    ap.add_argument("--out", default="outputs/session33/taps_v2p2_ext.json")
    args = ap.parse_args(argv)

    osp_path = _resolve(args.osp_taps)
    qdeim_path = _resolve(args.qdeim_taps)
    osp = json.loads(osp_path.read_text())
    qdeim = json.loads(qdeim_path.read_text())

    out = {}
    for model, stair in osp.items():
        if not isinstance(stair, dict) or "K2" not in stair:
            out[model] = stair  # provenance blocks etc.
            continue
        ext = dict(stair)
        ext["K1"] = [int(stair["K2"][0])]
        out[model] = ext
    # qdeim_shared may live inside osp already; also mirror the standalone file.
    if "qdeim_shared" not in out and "K8" in qdeim:
        out["qdeim_shared"] = {
            f"K{k}": [int(t) for t in qdeim["K16"][:k]] for k in (1, 2, 4, 8, 16)
        }
    elif "qdeim_shared" in out and "K1" not in out["qdeim_shared"]:
        out["qdeim_shared"]["K1"] = [int(out["qdeim_shared"]["K2"][0])]

    out["provenance_ext"] = {
        "derivation": "K1 = prefix of the nested K2 staircase (greedy first pick)",
        "source_osp": str(args.osp_taps),
        "source_osp_sha256": hashlib.sha256(osp_path.read_bytes()).hexdigest(),
        "source_qdeim": str(args.qdeim_taps),
        "source_qdeim_sha256": hashlib.sha256(qdeim_path.read_bytes()).hexdigest(),
        "note": "frozen session32 taps files are unmodified; this file only ADDS K1",
    }

    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    for m in ("jepa_pool", "qdeim_shared"):
        if m in out and isinstance(out[m], dict):
            print(f"[taps-ext] {m}: K1={out[m].get('K1')} K2={out[m].get('K2')}")
    print(f"[taps-ext] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

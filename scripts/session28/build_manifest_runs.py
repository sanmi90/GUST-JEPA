"""Build manifest_runs.yaml: tag -> output dir -> W&B id for every session-28 run.

Gate GA1 artifact (master plan A1). Scans outputs/runs/session28/*/ (and the
encoder/ subdir convention for JEPA cells), reads the config line (first line)
of metrics.jsonl, and emits scripts/session28/manifest_runs.yaml with one entry
per run: tag, output_dir, wandb_run_id, gpu_name, seed, d, split_sha256,
code_sha256, final_checkpoint, status (complete / running / missing).

Usage:
    python scripts/session28/build_manifest_runs.py \
        [--root outputs/runs/session28] [--out scripts/session28/manifest_runs.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]

CONFIG_KEYS = (
    "wandb_run_id", "gpu_name", "seed", "d", "max_iters", "split_sha256",
    "inventory_sha256", "code_sha256", "partition_version", "lambda_sigreg",
    "predictor_cond_dim", "predictor_type", "lambda_wake", "observable_head",
    "auto_fallback_triggered", "baseline", "lambda_lift", "recon_loss_type",
    "lr", "encoder",
)


def read_config(metrics_path: Path) -> dict:
    try:
        with open(metrics_path) as f:
            first = json.loads(f.readline())
    except (OSError, json.JSONDecodeError):
        return {}
    if first.get("event") != "config":
        return {}
    return {k: first[k] for k in CONFIG_KEYS if k in first}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs/runs/session28")
    p.add_argument("--out", default="scripts/session28/manifest_runs.yaml")
    args = p.parse_args()

    root = REPO / args.root
    entries = {}
    for run_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        tag = run_dir.name
        # JEPA cells write into <tag>/encoder/, fukami/bvae cells into <tag>/.
        candidates = [run_dir / "encoder", run_dir]
        sub = next((c for c in candidates if (c / "metrics.jsonl").exists()), None)
        if sub is None:
            entries[tag] = {"output_dir": str(run_dir.relative_to(REPO)), "status": "missing"}
            continue
        cfg = read_config(sub / "metrics.jsonl")
        ckpts = sorted(sub.glob("checkpoint_iter*.pt"))
        final = ckpts[-1].name if ckpts else None
        expected = int(cfg.get("max_iters", 20000))
        complete = final is not None and f"{expected:06d}" in final
        entries[tag] = {
            "output_dir": str(sub.relative_to(REPO)),
            "final_checkpoint": final,
            "status": "complete" if complete else "running",
            **cfg,
        }

    payload = {
        "_doc": (
            "Session 28 training matrix manifest (gate GA1). Regenerate with "
            "scripts/session28/build_manifest_runs.py; do not edit by hand."
        ),
        "n_runs": len(entries),
        "runs": entries,
    }
    out = REPO / args.out
    with open(out, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    n_done = sum(1 for e in entries.values() if e.get("status") == "complete")
    print(f"[manifest] {len(entries)} runs ({n_done} complete) -> {out}")
    for tag, e in entries.items():
        gpu = e.get("gpu_name", "?")
        print(f"  {e.get('status', '?'):8s} {tag:32s} wandb={e.get('wandb_run_id', '-'):10s} gpu={gpu}")


if __name__ == "__main__":
    main()

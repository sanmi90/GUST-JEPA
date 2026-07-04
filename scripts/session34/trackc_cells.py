"""Canonical Track C cell -> run-directory mapping (Session 34).

Single source of truth for every Track C eval script. Cells are the 2x2x2
conditioning cube over {L, W, N} on the pooled d=32 vector-predictor JEPA,
plus the AE objective anchors. Reused runs (trained in sessions 32/33 with
byte-identical configs) are symlinked into ``outputs/runs/session34`` by
:func:`lay_symlinks` so every eval path goes through one runs base.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_BASE = REPO_ROOT / "outputs" / "runs" / "session34"

# cell -> {seed: run_name}; run_name is a directory under outputs/runs/session34
# (real for new runs, symlink for reused ones).
CELLS: dict[str, dict[int, str]] = {
    "c0": {0: "jepa_pool_c0_s0", 1: "jepa_pool_c0_s1", 2: "jepa_pool_c0_s2"},
    "cl": {0: "jepa_nowake_pool_vec", 1: "jepa_nowake_pool_vec_s1", 2: "jepa_nowake_pool_vec_s2"},
    "cw": {0: "jepa_pool_w_s0", 1: "jepa_pool_w_s1", 2: "jepa_pool_w_s2"},
    "cn": {0: "jepa_pool_n_s0", 1: "jepa_pool_n_s1", 2: "jepa_pool_n_s2"},
    "clw": {0: "jepa_pool_vec", 1: "jepa_pool_vec_s1", 2: "jepa_pool_vec_s2"},
    "cln": {0: "jepa_pool_ln_s0", 1: "jepa_pool_ln_s1", 2: "jepa_pool_ln_s2"},
    "cwn": {0: "jepa_pool_wn_s0", 1: "jepa_pool_wn_s1", 2: "jepa_pool_wn_s2"},
    "clwn": {0: "jepa_pool_lwn_s0", 1: "jepa_pool_lwn_s1", 2: "jepa_pool_lwn_s2"},
    "ae_l": {0: "ae_nowake_pool", 1: "ae_nowake_pool_s1", 2: "ae_nowake_pool_s2"},
    "ae_w": {0: "ae_w_pool_s0", 1: "ae_w_pool_s1", 2: "ae_w_pool_s2"},
    "ae_lw": {0: "ae_wake_pool", 1: "ae_wake_pool_s1", 2: "ae_wake_pool_s2"},
}

# Reused runs: run_name -> source directory (relative to repo root).
REUSED: dict[str, str] = {
    "jepa_pool_vec": "outputs/runs/session33/jepa_pool_vec",
    "jepa_pool_vec_s1": "outputs/runs/session33/jepa_pool_vec_s1",
    "jepa_pool_vec_s2": "outputs/runs/session33/jepa_pool_vec_s2",
    "jepa_nowake_pool_vec": "outputs/runs/session33/jepa_nowake_pool_vec",
    "ae_nowake_pool": "outputs/runs/session32/ae_nowake_pool",
    "ae_wake_pool": "outputs/runs/session32/ae_wake_pool",
    "ae_wake_pool_s1": "outputs/runs/session33/ae_wake_pool_s1",
    "ae_wake_pool_s2": "outputs/runs/session33/ae_wake_pool_s2",
}

# JEPA cube cells (pred objective); the AE anchors are the rest.
JEPA_CELLS = ("c0", "cl", "cw", "cn", "clw", "cln", "cwn", "clwn")
AE_CELLS = ("ae_l", "ae_w", "ae_lw")

CHECKPOINT = "checkpoint_iter010000.pt"


def all_run_names(cells: dict[str, dict[int, str]] | None = None) -> list[str]:
    """Every distinct run name across the cell map, sorted."""
    cells = CELLS if cells is None else cells
    names = {rn for seeds in cells.values() for rn in seeds.values()}
    return sorted(names)


def lay_symlinks(runs_base: Path = RUNS_BASE, verbose: bool = True) -> None:
    """Symlink reused runs into ``runs_base`` (idempotent)."""
    runs_base.mkdir(parents=True, exist_ok=True)
    for run_name, src_rel in REUSED.items():
        dst = runs_base / run_name
        src = REPO_ROOT / src_rel
        if dst.exists() or dst.is_symlink():
            continue
        if not src.exists():
            raise FileNotFoundError(f"reused run missing: {src}")
        dst.symlink_to(src)
        if verbose:
            print(f"[trackc-cells] symlinked {dst.name} -> {src_rel}", flush=True)


def missing_checkpoints(runs_base: Path = RUNS_BASE) -> list[str]:
    """Run names (across all cells) whose done-marker checkpoint is absent."""
    return [
        rn for rn in all_run_names()
        if not (runs_base / rn / CHECKPOINT).exists()
    ]


if __name__ == "__main__":
    lay_symlinks()
    missing = missing_checkpoints()
    print(f"[trackc-cells] {len(all_run_names())} runs, {len(missing)} missing checkpoints")
    for rn in missing:
        print(f"  missing: {rn}")

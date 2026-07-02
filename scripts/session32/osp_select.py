"""Track O -- per-model optimal sensor placement (OSP) via TCSI greedy selection.

DIRECTIVE 1 (Carlos): sensor placement is MODEL-CONDITIONED, not one shared
target-blind array for every family. For each model we select its own K taps by
TCSI greedy forward selection (the verified primitive
``scripts/session14_tcsi_pilot.greedy_forward_selection``) conditioned on THAT
model's latent -- target = the first principal component of the model's pooled
latent on TRAIN (PCA n_components=1), the established TCSI convention (see
``scripts/session21/exp_pressure_v2_tcsi.py``). The Session-21 study's flaw was
applying ONE array (derived on jepa d=64) to every family; here each model gets
its own array. The target-blind qDEIM set is retained as a comparison row.

Emits ``outputs/session32/osp_taps_v2p2.json`` keyed by model with staircase K in
{2,4,8,16} (greedy is prefix-consistent; extended via the ``initial`` warm-start),
plus a ``qdeim_shared`` entry. Track B reads the per-model K8 taps from this file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from session14_tcsi_pilot import greedy_forward_selection  # noqa: E402

KS = (2, 4, 8, 16)
N_TAPS = 192


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def encounter_impact_windows(
    z_gap: np.ndarray,
    case_id: np.ndarray,
    enc: np.ndarray,
    frame: np.ndarray,
    p_rows: np.ndarray,
    windows: dict,
    w: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """One impact-anchored pressure window + impact-frame latent per TRAIN encounter.

    Returns ``(X_window (n_enc, 192, W), Z_impact (n_enc, d), keys)`` where the
    window is the causal ``[t_impact-W+1, t_impact]`` block (left-clamped) at all
    192 taps and ``Z_impact`` is the model's pooled latent at ``t_impact``.
    """
    keys = np.array([f"{c}/{int(e):02d}" for c, e in zip(case_id, enc)])
    Xw, Zi, klist = [], [], []
    for key in sorted(set(keys.tolist())):
        w_entry = windows.get(key)
        if w_entry is None:
            continue
        ti = int(w_entry["t_impact"])
        rows = np.where(keys == key)[0]
        fr = frame[rows]
        order = np.argsort(fr)
        rows_s, fr_s = rows[order], fr[order]
        pos = {int(f): i for i, f in enumerate(fr_s)}
        # impact-frame latent (nearest available frame to t_impact)
        if ti in pos:
            zi = z_gap[rows_s[pos[ti]]]
        else:
            j = int(np.argmin(np.abs(fr_s - ti)))
            zi = z_gap[rows_s[j]]
        # causal window ending at t_impact, left-clamped to earliest frame
        block = p_rows[rows_s]  # (T, 192)
        win = np.empty((N_TAPS, w), dtype=np.float32)
        for t in range(w):
            ff = ti - (w - 1) + t
            ff = max(int(fr_s[0]), min(ff, int(fr_s[-1])))
            win[:, t] = block[pos[int(ff)]]
        Xw.append(win)
        Zi.append(zi)
        klist.append(key)
    return np.stack(Xw, axis=0), np.stack(Zi, axis=0), klist


def tcsi_staircase(X_window: np.ndarray, target: np.ndarray, ks=KS) -> dict:
    """Greedy TCSI staircase; each K extends the previous via warm-start."""
    sel: list[int] = []
    out: dict[str, list[int]] = {}
    for k in ks:
        sel = greedy_forward_selection(X_window, target, int(k), initial=sel)
        out[f"K{int(k)}"] = [int(x) for x in sel[: int(k)]]
    return out


def build_osp_taps(
    model_caches: dict[str, dict],
    windows: dict,
    pres_train: np.ndarray,
    *,
    w: int,
    qdeim_taps: dict,
    seed: int,
) -> dict:
    """Per-model TCSI taps + qdeim_shared baseline. ``model_caches[m]`` is the
    TRAIN cache dict for model ``m`` (needs z_gap, case_id, encounter_index,
    frame)."""
    from sklearn.decomposition import PCA

    payload: dict = {
        "schema": {
            "<model>": {
                "method": "tcsi_greedy",
                "target": "latent_pc1",
                "K2/K4/K8/K16": "physical wall-tap indices 0..191, greedy addition "
                "order, nested (warm-start staircase)",
            },
            "qdeim_shared": "target-blind qDEIM baseline (same taps for every model)",
            "n_taps": N_TAPS,
        },
        "params": {
            "W_window": w,
            "target": "PCA(n_components=1) of pooled latent at t_impact",
            "primitive": "scripts/session14_tcsi_pilot.greedy_forward_selection",
            "seed": seed,
        },
    }
    for m, cache in model_caches.items():
        Xw, Zi, _ = encounter_impact_windows(
            cache["z_gap"],
            cache["case_id"],
            cache["encounter_index"],
            cache["frame"],
            pres_train,
            windows,
            w,
        )
        target = PCA(n_components=1, random_state=seed).fit_transform(Zi).ravel()
        taps = tcsi_staircase(Xw, target, KS)
        payload[m] = {
            "method": "tcsi_greedy",
            "target": "latent_pc1",
            "n_encounters": int(Xw.shape[0]),
            **taps,
        }
        print(f"[osp] {m}: K8={taps['K8']}", flush=True)
    payload["qdeim_shared"] = {
        "method": "qdeim",
        "target": "none (target-blind)",
        **{f"K{k}": [int(x) for x in qdeim_taps[f"K{k}"]] for k in KS},
    }
    return payload


def save_osp_taps(payload: dict, out_path: str | Path) -> Path:
    out = _resolve(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out

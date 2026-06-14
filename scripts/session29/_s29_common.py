"""Shared helpers for SESSION29 (JFM remediation, reconciled to v2.1).

Loaders for the frozen v2.1 per-family latents and the family-independent DNS
per-frame observables, plus a provenance-header writer that every SESSION29
script stamps onto its JSON output (git SHA, command line, UTC timestamp, input
file hashes, package versions, seed). Reuses session28 `stats_lib` for the
case-clustered bootstrap so the uncertainty convention is identical to the paper.

All paths are v2.1 (split_v2p1, outputs/session28/latents/<tag>/). NEVER --split v2.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LATENTS = REPO / "outputs" / "session28" / "latents"
TARGETS = REPO / "outputs" / "session28" / "exp2" / "per_frame_targets"
sys.path.insert(0, str(REPO / "scripts" / "session28"))
import stats_lib  # noqa: F401,E402  (re-exported as cm.stats_lib for downstream tracks)

# Canonical v2.1 frozen families for the probe/floor comparisons (seed 42 = lead).
FAMILY_TAGS = {
    "jepa_tf_noc": "jepa_tf_noc_d64_s42",
    "fukami": "fukami_d64_s42",
    "pod": "pod_d64",
    # SESSION29 Track E (F4): wake+lift heads + SIGReg, NO predictive objective
    # (predictive_weight=0). Isolates whether wake supervision alone suffices.
    "supervised_only": "supervised_only_d64_s42",
    # Matched-wake-head reconstructive control (fuk_matched recipe: reconstruction
    # + the SAME wake head as jepa/supervised_only). This is the correct "same
    # supervision, reconstructive objective" cell; `fukami` is the lineage AE with
    # NO wake head and is a separate (headline) baseline.
    "ctrl_recon": "ctrl_recon_cnnvit_s1",
}
ENCODER_SEED_TAGS = {  # for seed-variance robustness (encoder seeds, not probe)
    "jepa_tf_noc": ["jepa_tf_noc_d64_s0", "jepa_tf_noc_d64_s1", "jepa_tf_noc_d64_s42"],
    "fukami": ["fukami_d64_s1", "fukami_d64_s2"],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "UNKNOWN"


def provenance(inputs: list[Path], seed: int | None = None) -> dict:
    """Provenance header for a SESSION29 artifact."""
    import sklearn

    return {
        "git_sha": git_sha(),
        "command": " ".join(sys.argv),
        "utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {str(p): _sha256(p) for p in inputs if Path(p).exists()},
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "seed": seed,
        "split": "v2p1",
        "reconciled_from": "SESSION29 plan (was main_13/v2); see RECONCILIATION_v2p1.md",
    }


def load_family(tag: str, split: str) -> dict:
    """Load one family's frozen latents for one split (z_full + alignment keys)."""
    p = LATENTS / tag / f"{split}.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing latents: {p}")
    d = np.load(p, allow_pickle=True)
    return {
        "z_full": d["z_full"],  # (n, 120, d)
        "case_ids": np.asarray(d["case_ids"]).astype(str),
        "encounter_indices": np.asarray(d["encounter_indices"]).astype(int),
        "impact_frame": np.asarray(d["impact_frame"]).astype(int),
        "path": p,
    }


def load_dns_observable(split: str, observable: str) -> dict:
    """Family-independent DNS per-frame observable, keyed by (case_id, encounter)."""
    p = TARGETS / f"{split}.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing DNS targets: {p}")
    d = np.load(p, allow_pickle=True)
    if observable not in d:
        raise KeyError(f"observable {observable!r} not in {p}; have {list(d.keys())}")
    cid = np.asarray(d["case_id"]).astype(str)
    enc = np.asarray(d["encounter_index"]).astype(int)
    obs = np.asarray(d[observable])  # (n, 120)
    return {(cid[i], int(enc[i])): obs[i] for i in range(len(cid))}, p


# Canonical DNS metrics: the SAME wake_enstrophy the published closure headline
# uses (reproduces the 0.79 number). dns_physical_metrics keys are <split>_<name>.
DNS_CANON = REPO / "outputs" / "session28" / "exp2" / "dns_physical_metrics.npz"


def load_dns_canonical(split: str, observable: str) -> dict:
    """Canonical (headline) DNS per-frame observable from dns_physical_metrics.npz,
    keyed by (case_id, encounter). Use this for headline-comparable numbers; the
    per_frame_targets variant (load_dns_observable) is a different/degraded scale."""
    if not DNS_CANON.exists():
        raise FileNotFoundError(f"missing canonical DNS metrics: {DNS_CANON}")
    d = np.load(DNS_CANON, allow_pickle=True)
    key = f"{split}_{observable}"
    if key not in d:
        raise KeyError(f"{key!r} not in {DNS_CANON}")
    cid = np.asarray(d[f"{split}_case_id"]).astype(str)
    enc = np.asarray(d[f"{split}_encounter_index"]).astype(int)
    obs = np.asarray(d[key])  # (n, 120)
    return {(cid[i], int(enc[i])): obs[i] for i in range(len(cid))}, DNS_CANON


def readout_xy(
    tag: str,
    split: str,
    observable: str,
    horizon: int,
    target_source: str = "per_frame",
):
    """Readout-frame design matrix: X = latent at impact+H, y = DNS obs at impact+H.

    Returns (X (n,d), y (n,), case_ids (n,)). One row per encounter, grouped by
    case_id. target_source 'per_frame' (default, back-compat) uses per_frame_targets;
    'canonical' uses dns_physical_metrics (the published-headline wake target). The
    fit/eval regime is the readout frame impact+H; the paper's pooled-frame ridge
    headline is a separate number.
    """
    fam = load_family(tag, split)
    if target_source == "canonical":
        obs_map, _ = load_dns_canonical(split, observable)
    else:
        obs_map, _ = load_dns_observable(split, observable)
    z, imp, cid, enc = (
        fam["z_full"],
        fam["impact_frame"],
        fam["case_ids"],
        fam["encounter_indices"],
    )
    X, y, groups = [], [], []
    for i in range(len(cid)):
        fr = int(imp[i]) + horizon
        key = (cid[i], int(enc[i]))
        if key not in obs_map or fr >= z.shape[1]:
            continue
        X.append(z[i, fr, :])
        y.append(float(obs_map[key][fr]))
        groups.append(cid[i])
    return np.asarray(X), np.asarray(y), np.asarray(groups)


def case_clustered_r2_ci(y_true, y_pred, case_ids, n_boot=2000, seed=0):
    """Case-clustered bootstrap CI on R^2 (resample cases). Reuses stats_lib RNG style."""
    rng = np.random.default_rng(seed)
    cases = np.unique(case_ids)
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    r2_point = (
        1.0 - float(((y_true - y_pred) ** 2).sum()) / sst if sst > 0 else float("nan")
    )
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(cases, size=len(cases), replace=True)
        idx = np.concatenate([np.where(case_ids == c)[0] for c in pick])
        yt, yp = y_true[idx], y_pred[idx]
        s = float(((yt - yt.mean()) ** 2).sum())
        if s > 0:
            boots.append(1.0 - float(((yt - yp) ** 2).sum()) / s)
    lo, hi = (
        (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        if boots
        else (float("nan"), float("nan"))
    )
    return r2_point, lo, hi


def write_artifact(out_json: Path, payload: dict) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=float))

"""SESSION29.8 A1: latent-drift diagnostic for the supervised_only encoder family.

Adds the supervised_only encoder to the manuscript latent-drift table
(tab:latent_drift) at the SAME horizon, reference cloud, covariance estimator,
and rollout protocol used for the existing families (jepa, fukami, pod, bvae) in
scripts/session28/drift_ce1.py. We do NOT reimplement any formula: this module
imports drift_ce1 and reuses its compute_key, which means the supervised_only
number is apples-to-apples by construction.

The three reported metrics (verbatim manuscript macro names):
    DriftRelLtwoSupOnly       relative-l2 drift of the rollout vs the DNS-encoded
                              latent, mean over test_b at H = 24, d = 64.
    DriftMahaSupOnly          Mahalanobis ratio of the rolled latent to the
                              per-frame TRAIN encoded distribution (v2 floored-
                              covariance regime), H = 24, d = 64.
    DriftSpecNearNullSupOnly  fraction of the rollout's squared-Mahalanobis
                              departure carried by the near-null encoded eigen-
                              directions at the headline horizon, d = 64.

Inputs (mirroring how the other families are wired in drift_ce1.py):
    encoded latents : outputs/session28/latents/supervised_only_d64_s42/
                      {train,test_b}.npz
    matched rollout : outputs/session28/rollouts/supervised_only_d64/test_b.npz
                      (built with the matched AR-transformer predictor
                      outputs/session28/predictors/supervised_only_d64/
                      checkpoint_iter020000.pt; z_dns is the encoded reference,
                      byte-identical to the encoded z_full as for every family).

Outputs:
    outputs/session28/numbers_parts/drift_supervised_only.json   eval_all part.
    outputs/session29/reports/drift_supervised_only.md           provenance +
                      a 1-row comparison against jepa and fukami.

CPU only; run under nice. Usage:
    python scripts/session29/drift_supervised_only.py
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import drift_ce1  # noqa: E402  (path injection above)

ROLL_DIR = REPO / "outputs" / "session28" / "rollouts"
LAT_DIR = REPO / "outputs" / "session28" / "latents"
PRED_CKPT = (
    REPO / "outputs" / "session28" / "predictors" / "supervised_only_d64"
    / "checkpoint_iter020000.pt"
)
PARTS_DIR = REPO / "outputs" / "session28" / "numbers_parts"
REPORT_DIR = REPO / "outputs" / "session29" / "reports"

SUPONLY_KEY = "supervised_only_d64"
SUPONLY_LAT = "supervised_only_d64_s42"
SOURCE = "drift_supervised_only (SESSION29.8 A1)"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO)
        ).decode().strip()
    except Exception:  # pragma: no cover - provenance best-effort
        return "unknown"


def compute_suponly() -> dict:
    """Run drift_ce1.compute_key on the supervised_only rollout key.

    We register the supervised_only key in drift_ce1.PRED_LAT so compute_key
    resolves its encoded TRAIN latents exactly as it does for every family, then
    invoke the unmodified compute_key. The horizon set, headline horizon, the
    per-frame TRAIN floored-covariance reference cloud, the Mahalanobis ratio
    against z_dns, and the near-null departure-spectrum cutoff are all taken from
    drift_ce1 (no local copies), so the protocol is identical.
    """
    if not (ROLL_DIR / SUPONLY_KEY / "test_b.npz").exists():
        raise SystemExit(f"[drift-suponly] missing rollout {ROLL_DIR / SUPONLY_KEY}")
    if not (LAT_DIR / SUPONLY_LAT / "train.npz").exists():
        raise SystemExit(f"[drift-suponly] missing encoded latents {LAT_DIR / SUPONLY_LAT}")
    # Register the suponly key with its encoded-latent directory (same mechanism
    # PRED_LAT uses for jepa/fukami/pod/bvae) so compute_key can resolve it.
    drift_ce1.PRED_LAT[SUPONLY_KEY] = SUPONLY_LAT
    return drift_ce1.compute_key(SUPONLY_KEY)


def build_part(r: dict) -> dict:
    """Assemble the eval_all numbers part for supervised_only.

    Values come straight from the drift_ce1 compute_key output at the headline
    horizon (H = 24), identical to how build_numbers_part reads the other
    families.
    """
    hidx = drift_ce1.HORIZONS.index(drift_ce1.HEADLINE_H)
    rel = drift_ce1.nanmean(r["rel_l2"][:, hidx])
    mah = drift_ce1.nanmean(r["maha_ratio"][:, hidx])
    nnull = r["spec_near_null_departure_frac"]
    n_null_dirs = r["spec_n_null_dirs"]
    n_enc = r["n_enc"]
    H = drift_ce1.HEADLINE_H
    numbers = {
        "drift_rel_l2_H24_d64_suponly": {
            "macro": "DriftRelLtwoSupOnly",
            "value": rel,
            "fmt": "%.2f",
            "n": n_enc,
            "split": "test_b",
            "horizon": H,
            "source": SOURCE,
            "note": (
                "mean rel-l2 deviation, supervised_only d=64, full-context "
                f"rollout (matched AR-transformer), H={H}; identical protocol to "
                "drift_ce1.py (tab:latent_drift)"
            ),
        },
        "drift_maha_ratio_H24_d64_suponly": {
            "macro": "DriftMahaSupOnly",
            "value": mah,
            "fmt": "%.2f",
            "n": n_enc,
            "split": "test_b",
            "horizon": H,
            "source": SOURCE,
            "note": (
                "mean Mahalanobis ratio to per-frame TRAIN encoded cloud (v2 "
                f"floored-covariance regime), supervised_only d=64, H={H}"
                + ("; ratio<1 (predictor regresses to train cloud)" if mah < 1 else "")
            ),
        },
        "drift_spec_nearnull_frac_d64_suponly": {
            "macro": "DriftSpecNearNullSupOnly",
            "value": nnull,
            "fmt": "%.2f",
            "split": "test_b",
            "horizon": H,
            "source": SOURCE,
            "note": (
                "fraction of squared-Mahalanobis departure in near-null encoded "
                f"eigen-dirs ({n_null_dirs} dirs), supervised_only d=64"
            ),
        },
    }
    return {"part": "drift_supervised_only", "numbers": numbers}


def load_family_headline(fam_key: str) -> dict:
    """Recompute the headline triple for an existing family for the report row."""
    r = drift_ce1.compute_key(fam_key)
    hidx = drift_ce1.HORIZONS.index(drift_ce1.HEADLINE_H)
    return {
        "rel_l2": drift_ce1.nanmean(r["rel_l2"][:, hidx]),
        "maha_ratio": drift_ce1.nanmean(r["maha_ratio"][:, hidx]),
        "near_null_frac": r["spec_near_null_departure_frac"],
        "n_null_dirs": r["spec_n_null_dirs"],
        "enc_topk_var": float(r["encoded_topk_var_frac"][hidx]),
    }


def write_report(part: dict, sup: dict, jepa: dict, fukami: dict) -> None:
    """Write the provenance + 3-family comparison markdown."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    nums = part["numbers"]
    s_rel = nums["drift_rel_l2_H24_d64_suponly"]["value"]
    s_mah = nums["drift_maha_ratio_H24_d64_suponly"]["value"]
    s_nn = nums["drift_spec_nearnull_frac_d64_suponly"]["value"]
    H = drift_ce1.HEADLINE_H

    # Resemblance verdict: supervised_only resembles whichever of the predictive
    # (jepa, on-manifold) or reconstructive (fukami, off-manifold along near-null)
    # family it is geometrically closer to on the (Maha ratio, near-null frac)
    # plane, both measured under the IDENTICAL drift_ce1 protocol.
    def _dist(a_mah, a_nn):
        return ((s_mah - a_mah) ** 2 + (s_nn - a_nn) ** 2) ** 0.5

    d_jepa = _dist(jepa["maha_ratio"], jepa["near_null_frac"])
    d_fukami = _dist(fukami["maha_ratio"], fukami["near_null_frac"])
    verdict = "jepa (predictive, on-manifold)" if d_jepa < d_fukami else (
        "fukami (reconstructive, off-manifold along near-null directions)"
    )

    lines = [
        "# Latent drift: supervised_only family (SESSION29.8 A1)\n",
        "Adds the supervised_only encoder to the manuscript latent-drift table "
        "(`tab:latent_drift`) using the identical horizon, reference cloud, "
        "covariance estimator, and rollout protocol as the existing families "
        "(jepa, fukami, pod, bvae) in `scripts/session28/drift_ce1.py`. The three "
        "metrics are computed by importing and calling `drift_ce1.compute_key`, "
        "not by reimplementing any formula, so they are apples-to-apples.\n",
        "## Provenance\n",
        f"- git sha: `{_git_sha()}`",
        "- command: `python scripts/session29/drift_supervised_only.py`",
        f"- UTC timestamp: {ts}",
        f"- horizon H = {H}, latent d = 64, split = test_b, full-context rollout "
        "(`z_full`)",
        "- inputs:",
        f"    - encoded latents: `{LAT_DIR / SUPONLY_LAT}/{{train,test_b}}.npz`",
        f"    - matched rollout: `{ROLL_DIR / SUPONLY_KEY}/test_b.npz`",
        f"    - predictor checkpoint: `{PRED_CKPT}`",
        f"- part file: `{PARTS_DIR / 'drift_supervised_only.json'}`\n",
        "## supervised_only headline metrics (H = 24, d = 64, test_b)\n",
        f"- `DriftRelLtwoSupOnly` = {s_rel:.4f} "
        "(relative-l2 rollout deviation vs DNS-encoded latent)",
        f"- `DriftMahaSupOnly` = {s_mah:.4f} "
        "(Mahalanobis ratio to per-frame TRAIN encoded cloud, v2 floored regime)",
        f"- `DriftSpecNearNullSupOnly` = {s_nn:.4f} "
        f"(near-null departure fraction; {sup['n_null_dirs']} near-null encoded "
        "eigen-directions)\n",
        "## 1-row comparison (same protocol, H = 24, d = 64, test_b)\n",
        "| family | rel-l2 | Maha ratio | near-null dep frac | n null dirs | enc top-5 var |",
        "|---|---|---|---|---|---|",
        f"| supervised_only | {s_rel:.3f} | {s_mah:.3f} | {s_nn:.3f} | "
        f"{sup['n_null_dirs']} | {sup['enc_topk_var']:.3f} |",
        f"| jepa (predictive, on-manifold) | {jepa['rel_l2']:.3f} | "
        f"{jepa['maha_ratio']:.3f} | {jepa['near_null_frac']:.3f} | "
        f"{jepa['n_null_dirs']} | {jepa['enc_topk_var']:.3f} |",
        f"| fukami (reconstructive, off-manifold) | {fukami['rel_l2']:.3f} | "
        f"{fukami['maha_ratio']:.3f} | {fukami['near_null_frac']:.3f} | "
        f"{fukami['n_null_dirs']} | {fukami['enc_topk_var']:.3f} |\n",
        "## Interpretation\n",
        "The reconstructive Fukami AE keeps a tiny rel-l2 but blows up the "
        "Mahalanobis ratio because its rollout exits along near-null encoded "
        "directions (high near-null departure fraction, many near-null dirs, "
        "variance concentrated in the top-5). The predictive JEPA wanders within "
        "the broad isotropised cloud (larger rel-l2, in-distribution Mahalanobis, "
        "few near-null dirs). On the (Mahalanobis ratio, near-null fraction) plane "
        f"supervised_only is closer to {verdict}.\n",
    ]
    (REPORT_DIR / "drift_supervised_only.md").write_text("\n".join(lines))
    return verdict


def main() -> None:
    print("[drift-suponly] computing supervised_only drift diagnostics (test_b) ...")
    r = compute_suponly()
    part = build_part(r)

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    (PARTS_DIR / "drift_supervised_only.json").write_text(
        json.dumps(part, indent=2, sort_keys=True)
    )
    print(f"[drift-suponly] wrote {PARTS_DIR / 'drift_supervised_only.json'}")

    print("[drift-suponly] recomputing jepa_d64 / fukami_d64 reference rows ...")
    jepa = load_family_headline("jepa_d64")
    fukami = load_family_headline("fukami_d64")
    verdict = write_report(part, _sup_row(r), jepa, fukami)
    print(f"[drift-suponly] wrote {REPORT_DIR / 'drift_supervised_only.md'}")

    nums = part["numbers"]
    print(
        "[drift-suponly] DriftRelLtwoSupOnly="
        f"{nums['drift_rel_l2_H24_d64_suponly']['value']:.4f}  "
        "DriftMahaSupOnly="
        f"{nums['drift_maha_ratio_H24_d64_suponly']['value']:.4f}  "
        "DriftSpecNearNullSupOnly="
        f"{nums['drift_spec_nearnull_frac_d64_suponly']['value']:.4f}"
    )
    print(f"[drift-suponly] resembles: {verdict}")
    print("[drift-suponly] done.")


def _sup_row(r: dict) -> dict:
    """Headline triple for supervised_only, for the report comparison row."""
    hidx = drift_ce1.HORIZONS.index(drift_ce1.HEADLINE_H)
    return {
        "rel_l2": drift_ce1.nanmean(r["rel_l2"][:, hidx]),
        "maha_ratio": drift_ce1.nanmean(r["maha_ratio"][:, hidx]),
        "near_null_frac": r["spec_near_null_departure_frac"],
        "n_null_dirs": r["spec_n_null_dirs"],
        "enc_topk_var": float(r["encoded_topk_var_frac"][hidx]),
    }


if __name__ == "__main__":
    main()

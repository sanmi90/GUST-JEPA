"""Session 17, Experiment 1, Part (d): cross-seed trajectory agreement.

For each of the 10 representative Test B encounters (chosen in Part b), and
each of the 4 seeds (production + Thrust-6 seed0/1/2):
  1. Take the per-encounter z_full (120 frames, 64-D).
  2. Compute the full pairwise distance matrix D[i, j] = ||z(t_i) - z(t_j)||.
  3. Normalize D by its median off-diagonal value (so the descriptor is
     scale-invariant). This is the basis-invariant trajectory descriptor.

For each encounter, compute Spearman correlation of upper-triangular flattened
distance matrices across the 6 seed pairs.

Acceptance gate (SESSION17_PLAN.md):
  cross-seed trajectory distance correlations exceed 0.7 on at least 7 of 10
  Test B encounters.

Outputs:
    outputs/session17/exp1/cross_seed_distance_corr.json
    outputs/session17/figures/exp1_cross_seed_distance.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[2]
SEED_LATENTS = REPO / "outputs" / "session17" / "seed_latents"
EXP1 = REPO / "outputs" / "session17" / "exp1"
FIGS = REPO / "outputs" / "session17" / "figures"
SEEDS = ("production", "seed0", "seed1", "seed2")
T_ENC = 120
T_IMPACT = 40


def load_test_b_full(seed: str) -> dict:
    f = SEED_LATENTS / seed / "test_b.npz"
    d = np.load(f, allow_pickle=True)
    return {
        "z_full": d["z_full"].astype(np.float32),
        "G": d["G"].astype(np.float32),
        "D": d["D"].astype(np.float32),
        "Y": d["Y"].astype(np.float32),
        "case_id": np.asarray(d["case_id"]).astype(object),
        "encounter_index": d["encounter_index"].astype(np.int32),
    }


def normalised_distance_matrix(z_full: np.ndarray) -> np.ndarray:
    """z_full: (T, d). Returns the upper-triangle (flattened) of D / median(D)."""
    T = z_full.shape[0]
    # pairwise squared distances
    diff2 = np.sum(
        (z_full[:, None, :] - z_full[None, :, :]) ** 2, axis=-1
    )
    D = np.sqrt(np.maximum(diff2, 0.0))
    iu = np.triu_indices(T, k=1)
    flat = D[iu]
    med = float(np.median(flat))
    if med < 1e-12:
        return flat * 0.0
    return flat / med


def main() -> None:
    rep = json.loads((EXP1 / "representative_encounters.json").read_text())
    # representative_encounters.json was built against session14 ordering
    # (v1 + v1p5 supplement). Re-resolve to the seed_latents v1p5 ordering by
    # case_id + encounter_index lookup so that we compare the SAME encounters
    # across seeds.
    session14_v1 = np.load(
        REPO / "outputs" / "session14" / "latents" / "S12_E_d64" / "test_b.npz",
        allow_pickle=True,
    )
    session14_supp = np.load(
        REPO / "outputs" / "session14" / "latents" / "S12_E_d64"
        / "test_b_v1p5_supplement.npz",
        allow_pickle=True,
    )
    merged_cid = np.concatenate(
        [session14_v1["case_id"], session14_supp["case_id"]]
    ).astype(str)
    merged_ei = np.concatenate(
        [session14_v1["encounter_index"], session14_supp["encounter_index"]]
    ).astype(int)

    seed_data = {seed: load_test_b_full(seed) for seed in SEEDS}
    base = seed_data["production"]
    prod_cid = base["case_id"].astype(str)
    prod_ei = base["encounter_index"].astype(int)

    rep_idx = []
    rep_labels = []
    rep_spec = []
    for r in rep:
        old_i = r["idx"]
        cid = merged_cid[old_i]
        ei = int(merged_ei[old_i])
        mask = (prod_cid == cid) & (prod_ei == ei)
        if not mask.any():
            print(f"[exp1d] WARN: ({cid}, {ei}) not in seed_latents")
            continue
        rep_idx.append(int(np.where(mask)[0][0]))
        rep_labels.append(r["label"])
        rep_spec.append({"case_id": cid, "encounter_index": ei, "label": r["label"]})
    print(f"[exp1d] representative encounter indices (seed_latents order): {rep_idx}")

    # Sanity-check case/encounter alignment across seeds.
    for seed, dat in seed_data.items():
        if seed == "production":
            continue
        for k in ("case_id", "encounter_index", "G", "D", "Y"):
            if not np.array_equal(dat[k], base[k]):
                print(f"[exp1d] WARN: {seed} {k} differs from production")

    # Compute normalised distance vectors per (seed, encounter).
    distance_vecs: dict[str, dict[int, np.ndarray]] = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        zf = seed_data[seed]["z_full"]
        for i in rep_idx:
            distance_vecs[seed][i] = normalised_distance_matrix(zf[i])

    # Pairwise Spearman correlation per encounter.
    pair_results = []
    seed_pairs = []
    for a in range(len(SEEDS)):
        for b in range(a + 1, len(SEEDS)):
            seed_pairs.append((SEEDS[a], SEEDS[b]))
    print(f"[exp1d] seed pairs ({len(seed_pairs)}): {seed_pairs}")

    per_enc_records = []
    for i, label in zip(rep_idx, rep_labels):
        per_pair = {}
        rhos = []
        for sa, sb in seed_pairs:
            rho, _p = spearmanr(distance_vecs[sa][i], distance_vecs[sb][i])
            per_pair[f"{sa}_vs_{sb}"] = float(rho)
            rhos.append(rho)
        per_enc_records.append(
            {
                "test_b_idx": int(i),
                "label": label,
                "G": float(base["G"][i]),
                "D": float(base["D"][i]),
                "Y": float(base["Y"][i]),
                "spearman_per_pair": per_pair,
                "mean_spearman": float(np.mean(rhos)),
                "min_spearman": float(np.min(rhos)),
                "n_pairs_above_0p7": int(np.sum(np.array(rhos) > 0.7)),
            }
        )

    # Acceptance gate.
    means = np.array([r["mean_spearman"] for r in per_enc_records])
    n_pass = int(np.sum(means > 0.7))
    gate_pass = n_pass >= 7

    summary = {
        "acceptance_gate_text": (
            "mean Spearman > 0.7 across 6 seed pairs on at least 7 of 10 "
            "representative Test B encounters"
        ),
        "gate_pass": gate_pass,
        "n_pass_mean_above_0p7": n_pass,
        "n_total": int(len(means)),
        "per_encounter": per_enc_records,
        "global_stats": {
            "mean_of_per_encounter_means": float(np.mean(means)),
            "min_per_encounter_mean": float(np.min(means)),
            "max_per_encounter_mean": float(np.max(means)),
            "median_per_encounter_mean": float(np.median(means)),
        },
    }
    (EXP1 / "cross_seed_distance_corr.json").write_text(json.dumps(summary, indent=2))
    print(f"[exp1d] wrote {EXP1 / 'cross_seed_distance_corr.json'}")

    # Console report.
    print("\n[exp1d] Per-encounter mean Spearman correlation across seed pairs:")
    print(f"  {'idx':>4}  {'G':>5} {'D':>4} {'Y':>5}  {'min':>5} {'mean':>5} {'max':>5}  >0.7?")
    rho_per_pair = np.array(
        [[r["spearman_per_pair"][f"{sa}_vs_{sb}"] for sa, sb in seed_pairs]
         for r in per_enc_records]
    )
    for r, row in zip(per_enc_records, rho_per_pair):
        flag = "PASS" if r["mean_spearman"] > 0.7 else "FAIL"
        print(
            f"  {r['test_b_idx']:>4}  {r['G']:>+5.2f} {r['D']:>4.2f} {r['Y']:>+5.2f}  "
            f"{row.min():.2f} {row.mean():.2f} {row.max():.2f}   {flag}"
        )
    print(
        f"\n[exp1d] gate (>=7/10 above 0.7): {n_pass}/10  -> "
        f"{'PASS' if gate_pass else 'FAIL'}"
    )

    # Figure: per-encounter heatmap of pairwise Spearman.
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, ax in enumerate(axes.flat):
        r = per_enc_records[i]
        m = np.full((len(SEEDS), len(SEEDS)), np.nan)
        for a in range(len(SEEDS)):
            for b in range(len(SEEDS)):
                if a == b:
                    m[a, b] = 1.0
                else:
                    sa, sb = SEEDS[a], SEEDS[b]
                    key = f"{sa}_vs_{sb}" if (
                        f"{sa}_vs_{sb}" in r["spearman_per_pair"]
                    ) else f"{sb}_vs_{sa}"
                    m[a, b] = r["spearman_per_pair"][key]
        im = ax.imshow(m, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(SEEDS)))
        ax.set_yticks(range(len(SEEDS)))
        ax.set_xticklabels(SEEDS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(SEEDS, fontsize=8)
        ax.set_title(
            f"idx={r['test_b_idx']} {r['label']}\n"
            f"mean={r['mean_spearman']:.2f}",
            fontsize=9,
        )
        for a in range(len(SEEDS)):
            for b in range(len(SEEDS)):
                if a == b:
                    continue
                ax.text(b, a, f"{m[a, b]:.2f}", ha="center", va="center",
                        color="white" if abs(m[a, b]) > 0.5 else "black",
                        fontsize=8)
    fig.suptitle(
        "Cross-seed Spearman correlation of normalised distance matrices\n"
        f"(gate: mean > 0.7 on >= 7/10; got {n_pass}/10)"
    )
    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    fig.savefig(FIGS / "exp1_cross_seed_distance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp1d] wrote {FIGS / 'exp1_cross_seed_distance.png'}")


if __name__ == "__main__":
    main()

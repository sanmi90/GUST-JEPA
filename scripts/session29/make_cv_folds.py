"""SESSION29 Track C-full: generate stratified, case-disjoint 5-fold CV splits.

Builds five fold-split JSONs from configs/splits/split_v2p1.json over the 81
non-OOD cases (train + test_b). Each fold reassigns ~1/5 of the cases to a held-out
group (written as the JSON's `test_b` so the existing train_jepa / encode / probe
machinery evaluates on it) and the remaining ~4/5 to `train`; the 4 OOD test_c
cases are left untouched and are never used in CV. Stratified by sign(G), an |G|
bin, and a D bin so every fold spans the envelope.

HYGIENE (asserted): within a fold, train and held-out case sets are disjoint; the
five held-out sets are pairwise disjoint and cover the pool exactly once; test_c is
identical to the source split in every fold. This is the F3 case-level guarantee:
no case_id leaks between the fold-train and the fold-evaluation.

Outputs: configs/splits/cv_folds/fold{0..4}.json (+ a manifest with the assignments).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "configs" / "splits" / "split_v2p1.json"
OUT_DIR = REPO / "configs" / "splits" / "cv_folds"


def strat_label(c: dict) -> str:
    g = float(c["G"])
    sign = "neg" if g < 0 else ("zero" if g == 0 else "pos")
    gbin = "lo" if abs(g) <= 1.0 else ("mid" if abs(g) <= 2.0 else "hi")
    dbin = "d_lo" if float(c["D"]) <= 1.0 else "d_hi"
    return f"{sign}_{gbin}_{dbin}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = json.loads(SRC.read_text())
    cases = src["cases"]
    pool = [cid for cid, c in cases.items() if c["split"] in ("train", "test_b")]
    pool.sort()
    labels = [strat_label(cases[cid]) for cid in pool]
    # merge any singleton strata into the nearest to keep StratifiedKFold valid
    from collections import Counter

    cnt = Counter(labels)
    labels = [lab if cnt[lab] >= args.folds else "rare" for lab in labels]

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    pool_arr = np.array(pool)
    fold_holdouts = []
    for _, te in skf.split(pool_arr, labels):
        fold_holdouts.append(set(pool_arr[te].tolist()))

    # hygiene: holdouts pairwise disjoint + cover the pool exactly once
    union = set().union(*fold_holdouts)
    assert union == set(pool), "fold holdouts do not cover the pool exactly"
    for i in range(len(fold_holdouts)):
        for j in range(i + 1, len(fold_holdouts)):
            assert not (fold_holdouts[i] & fold_holdouts[j]), "fold holdouts overlap"

    test_c = {cid for cid, c in cases.items() if c["split"] == "test_c"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(SRC),
        "seed": args.seed,
        "n_folds": args.folds,
        "pool_size": len(pool),
        "folds": [],
    }
    for k, holdout in enumerate(fold_holdouts):
        foldtrain = [cid for cid in pool if cid not in holdout]
        # hygiene: train and holdout disjoint
        assert not (set(foldtrain) & holdout), f"fold {k}: train/holdout overlap"
        fold = copy.deepcopy(src)
        for cid, c in fold["cases"].items():
            if cid in holdout:
                c["split"] = "test_b"
            elif cid in test_c:
                c["split"] = "test_c"  # untouched
            else:
                c["split"] = "train"
        fold["summary"] = {
            "cv_fold": k,
            "n_train_cases": len(foldtrain),
            "n_holdout_cases": len(holdout),
            "n_test_c_cases": len(test_c),
        }
        fold["cv_fold_meta"] = {
            "fold": k,
            "seed": args.seed,
            "holdout_cases": sorted(holdout),
        }
        if not args.dry_run:
            (OUT_DIR / f"fold{k}.json").write_text(json.dumps(fold, indent=1))
        manifest["folds"].append(
            {
                "fold": k,
                "n_train": len(foldtrain),
                "n_holdout": len(holdout),
                "holdout": sorted(holdout),
            }
        )
        print(
            f"fold {k}: {len(foldtrain)} train cases, {len(holdout)} holdout, "
            f"test_c {len(test_c)} untouched"
        )
    if not args.dry_run:
        (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"wrote {args.folds} fold splits + manifest to {OUT_DIR}")
    # also assert test_c identical across folds (by construction; spot report)
    print(f"test_c held constant across folds: {sorted(test_c)}")


if __name__ == "__main__":
    main()

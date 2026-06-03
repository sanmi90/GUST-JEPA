#!/usr/bin/env python3
"""
run_causal_analysis.py
======================

Orchestrates the information-theoretic / causal analysis for the vortex-jepa
manuscript. Two modes:

  --synthetic   run on a built-in synthetic design with known ground truth. No
                real data needed. Use this first to confirm the pipeline executes
                and reproduces the expected qualitative pattern (future lift
                ~unique to parameters; future wake ~synergistic in
                (parameters, current wake state); JEPA latent most observable of
                the future wake).

  --real        wire to the actual cache/artifacts via infotheory.io_vortex. You
                MUST pass --split and --latent and confirm the SCHEMA HOOKs first;
                the script runs validate_against_known() as a gate before any
                causal number is produced.

This is deliberately structured as the "one-day de-risking experiment" described
in review: the SURD-on-observables block (Block A) uses NO model and answers, at
almost no cost, whether the synergy story holds. If it does, Blocks B and C
(latent observability; staged SURD) turn it into manuscript figures. If it does
not, you have learned in a day that the causal section will not add much, and you
have spent no manuscript length on it.

Outputs: a JSON results bundle + human-readable tables to --out (default
./outputs_causal/). Reproducible: fixed --seed threaded into every estimator.

NOTE ON HONESTY (matches the manuscript's stance): SURD/IND are OBSERVATIONAL.
They characterise the information structure of the data and of the latents. They
do NOT validate the predictor as a counterfactual operator; the interventional
test (HANDOFF D154) already failed and that conclusion stands. Frame any result
from this script as "what causes the future observable" and "which latent is
observable of it", not as evidence for an interventional / world-model claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# allow running as a plain script: put the package root (parent of this file's
# parent) on sys.path so `import infotheory` works without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infotheory import io_vortex
from infotheory.estimators import quantise, joint_pmf
from infotheory.observability import observability_table, pretty_print_observability
from infotheory.surd import surd_discrete, surd_by_stage, pretty_print_surd


# --------------------------------------------------------------------------- #
def block_a_surd_on_observables(design: io_vortex.CausalDesign, args) -> dict:
    """
    Block A (the de-risking core): SURD of each future observable from the
    physically interpretable scalar sources, using NO model.

    Targets : future wake enstrophy, future lift.
    Sources : a 2-source set {gust strength G, current wake enstrophy} for the
              clean 2-source lattice (validated against canonical gates), and the
              parameter-vs-wake-state contrast the manuscript needs.

    The headline test: for target=future wake enstrophy, is the SYNERGISTIC term
    S[G + wake_now] a substantial fraction of H(target), while for target=future
    lift the UNIQUE term U[G] dominates?
    """
    tr = design.subset("train")  # use the large training pool for estimation power
    nb = args.bins
    src_names = (args.source_a, args.source_b)
    s1 = quantise(tr.sources_impact[src_names[0]], nb)
    s2 = quantise(tr.sources_impact[src_names[1]], nb)

    results = {}
    for tname in args.targets:
        t = quantise(tr.targets_future[tname], nb)
        labels = np.column_stack([t, s1, s2])
        p = joint_pmf(labels, (nb, nb, nb))
        res = surd_discrete(p)
        names = {1: src_names[0], 2: src_names[1]}
        results[tname] = {
            "pretty": pretty_print_surd(res, names),
            "normalised": res.normalised(),
            "info_leak": res.info_leak,
            "mi_total_bits": res.mi_total,
            "h_target_bits": res.h_target,
        }
    return results


def block_b_latent_observability(design: io_vortex.CausalDesign, args) -> dict:
    """
    Block B: information-theoretic observability O = I(T;S)/H(T) of each latent
    family (and each pressure set) toward each future observable. This is the
    principled version of the manuscript's Section 4.7 recoverability result and
    of the forecast ordering: predict that the predictive (JEPA) latent has the
    highest O toward the future WAKE while the reconstructive/POD latents are
    markedly lower, on estimator-independent ground.

    Computed on test_b (held-out) to mirror the manuscript's held-out reporting;
    falls back to all rows if test_b is too small.
    """
    te = design.subset("test_b")
    base = te if len(te.keys) >= 100 else design
    finite = {}
    for fam, mat in {**base.latents, **{f"pressure_{k}": v for k, v in base.pressure.items()}}.items():
        ok = np.all(np.isfinite(mat), axis=1)
        if ok.sum() >= 80:
            finite[fam] = mat[ok]
    # targets restricted to the same finite rows is approximate across families;
    # for a strict comparison, intersect masks. Here we use full targets and rely
    # on each family having near-complete coverage (validated by attach_*).
    targets = {t: base.targets_future[t] for t in args.targets if t in base.targets_future}
    rows = observability_table(
        finite,
        {t: targets[t] for t in targets},
        k=args.knn_k,
        n_bins=args.bins,
        n_surrogate=args.surrogate,
        random_state=args.seed,
    )
    return {
        "pretty": pretty_print_observability(rows),
        "rows": [r.__dict__ for r in rows],
    }


def block_c_staged_surd(design: io_vortex.CausalDesign, args) -> dict:
    """
    Block C: staged SURD (state-aware stand-in) of the future wake enstrophy. The
    manuscript asserts the causal structure shifts from parameter-dominated at
    impact to wake-state-dominated at the horizon (the conditioning-floor argument
    in Section 4.1). This block quantifies that shift per stage. The released
    state-aware SURD (ALD-Lab/SURD-states) is the rigorous tool; this is the cheap
    in-repo stand-in.
    """
    if design.stage_index is None:
        return {"skipped": "no stage_index; provide phase_at_impact or a per-frame stage loader."}
    tr = design.subset("train")
    if tr.stage_index is None:
        return {"skipped": "no stage_index on train subset."}
    nb = args.bins
    t = quantise(tr.targets_future[args.targets[0]], nb)
    s = np.column_stack(
        [quantise(tr.sources_impact[args.source_a], nb), quantise(tr.sources_impact[args.source_b], nb)]
    )
    staged = surd_by_stage(t, s, tr.stage_index, (nb, nb, nb))
    names = {1: args.source_a, 2: args.source_b}
    return {stage: {"pretty": pretty_print_surd(r, names), "normalised": r.normalised()}
            for stage, r in staged.items()}


# --------------------------------------------------------------------------- #
def load_real_design(args) -> io_vortex.CausalDesign:
    cache = io_vortex.resolve_cache_root()
    split = io_vortex.load_split(args.split)
    design = io_vortex.load_observables(cache, split, horizon=args.horizon, partition=args.partition)
    if args.latent:
        latent_files = dict(pair.split("=", 1) for pair in args.latent)
        design = io_vortex.attach_latents(design, latent_files)
    if args.pressure:
        pressure_files = dict(pair.split("=", 1) for pair in args.pressure)
        design = io_vortex.attach_pressure(design, pressure_files)
    design = io_vortex.assign_stages_from_phase(design)
    # GATE: catch stale schema hooks before producing any causal number
    if args.latent and not args.skip_validation:
        fam = next(iter(dict(pair.split("=", 1) for pair in args.latent)))
        try:
            io_vortex.validate_against_known(design, latent_family=fam)
            print(f"[gate] validate_against_known passed for {fam}.")
        except AssertionError as e:
            print(f"[gate] FAILED: {e}")
            print("[gate] Fix the SCHEMA HOOKs in io_vortex before trusting causal numbers.")
            raise
    return design


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true", help="run on built-in synthetic design")
    mode.add_argument("--real", action="store_true", help="run on the real cache/artifacts")

    ap.add_argument("--split", type=str, help="path to configs/splits/split_v1.json (real mode)")
    ap.add_argument("--latent", action="append", default=[],
                    help="family=path.npz (repeatable), e.g. JEPA_d64=runs/.../latents.npz")
    ap.add_argument("--pressure", action="append", default=[],
                    help="tag=path.npz (repeatable), e.g. K8=runs/.../pressure_k8.npz")
    ap.add_argument("--partition", type=str, default="v1")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--skip-validation", action="store_true",
                    help="skip the validate_against_known gate (NOT recommended)")

    ap.add_argument("--targets", nargs="+",
                    default=["wake_enstrophy_future", "CL_future"])
    ap.add_argument("--source-a", type=str, default="G", help="first SURD source")
    ap.add_argument("--source-b", type=str, default="wake_enstrophy_impact",
                    help="second SURD source (parameter-vs-state contrast)")
    ap.add_argument("--bins", type=int, default=6, help="quantile bins for discrete SURD/hist MI")
    ap.add_argument("--knn-k", type=int, default=4)
    ap.add_argument("--surrogate", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="./outputs_causal")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        design = io_vortex.make_synthetic_design(random_state=args.seed)
        print(f"[mode] synthetic design: {len(design.keys)} rows, "
              f"latents={list(design.latents)}")
    else:
        if not args.split:
            ap.error("--real requires --split")
        design = load_real_design(args)
        print(f"[mode] real design: {len(design.keys)} rows across "
              f"{sorted(set(design.partition))}")

    bundle = {"args": vars(args)}

    print("\n" + "=" * 72 + "\nBLOCK A  SURD of future observables from physical sources (no model)\n" + "=" * 72)
    a = block_a_surd_on_observables(design, args)
    for tname, r in a.items():
        print(f"\n--- target: {tname} ---")
        print(r["pretty"])
    bundle["block_a_surd_observables"] = a

    print("\n" + "=" * 72 + "\nBLOCK B  Information-theoretic observability of latents/pressure\n" + "=" * 72)
    b = block_b_latent_observability(design, args)
    print(b["pretty"])
    bundle["block_b_observability"] = b

    print("\n" + "=" * 72 + "\nBLOCK C  Staged SURD of the future wake (state-aware stand-in)\n" + "=" * 72)
    c = block_c_staged_surd(design, args)
    if "skipped" in c:
        print("  skipped:", c["skipped"])
    else:
        for stage, r in c.items():
            print(f"\n--- stage: {stage} ---")
            print(r["pretty"])
    bundle["block_c_staged_surd"] = c

    # JSON bundle (drop the non-serialisable pretty strings into a separate .txt)
    serialisable = json.loads(json.dumps(bundle, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    (out / "causal_results.json").write_text(json.dumps(serialisable, indent=2))
    print(f"\n[done] wrote {out/'causal_results.json'}")
    print("[reminder] SURD/IND are observational; do not read these as an "
          "interventional or world-model validation (see HANDOFF D154).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""SESSION29 Track C-min: per-CASE slopegraph for the headline wake result (referee F3).

The encounter-level bars overstate the predictive (JEPA) advantage because the 42
held-out test_b encounters are not independent: they cluster into 10 cases. This
script collapses to ONE point per case and runs the case-level paired location test
that the referee asked for. The headline wake observable is the representational
wake-enstrophy readout at H = 16 on test_b (split v2.1, seed 42).

Source of truth: outputs/session28/closure_matrix/per_encounter/dump.npz already
holds the per-encounter (y_true, y_pred, case_id, encounter_index) for both families
on this exact cell. We read it directly. A --recompute flag re-derives the same
arrays from the frozen latents (cm.load_family) + the dump's own DNS metrics file,
reproducing the closure-matrix pooled-frame Ridge probe, as a cross-check / fallback.

Per-encounter absolute error = |y_true - y_pred|. Per-case error = mean over that
case's test_b encounters (cm.stats_lib.case_means). The paired test is on the 10
case-mean deltas delta = err_fukami - err_jepa (positive favours JEPA): an exact
one-sided sign test and a one-sided Wilcoxon signed-rank, BOTH via
cm.stats_lib.case_level_paired_stats. We do NOT use case_permutation_p here: it is a
DEPENDENCE test, degenerate for a paired location shift (the B6 lesson).

VERDICT: STRONG if the case-level paired test confirms the advantage (sign test AND
signed-rank significant at 0.05); WEAK if the advantage survives only the
encounter-level sign test (the known v2.1 state). Either way the slopegraph shows the
honest spread over 10 cases.

Outputs (under outputs/session29/):
    fig_case_slopegraph.{pdf,png}  -- one line per case, AE (left) -> JEPA (right)
    case_slopegraph.json           -- per-case errors + paired-test stats + provenance
    case_slopegraph.md             -- human-readable summary

CPU only. Honours nice -n 15 from the launcher. Fail-loud on any missing input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session29"))
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import _s29_common as cm  # noqa: E402

OUT_DIR = REPO / "outputs" / "session29"
DUMP = REPO / "outputs" / "session28" / "closure_matrix" / "per_encounter" / "dump.npz"

# Headline cell. The wake-enstrophy representational readout at H=16 on test_b is
# the figure the encounter-level bars currently carry (referee F3).
OBSERVABLE = "wake_enstrophy"
HORIZON = 16
SPLIT = "test_b"
ENDPOINT = "representational"
PROBE = "ridge"
TIER = "pooled"
JEPA_TAG = "jepa_tf_noc_d64_s42"
FUKAMI_TAG = "fukami_d64_s42"

# Axis unit: enstrophy of the pipeline-normalised omega in the wake region
# (dimensionless). The readout target is in those same normalised units.
OBS_LABEL = r"wake enstrophy $\Omega_w$ (normalised units)"


def _dump_base(tag: str) -> str:
    return f"{tag}|{OBSERVABLE}|{ENDPOINT}|{PROBE}|{SPLIT}|{TIER}|H{HORIZON}"


def load_from_dump(dump_path: Path) -> dict:
    """Read aligned per-encounter (y_true, y_pred, case_id, enc) for both families."""
    if not dump_path.exists():
        raise FileNotFoundError(
            f"per-encounter dump missing: {dump_path}\n"
            "Re-run with --recompute to derive it from frozen latents instead."
        )
    d = np.load(dump_path, allow_pickle=True)
    bj, bf = _dump_base(JEPA_TAG), _dump_base(FUKAMI_TAG)
    needed = [
        f"{b}__{s}" for b in (bj, bf) for s in ("y_pred", "y_true", "case_id", "encounter_index")
    ]
    missing = [k for k in needed if k not in d]
    if missing:
        raise KeyError(
            f"dump {dump_path} is missing keys for the headline cell:\n  "
            + "\n  ".join(missing)
            + "\nRe-run with --recompute as a fallback."
        )
    cid_j = np.asarray(d[f"{bj}__case_id"]).astype(str)
    enc_j = np.asarray(d[f"{bj}__encounter_index"]).astype(int)
    cid_f = np.asarray(d[f"{bf}__case_id"]).astype(str)
    enc_f = np.asarray(d[f"{bf}__encounter_index"]).astype(int)
    keys_j = list(zip(cid_j.tolist(), enc_j.tolist()))
    keys_f = list(zip(cid_f.tolist(), enc_f.tolist()))
    if keys_j != keys_f:
        raise ValueError(
            "JEPA and reconstructive per-encounter rows are not aligned in the dump; "
            "refusing to pair. Use --recompute."
        )
    yt_j = np.asarray(d[f"{bj}__y_true"], dtype=float)
    yt_f = np.asarray(d[f"{bf}__y_true"], dtype=float)
    if not np.allclose(yt_j, yt_f, equal_nan=True):
        raise ValueError(
            "y_true differs between families on the same encounters; the DNS target "
            "should be family-independent. Refusing to pair."
        )
    return {
        "case_id": cid_j,
        "encounter_index": enc_j,
        "y_true": yt_j,
        "y_pred_jepa": np.asarray(d[f"{bj}__y_pred"], dtype=float),
        "y_pred_fukami": np.asarray(d[f"{bf}__y_pred"], dtype=float),
        "source": "dump",
    }


DNS_METRICS = REPO / "outputs" / "session28" / "exp2" / "dns_physical_metrics.npz"


def _pooled_frame_xy(tag: str, split: str, dns):
    """Pooled-over-frames (z, y) for the representational readout, matching the dump.

    The closure-matrix 'representational' probe is fit on ALL frames pooled
    (z_full reshaped to (n*T, d), y the DNS observable at EVERY frame), NOT the
    single readout frame. Held-out predictions are then taken at frame impact+H.
    The DNS observable is read from the SAME source the dump used
    (outputs/session28/exp2/dns_physical_metrics.npz) via closure_matrix's own
    dns_split_array + match_index, so the recompute is byte-faithful to the dump
    rather than an independent readout-frame probe on a different DNS target file.

    Returns (z_pool (m*T, d), y_pool (m*T,), z_h (m, d) at impact+H, y_h (m,),
    case_ids (m,), enc (m,)) where m is the DNS-matched encounter count.
    """
    import closure_matrix as clm

    fam = cm.load_family(tag, split)
    z_full = fam["z_full"].astype(np.float64)  # (n, T, d)
    cid = fam["case_ids"]
    enc = fam["encounter_indices"]
    imp = fam["impact_frame"]
    n, t_len, _ = z_full.shape

    # Align to the DNS metrics file exactly as closure_matrix.load_split_data does.
    di = clm.match_index(
        cid,
        enc,
        clm.dns_split_array(dns, split, "case_id"),
        clm.dns_split_array(dns, split, "encounter_index"),
    )
    keep = di >= 0
    z_full, cid, enc, imp, di = (
        z_full[keep],
        cid[keep],
        enc[keep],
        imp[keep],
        di[keep],
    )
    y_all = clm.dns_split_array(dns, split, OBSERVABLE)[di].astype(np.float64)  # (m, T)

    m = z_full.shape[0]
    z_pool = z_full.reshape(m * t_len, z_full.shape[2])
    y_pool = y_all[:, :t_len].reshape(m * t_len)

    z_h, y_h, keep_cid, keep_enc = [], [], [], []
    for i in range(m):
        fr = int(imp[i]) + HORIZON
        if fr < t_len:
            z_h.append(z_full[i, fr, :])
            y_h.append(float(y_all[i, fr]))
            keep_cid.append(cid[i])
            keep_enc.append(int(enc[i]))
    return (
        z_pool,
        y_pool,
        np.asarray(z_h),
        np.asarray(y_h),
        np.asarray(keep_cid).astype(str),
        np.asarray(keep_enc, dtype=int),
    )


def load_recompute() -> dict:
    """Faithful fallback: reproduce the closure-matrix pooled-frame Ridge probe.

    Fits closure_matrix.fit_ridge on the TRAIN pooled-frame latents, then applies it
    at the test_b readout frame impact+H, reading the DNS observable from the dump's
    own DNS metrics file. This reproduces the dump's representational/ridge cell, so
    it cross-checks the dump rather than introducing a different probe regime.
    """
    import closure_matrix as clm  # session28 path on sys.path via _s29_common

    if not DNS_METRICS.exists():
        raise FileNotFoundError(f"DNS metrics for recompute missing: {DNS_METRICS}")
    dns = np.load(DNS_METRICS, allow_pickle=True)

    def fit_predict(tag: str):
        ztr, ytr, *_ = _pooled_frame_xy(tag, "train", dns)
        _, _, zh, yh, gte, ete = _pooled_frame_xy(tag, SPLIT, dns)
        probe = clm.fit_ridge(ztr, ytr, alpha=1.0)
        pred = clm.apply_probe(zh, probe)
        return yh, pred, gte, ete

    yt_j, yp_j, g_j, e_j = fit_predict(JEPA_TAG)
    yt_f, yp_f, g_f, e_f = fit_predict(FUKAMI_TAG)
    if list(zip(g_j.tolist(), e_j.tolist())) != list(zip(g_f.tolist(), e_f.tolist())):
        raise ValueError("recompute: test_b rows not aligned between families.")
    if not np.allclose(yt_j, yt_f):
        raise ValueError("recompute: DNS y_true differs between families.")
    return {
        "case_id": np.asarray(g_j).astype(str),
        "encounter_index": np.asarray(e_j, dtype=int),
        "y_true": np.asarray(yt_j, dtype=float),
        "y_pred_jepa": np.asarray(yp_j, dtype=float),
        "y_pred_fukami": np.asarray(yp_f, dtype=float),
        "source": "recompute",
    }


def compute(data: dict) -> dict:
    """Per-case mean absolute errors + paired case-level + encounter-level stats."""
    cid = data["case_id"]
    ae_j = np.abs(data["y_true"] - data["y_pred_jepa"])
    ae_f = np.abs(data["y_true"] - data["y_pred_fukami"])

    uc, cm_j = cm.stats_lib.case_means(ae_j, cid)
    _, cm_f = cm.stats_lib.case_means(ae_f, cid)
    case_delta = cm_f - cm_j  # positive favours JEPA

    case_paired = cm.stats_lib.case_level_paired_stats(case_delta)

    # Encounter-level sign test (jepa better == smaller abs error) for context only.
    enc_k, enc_n, enc_p = cm.stats_lib.sign_test_one_sided(ae_j, ae_f)

    per_case = []
    for c, f, j in zip(uc, cm_f, cm_j):
        per_case.append(
            {
                "case_id": str(c),
                "n_encounters": int(np.sum(cid == c)),
                "err_fukami": float(f),
                "err_jepa": float(j),
                "delta_fukami_minus_jepa": float(f - j),
                "jepa_wins": bool(j < f),
            }
        )

    # VERDICT: STRONG requires the case-level paired test to confirm (both the exact
    # sign test AND the signed-rank significant at 0.05). WEAK = survives the
    # encounter-level sign test but not robustly at case level.
    sign_p = float(case_paired["sign_p_one_sided"])
    wilcoxon_p = (
        float(case_paired["wilcoxon_p_one_sided"])
        if isinstance(case_paired["wilcoxon_p_one_sided"], (int, float))
        else float("nan")
    )
    case_confirms = (sign_p <= 0.05) and (wilcoxon_p <= 0.05)
    enc_confirms = enc_p <= 0.05
    verdict = "STRONG" if case_confirms else ("WEAK" if enc_confirms else "NULL")

    return {
        "per_case": per_case,
        "case_unique": [str(c) for c in uc],
        "err_jepa_by_case": [float(v) for v in cm_j],
        "err_fukami_by_case": [float(v) for v in cm_f],
        "case_mean_delta": [float(v) for v in case_delta],
        "case_level_paired": case_paired,
        "encounter_level_sign_test": {
            "k_jepa_better": enc_k,
            "n_effective": enc_n,
            "p_one_sided": float(enc_p),
        },
        "mean_encounter_delta": float((ae_f - ae_j).mean()),
        "mean_case_delta": float(case_delta.mean()),
        "verdict": verdict,
        "verdict_basis": {
            "case_level_confirms": bool(case_confirms),
            "encounter_level_confirms": bool(enc_confirms),
        },
    }


def make_figure(result: dict, out_pdf: Path, out_png: Path) -> None:
    import figstyle

    figstyle.use_style()
    plt = figstyle.plt

    win_color = figstyle.FAMILY_COLOR["jepa"]  # JEPA-green when JEPA wins the case
    lose_color = figstyle.FAMILY_COLOR["fukami"]  # reconstructive-red when it does not

    fig, ax = plt.subplots(figsize=figstyle.figure_size(0.62, aspect=0.95))
    x_left, x_right = 0.0, 1.0
    for pc in result["per_case"]:
        c = win_color if pc["jepa_wins"] else lose_color
        ax.plot(
            [x_left, x_right],
            [pc["err_fukami"], pc["err_jepa"]],
            "-",
            color=c,
            lw=1.1,
            alpha=0.85,
            marker="o",
            ms=3.2,
            zorder=2,
        )

    ax.set_xlim(-0.18, 1.18)
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels([figstyle.FAMILY_LABEL["fukami"], figstyle.FAMILY_LABEL["jepa"]])
    ax.set_ylabel("per-case mean abs. error in " + OBS_LABEL)
    ax.margins(y=0.05)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    n_case = len(result["per_case"])
    n_win = sum(pc["jepa_wins"] for pc in result["per_case"])
    handles = [
        plt.Line2D(
            [],
            [],
            color=win_color,
            marker="o",
            ms=3.2,
            lw=1.1,
            label=f"predictive lower ({n_win}/{n_case})",
        ),
        plt.Line2D(
            [],
            [],
            color=lose_color,
            marker="o",
            ms=3.2,
            lw=1.1,
            label=f"reconstructive lower ({n_case - n_win}/{n_case})",
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", fontsize=7)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_markdown(result: dict, out_md: Path, prov: dict) -> None:
    cp = result["case_level_paired"]
    enc = result["encounter_level_sign_test"]
    lines = [
        "# SESSION29 Track C-min: per-case wake-enstrophy slopegraph",
        "",
        f"Headline cell: {OBSERVABLE} {ENDPOINT} readout ({PROBE}) at H={HORIZON} on "
        f"{SPLIT} (split v2.1). Predictive = `{JEPA_TAG}`; reconstructive = "
        f"`{FUKAMI_TAG}`. Source: {result.get('source', prov.get('source', 'dump'))}.",
        "",
        f"**VERDICT: {result['verdict']}**  "
        f"(case-level confirms = {result['verdict_basis']['case_level_confirms']}; "
        f"encounter-level confirms = {result['verdict_basis']['encounter_level_confirms']})",
        "",
        "## Per-case mean absolute error (normalised wake-enstrophy units)",
        "",
        "| case | n_enc | reconstructive | predictive (JEPA) | delta (fuk-jepa) | JEPA wins |",
        "|---|---|---|---|---|---|",
    ]
    for pc in result["per_case"]:
        lines.append(
            f"| {pc['case_id']} | {pc['n_encounters']} | {pc['err_fukami']:.3f} | "
            f"{pc['err_jepa']:.3f} | {pc['delta_fukami_minus_jepa']:+.3f} | "
            f"{'yes' if pc['jepa_wins'] else 'no'} |"
        )
    lines += [
        "",
        "## Case-level paired test (delta = reconstructive - predictive, >0 favours JEPA)",
        "",
        f"- cases where JEPA is better: {cp['cases_jepa_better']}/{cp['n_cases']}",
        f"- exact one-sided sign test p = {cp['sign_p_one_sided']:.4f}",
        f"- Wilcoxon signed-rank (one-sided) stat = {cp['wilcoxon_stat']}, "
        f"p = {cp['wilcoxon_p_one_sided']}",
        f"- median per-case delta = {cp['median_case_mean_delta']:.3f}",
        f"- mean per-case delta = {result['mean_case_delta']:.3f}",
        "",
        "## Encounter-level sign test (context only; 42 encounters)",
        "",
        f"- JEPA better in {enc['k_jepa_better']}/{enc['n_effective']} encounters, "
        f"p = {enc['p_one_sided']:.4f}",
        f"- mean per-encounter delta = {result['mean_encounter_delta']:.3f}",
        "",
        "We do NOT use the case-permutation p here: it is a dependence test, "
        "degenerate for a paired location shift (the B6 lesson).",
        "",
        f"Provenance: git {prov['git_sha'][:12]}, {prov['utc']}.",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--recompute",
        action="store_true",
        help="re-derive per-encounter preds from frozen latents instead of the dump.",
    )
    ap.add_argument("--dump", type=Path, default=DUMP, help="per-encounter dump.npz path.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the plan; write nothing.",
    )
    args = ap.parse_args()

    out_json = args.out_dir / "case_slopegraph.json"
    out_md = args.out_dir / "case_slopegraph.md"
    out_pdf = args.out_dir / "fig_case_slopegraph.pdf"
    out_png = args.out_dir / "fig_case_slopegraph.png"

    if args.dry_run:
        print("[dry-run] SESSION29 Track C-min case slopegraph")
        print(f"  headline cell : {OBSERVABLE} {ENDPOINT} {PROBE} H={HORIZON} {SPLIT}")
        print(f"  predictive    : {JEPA_TAG}")
        print(f"  reconstructive: {FUKAMI_TAG}")
        print(f"  source        : {'recompute (latents)' if args.recompute else 'dump'}")
        if args.recompute:
            if not DNS_METRICS.exists():
                raise FileNotFoundError(f"[dry-run] missing DNS metrics: {DNS_METRICS}")
            print(f"  ok dns        : {DNS_METRICS}")
            for tag in (JEPA_TAG, FUKAMI_TAG):
                for sp in ("train", SPLIT):
                    p = cm.LATENTS / tag / f"{sp}.npz"
                    if not p.exists():
                        raise FileNotFoundError(f"[dry-run] missing latents: {p}")
                    print(f"  ok latents    : {p}")
        else:
            if not args.dump.exists():
                raise FileNotFoundError(f"[dry-run] missing dump: {args.dump}")
            load_from_dump(args.dump)
            print(f"  ok dump       : {args.dump}")
        print(f"  would write   : {out_json}")
        print(f"                  {out_md}")
        print(f"                  {out_pdf}")
        print(f"                  {out_png}")
        return 0

    data = load_recompute() if args.recompute else load_from_dump(args.dump)
    result = compute(data)
    result["source"] = data["source"]

    inputs = (
        [args.dump]
        if not args.recompute
        else [
            DNS_METRICS,
            cm.LATENTS / JEPA_TAG / "train.npz",
            cm.LATENTS / JEPA_TAG / f"{SPLIT}.npz",
            cm.LATENTS / FUKAMI_TAG / "train.npz",
            cm.LATENTS / FUKAMI_TAG / f"{SPLIT}.npz",
        ]
    )
    prov = cm.provenance(inputs, seed=42)
    prov["source"] = data["source"]

    payload = {
        "cell": {
            "observable": OBSERVABLE,
            "horizon": HORIZON,
            "split": SPLIT,
            "endpoint": ENDPOINT,
            "probe": PROBE,
            "tier": TIER,
            "jepa_tag": JEPA_TAG,
            "fukami_tag": FUKAMI_TAG,
        },
        **result,
        "provenance": prov,
    }
    cm.write_artifact(out_json, payload)
    write_markdown(result, out_md, prov)
    make_figure(result, out_pdf, out_png)

    cp = result["case_level_paired"]
    print(f"VERDICT: {result['verdict']}")
    print(
        f"  case-level: {cp['cases_jepa_better']}/{cp['n_cases']} cases favour JEPA; "
        f"sign p={cp['sign_p_one_sided']:.4f}; "
        f"signed-rank p={cp['wilcoxon_p_one_sided']}"
    )
    e = result["encounter_level_sign_test"]
    print(f"  encounter : {e['k_jepa_better']}/{e['n_effective']}; p={e['p_one_sided']:.4f}")
    print(f"  wrote {out_json}")
    print(f"  wrote {out_md}")
    print(f"  wrote {out_pdf} / {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

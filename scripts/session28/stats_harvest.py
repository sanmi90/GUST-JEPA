"""Session 28 B6: the core statistics harvest (master plan B6; closes S1-S3/S5-S7, D183-D187).

Consumes the B2 closure matrix (summary cells) and its per-encounter side-channel
(closure_matrix.py --dump-per-encounter -> per_encounter/dump.npz) and runs the
PAIRED tests the protocol pre-registers, calling scripts/session28/stats_lib.py
verbatim (no statistic is reimplemented here). Nothing in this script alters the
matrix summary outputs, the gate verdict, or numbers.json; it only adds a new
numbers PART (stats_harvest.json) and a human-readable report.

PROBE / ENDPOINT / PAIRING CONVENTIONS (frozen: configs/eval_protocol_v2p1.yaml)
    Primary endpoint    representational wake_enstrophy R^2 at H = 16, test_b
                        pooled, d = 64, ridge probe (D130/D165).
    Paired difference   delta_i = err_recon_i - err_pred_i on the per-encounter
                        ABSOLUTE error err = |y_pred - y_true|, exactly the D165
                        convention (scripts/session26/track1_stats.py delta = r - j;
                        regression-anchored in tests/test_stats_lib.py). delta > 0
                        means the PREDICTIVE family is better. Encounters are
                        matched on (case_id, encounter_index); the closure dump
                        already carries identical y_true per matched cell, so the
                        difference is well posed.
    Multiplicity        Holm over the 12-test family (6 observables x {repr,
                        forecast}, predictive jepa_tf_noc vs reconstructive Fukami
                        AE, test_b, H = 16). The Holm PRIMARY p is the case-
                        RESPECTING case-level one-sided sign-test p (one value per
                        case, honouring cluster_unit = case). The encounter-level
                        one-sided sign-test p (the exact D165 holm.json input) is
                        Holm-corrected alongside for continuity. stats_lib's
                        case_permutation_p is reserved for the Fig-8 Spearman
                        trends in block (d): a pooled mean-difference statistic is
                        invariant under its block permutation, so it cannot serve
                        the paired location test (it would give p ~ 1).

The five harvest blocks (master plan B6):
    (a) the 12-test Holm family, case-clustered CI + case-respecting sign p, survivors;
    (b) per-family seed-aggregated verdicts on the primary cell (predictive vs
        each baseline family), case-clustered CI, which separations survive;
    (c) Gate-GD hardening: is the WEAK-branch trigger statistically supported, or a
        point-estimate artifact? Paired predictive-minus-reconstructive wake
        difference under each NONLINEAR probe (krr_rbf, mlp_reg) with case-clustered
        CI, against the matched-head reconstructive AE (ctrl_recon_cnn);
    (d) Fig-8 trends: Spearman of the per-encounter representational wake closure
        vs |G| and vs D across test_b + test_c for the lead family, with
        case-permutation p;
    (e) emit the numbers PART (outputs/session28/numbers_parts/stats_harvest.json,
        eval_all.py format) and the report (outputs/session28/stats_harvest_report.md).

Usage (CPU-only; numpy/scipy on the NPZ dump):
    python scripts/session28/stats_harvest.py
    python scripts/session28/stats_harvest.py --check   # compute + print, do not write
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stats_lib  # noqa: E402

# The 6 protocol observables, in the protocol's declared order, with the short
# D165 labels used as the Holm-family test keys.
OBSERVABLES = ["C_L", "C_D", "I_y", "wake_enstrophy", "circulation_pos", "circulation_neg"]
SHORT = {
    "C_L": "CL",
    "C_D": "CD",
    "I_y": "Iy",
    "wake_enstrophy": "wake_enstrophy",
    "circulation_pos": "circ_pos",
    "circulation_neg": "circ_neg",
}
ENDPOINT_TAG = {"representational": "repr", "forecast": "forecast"}

# Family identities for the harvest. Lead member (seed 42 where present) drives the
# paired per-encounter tests; the matrix summary drives the seed-aggregated verdicts.
PREDICTIVE_LEAD = "jepa_tf_noc_d64_s42"
RECON_FUKAMI_LEAD = "fukami_d64_s42"
RECON_MATCHED_HEAD = "ctrl_recon_cnn"  # the matched-head reconstructive AE (Gate-GD weak branch)
LEAD_FAMILY = "jepa_tf_noc"  # Fig-8 trend family

PRIMARY_H = 16
PRIMARY_OBS = "wake_enstrophy"
PRIMARY_PROBE = "ridge"
NONLINEAR_PROBES = ("krr_rbf", "mlp_reg")


# --------------------------------------------------------------------------- dump access


@dataclass
class DumpCell:
    """One per-encounter cell of the closure dump (predictions + matching keys)."""

    y_pred: np.ndarray
    y_true: np.ndarray
    case_id: np.ndarray
    encounter_index: np.ndarray


class PerEncounterDump:
    """Loader/indexer over outputs/.../closure_matrix/per_encounter/dump.npz."""

    def __init__(self, npz_path: Path, index_path: Path) -> None:
        self.blob = np.load(npz_path, allow_pickle=True)
        self.index: dict[str, dict] = json.loads(index_path.read_text())

    @staticmethod
    def key(tag: str, obs: str, endpoint: str, probe: str, split: str, tier: str, h: int) -> str:
        """Reproduce closure_matrix.dump_cell_key for a cell."""
        return "|".join([tag, obs, endpoint, probe, split, tier, f"H{h}"])

    def has(self, key: str) -> bool:
        return key in self.index

    def get(self, key: str) -> DumpCell:
        if key not in self.index:
            raise KeyError(f"dump cell {key!r} not present (have {len(self.index)} cells)")
        return DumpCell(
            y_pred=self.blob[f"{key}__y_pred"],
            y_true=self.blob[f"{key}__y_true"],
            case_id=self.blob[f"{key}__case_id"].astype(str),
            encounter_index=self.blob[f"{key}__encounter_index"].astype(np.int64),
        )


# ------------------------------------------------------------------ paired-difference core


def abs_error(cell: DumpCell) -> np.ndarray:
    """Per-encounter absolute error |y_pred - y_true| (D165 convention)."""
    return np.abs(cell.y_pred.astype(np.float64) - cell.y_true.astype(np.float64))


def match_cells(pred: DumpCell, recon: DumpCell) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align two cells on (case_id, encounter_index); return matched (err_pred, err_recon, case).

    The closure dump emits both families' cells over the SAME DNS-matched
    encounter set in the same order, so this is usually the identity match; the
    explicit join is defensive (and reorders if the two members ever differ).
    """
    pk = list(zip(pred.case_id.tolist(), pred.encounter_index.tolist()))
    rk = {(c, int(e)): i for i, (c, e) in enumerate(zip(recon.case_id, recon.encounter_index))}
    ep = abs_error(pred)
    er = abs_error(recon)
    rows_p, rows_r = [], []
    for i, key in enumerate(pk):
        j = rk.get((key[0], int(key[1])))
        if j is not None:
            rows_p.append(i)
            rows_r.append(j)
    rows_p = np.asarray(rows_p, dtype=np.int64)
    rows_r = np.asarray(rows_r, dtype=np.int64)
    return ep[rows_p], er[rows_r], pred.case_id[rows_p]


def paired_difference_block(pred: DumpCell, recon: DumpCell, boot_seed: int) -> dict:
    """Full paired-difference summary for one matched cell, via stats_lib verbatim.

    delta_i = err_recon_i - err_pred_i (>0 = predictive better). Reports:
        - encounter-level one-sided sign test p (D165's Holm input; reported for
          continuity with the v2 holm.json);
        - case-clustered bootstrap CI of delta (cluster_unit = case, n = 10000);
        - case-level paired Wilcoxon + one-sided sign test on per-case mean delta
          (the case-RESPECTING p, one value per case; this script's Holm input,
          honouring the protocol's cluster_unit = case).

    Note on permutation: stats_lib.case_permutation_p permutes one signal's case
    BLOCKS against the other's cases, so a pooled-mean-difference statistic is
    invariant under it (the same multiset is reassembled) and the permutation
    p degenerates to ~1. case_permutation_p is therefore the right tool for the
    Fig-8 dependence (Spearman) trends in block (d), NOT for this paired location
    shift; the case-level paired sign test is the faithful case-respecting p here.
    """
    err_pred, err_recon, case = match_cells(pred, recon)
    delta = err_recon - err_pred

    k, n_eff, sign_p = stats_lib.sign_test_one_sided(err_pred, err_recon)
    cc = stats_lib.case_cluster_bootstrap(
        delta, case, n_boot=stats_lib.N_BOOT_CASE, rng=np.random.default_rng(boot_seed)
    )
    _, cmd = stats_lib.case_means(delta, case)
    clp = stats_lib.case_level_paired_stats(cmd)
    return {
        "n_encounters": int(delta.size),
        "n_cases": int(cc["n_cases"]),
        "mean_pred_err": float(err_pred.mean()),
        "mean_recon_err": float(err_recon.mean()),
        "enc_mean_delta": cc["enc_mean"],
        "case_clustered_ci": cc["enc_mean_ci"],
        "case_mean_delta": cc["case_mean"],
        "case_mean_ci": cc["case_mean_ci"],
        "sign_k": int(k),
        "sign_n_eff": int(n_eff),
        "sign_p_one_sided": float(sign_p),
        "case_level_paired": clp,
        "case_level_sign_p": float(clp["sign_p_one_sided"]),
        "case_level_wilcoxon_p": clp["wilcoxon_p_one_sided"],
    }


# ----------------------------------------------------------------------- matrix access


def load_matrix(npz_path: Path) -> dict[str, np.ndarray]:
    """Load the closure matrix.npz columns into a dict of arrays."""
    blob = np.load(npz_path, allow_pickle=True)
    return {k: blob[k] for k in blob.files}


def matrix_select(m: dict[str, np.ndarray], **eq) -> np.ndarray:
    """Boolean mask selecting matrix rows where every column == the given value."""
    mask = np.ones(len(m["family"]), dtype=bool)
    for col, val in eq.items():
        mask &= m[col] == val
    return mask


# ------------------------------------------------------------------- harvest blocks (a)-(d)


def harvest_holm_family(dump: PerEncounterDump, cfg: argparse.Namespace) -> dict:
    """(a) The 12-test Holm family: predictive jepa_tf_noc vs reconstructive Fukami AE.

    For each of 6 observables x {representational, forecast} on test_b pooled at
    H = 16, d = 64, ridge: the paired per-encounter difference (Fukami AE minus
    jepa_tf_noc absolute error), a case-clustered CI, and TWO one-sided p-values:
    the encounter-level sign-test p (the exact pre-registered D165 holm.json input,
    so the HEADLINE survivor count reproduces that procedure) and the stricter
    case-RESPECTING case-level sign-test p (one value per case, honouring
    cluster_unit = case; the conservative robustness companion). Both p-value
    families are Holm-corrected over the 12 tests and survivors reported.
    """
    tests: dict[str, dict] = {}
    case_p: dict[str, float] = {}  # case-level sign p (stricter, case-respecting)
    enc_p: dict[str, float] = {}  # encounter-level sign p (D165 pre-registered, headline)
    missing: list[str] = []
    for endpoint in ("representational", "forecast"):
        for obs in OBSERVABLES:
            tkey = f"{ENDPOINT_TAG[endpoint]}/{SHORT[obs]}"
            pk = dump.key(
                PREDICTIVE_LEAD, obs, endpoint, PRIMARY_PROBE, "test_b", "pooled", PRIMARY_H
            )
            rk = dump.key(
                RECON_FUKAMI_LEAD, obs, endpoint, PRIMARY_PROBE, "test_b", "pooled", PRIMARY_H
            )
            if not (dump.has(pk) and dump.has(rk)):
                missing.append(tkey)
                continue
            block = paired_difference_block(dump.get(pk), dump.get(rk), cfg.boot_seed)
            tests[tkey] = block
            case_p[tkey] = block["case_level_sign_p"]
            enc_p[tkey] = block["sign_p_one_sided"]

    holm_out: dict = {
        "predictive": PREDICTIVE_LEAD,
        "reconstructive": RECON_FUKAMI_LEAD,
        "cell": "test_b pooled, H=16, d=64, ridge",
        "holm_headline_p": "sign_p_one_sided (encounter-level, the D165 pre-registered input)",
        "holm_robustness_p": "case_level_sign_p (stricter, case-respecting, cluster_unit=case)",
        "n_tests": len(tests),
        "missing": missing,
        "tests": tests,
    }
    if tests:
        adj_case, surv_case = stats_lib.holm(case_p)
        adj_enc, surv_enc = stats_lib.holm(enc_p)
        for tkey in tests:
            tests[tkey]["holm_case_p"] = adj_case[tkey]
            tests[tkey]["holm_case_survives"] = surv_case[tkey]
            tests[tkey]["holm_enc_p"] = adj_enc[tkey]
            tests[tkey]["holm_enc_survives"] = surv_enc[tkey]
        holm_out["survivors_case"] = sorted(k for k, v in surv_case.items() if v)
        holm_out["survivors_enc"] = sorted(k for k, v in surv_enc.items() if v)
        holm_out["n_survivors_case"] = len(holm_out["survivors_case"])
        holm_out["n_survivors_enc"] = len(holm_out["survivors_enc"])
    return holm_out


def harvest_family_verdicts(
    dump: PerEncounterDump, m: dict[str, np.ndarray], cfg: argparse.Namespace
) -> dict:
    """(b) Per-family verdicts on the primary cell: predictive vs each baseline family.

    Seed-aggregated point + sd come from the matrix summary (every member's R^2);
    the paired separation test uses the LEAD member of each family against the
    predictive lead, on the per-encounter dump, with a case-clustered CI.
    """
    # seed-aggregated R^2 per family at the primary cell (from the matrix summary)
    base = (
        matrix_select(
            m,
            observable=PRIMARY_OBS,
            endpoint="representational",
            probe=PRIMARY_PROBE,
            split="test_b",
            tier="pooled",
        )
        & (m["H"] == PRIMARY_H)
        & (m["d"] == 64)
    )
    fam_summary: dict[str, dict] = {}
    for fam in sorted(set(m["family"][base].tolist())):
        vals = m["value"][base & (m["family"] == fam)].astype(np.float64)
        seeds = m["seed"][base & (m["family"] == fam)].astype(np.int64)
        obj = m["objective"][base & (m["family"] == fam)][0]
        fam_summary[fam] = {
            "objective": str(obj),
            "seed_mean": float(vals.mean()),
            "seed_sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "n_seeds": int(vals.size),
            "seeds": seeds.tolist(),
        }

    # lead-member tag per family (seed 42 if present, else lowest seed, pod is seedless)
    lead_tag: dict[str, str] = {}
    for fam in fam_summary:
        rows = base & (m["family"] == fam)
        tags = [str(t) for t in m["tag"][rows]]
        seeds = [int(s) for s in m["seed"][rows]]
        # prefer seed 42, then the lowest seed; stable on ties
        order = sorted(range(len(seeds)), key=lambda i: (seeds[i] != 42, seeds[i]))
        lead_tag[fam] = tags[order[0]]

    pred_key = dump.key(
        PREDICTIVE_LEAD,
        PRIMARY_OBS,
        "representational",
        PRIMARY_PROBE,
        "test_b",
        "pooled",
        PRIMARY_H,
    )
    pred_cell = dump.get(pred_key) if dump.has(pred_key) else None

    verdicts: dict[str, dict] = {}
    for fam, summ in fam_summary.items():
        # One verdict per comparator family: predictive lead vs every OTHER family
        # (reconstructive baselines and the other predictive families, e.g.
        # jepa_lstm_noc / ctrl_pred_cnn). Skip the lead vs itself (delta == 0).
        if fam == LEAD_FAMILY:
            continue
        rk = dump.key(
            lead_tag[fam],
            PRIMARY_OBS,
            "representational",
            PRIMARY_PROBE,
            "test_b",
            "pooled",
            PRIMARY_H,
        )
        rec = {
            "lead_tag": lead_tag[fam],
            "objective": summ["objective"],
            "predictive_seed_mean": fam_summary[LEAD_FAMILY]["seed_mean"],
            "baseline_seed_mean": summ["seed_mean"],
            "baseline_seed_sd": summ["seed_sd"],
        }
        if pred_cell is not None and dump.has(rk):
            err_pred, err_base, case = match_cells(pred_cell, dump.get(rk))
            delta = err_base - err_pred  # >0 = predictive better
            cc = stats_lib.case_cluster_bootstrap(
                delta, case, n_boot=stats_lib.N_BOOT_CASE, rng=np.random.default_rng(cfg.boot_seed)
            )
            rec["enc_mean_delta"] = cc["enc_mean"]
            rec["case_clustered_ci"] = cc["enc_mean_ci"]
            rec["case_mean_ci"] = cc["case_mean_ci"]
            rec["significant_clustered"] = bool(
                cc["enc_mean_ci"][0] > 0 or cc["enc_mean_ci"][1] < 0
            )
            rec["predictive_better"] = bool(cc["enc_mean"] > 0)
        else:
            rec["note"] = "no matched per-encounter dump cell for this family lead"
        verdicts[fam] = rec
    return {
        "predictive": LEAD_FAMILY,
        "cell": "representational wake_enstrophy R^2, H=16, test_b pooled, d=64, ridge",
        "family_summary": fam_summary,
        "verdicts": verdicts,
    }


def harvest_gate_gd_hardening(dump: PerEncounterDump, cfg: argparse.Namespace) -> dict:
    """(c) Is the Gate-GD WEAK branch statistically supported, or a wide-CI artifact?

    The B2 gate compared SEED-MEAN point values and fired weak because a nonlinear
    probe brought a reconstructive family within 0.15 of the predictive FLOOR. Here
    we test the paired predictive-minus-reconstructive wake difference for the
    matched-head reconstructive AE (ctrl_recon_cnn) under each nonlinear probe, with
    a case-clustered CI: if the CI excludes a difference of 0.15 (i.e. the predictive
    family is reliably MORE than 0.15 better), the weak trigger is a point-estimate
    artifact; if the CI includes small/zero differences, the weak branch is supported.
    """
    out: dict[str, dict] = {}
    for probe in NONLINEAR_PROBES:
        pk = dump.key(
            PREDICTIVE_LEAD, PRIMARY_OBS, "representational", probe, "test_b", "pooled", PRIMARY_H
        )
        # ctrl_recon_cnn is seedless-aligned by tag ctrl_recon_cnn_s{0,1,2}; the gate
        # used the seed MEAN, so we test each seed's lead and report the union, plus
        # the s0 lead as the canonical paired cell.
        recon_tags = [f"{RECON_MATCHED_HEAD}_s{s}" for s in (0, 1, 2)]
        per_seed: dict[str, dict] = {}
        if not dump.has(pk):
            out[probe] = {"note": f"no predictive dump cell {pk}"}
            continue
        pred_cell = dump.get(pk)
        for rtag in recon_tags:
            rk = dump.key(
                rtag, PRIMARY_OBS, "representational", probe, "test_b", "pooled", PRIMARY_H
            )
            if not dump.has(rk):
                continue
            err_pred, err_base, case = match_cells(pred_cell, dump.get(rk))
            delta = err_base - err_pred  # >0 = predictive better (lower error)
            cc = stats_lib.case_cluster_bootstrap(
                delta, case, n_boot=stats_lib.N_BOOT_CASE, rng=np.random.default_rng(cfg.boot_seed)
            )
            per_seed[rtag] = {
                "enc_mean_delta": cc["enc_mean"],
                "case_clustered_ci": cc["enc_mean_ci"],
                "case_mean_ci": cc["case_mean_ci"],
                # the gate's WEAK test is on R^2 within 0.15; here the paired metric
                # is the absolute-error difference. A clustered CI that includes 0
                # (the predictive AE is NOT reliably better than the matched-head
                # reconstructive AE on wake) directly SUPPORTS the weak branch.
                "ci_includes_zero": bool(cc["enc_mean_ci"][0] <= 0 <= cc["enc_mean_ci"][1]),
                "predictive_better_clustered": bool(cc["enc_mean_ci"][0] > 0),
            }
        out[probe] = {"per_seed": per_seed}
    # Verdict: weak branch is SUPPORTED if, under any nonlinear probe, the matched-head
    # reconstructive AE is not reliably worse than predictive (clustered CI includes 0)
    # for the seed-majority of comparisons.
    weak_supported = False
    rationale = []
    for probe, blk in out.items():
        ps = blk.get("per_seed", {})
        if not ps:
            continue
        n_incl = sum(1 for v in ps.values() if v["ci_includes_zero"])
        if n_incl >= max(1, (len(ps) + 1) // 2):
            weak_supported = True
            rationale.append(
                f"{probe}: {n_incl}/{len(ps)} matched-head seeds have a clustered CI "
                "that includes zero (predictive not reliably better)"
            )
        else:
            rationale.append(
                f"{probe}: {len(ps) - n_incl}/{len(ps)} matched-head seeds have a "
                "clustered CI excluding zero (predictive reliably better)"
            )
    return {
        "predictive": PREDICTIVE_LEAD,
        "matched_head_reconstructive": RECON_MATCHED_HEAD,
        "cell": "representational wake_enstrophy, H=16, test_b pooled, d=64, nonlinear probes",
        "per_probe": out,
        "weak_branch_supported": weak_supported,
        "rationale": rationale,
        "interpretation": (
            "Weak branch SUPPORTED: the matched-head reconstructive AE is not reliably "
            "separated from the predictive family on linear/nonlinear wake decodability "
            "after case clustering; D183 lands on WEAK (linear-decodability framing)."
            if weak_supported
            else "Weak branch NOT statistically supported by the matched-head comparison: "
            "the predictive family separates from the matched-head reconstructive AE "
            "under case clustering; the point-estimate weak trigger should be reviewed."
        ),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho via scipy (statistic only; used as the case-permutation stat_fn)."""
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def harvest_fig8_trends(dump: PerEncounterDump, cfg: argparse.Namespace) -> dict:
    """(d) Spearman of the per-encounter wake closure vs |G| and D, lead family.

    The per-encounter "closure" proxy is the NEGATIVE absolute error -err: higher
    means the latent tracks the wake better on that encounter (so a positive
    Spearman vs |G| means the wake is BETTER tracked at larger gust amplitude).
    test_b and test_c encounters are pooled (the trend is a physics regularity, not
    a held-out claim). case-permutation p via stats_lib.
    """
    cells = []
    for split in ("test_b", "test_c"):
        tier = "pooled" if split == "test_b" else "all"
        k = dump.key(
            f"{LEAD_FAMILY}_d64_s42",
            PRIMARY_OBS,
            "representational",
            PRIMARY_PROBE,
            split,
            tier,
            PRIMARY_H,
        )
        if dump.has(k):
            cells.append(dump.get(k))
    if not cells:
        return {"note": "no lead-family wake dump cells for test_b/test_c"}

    case = np.concatenate([c.case_id for c in cells])
    enc = np.concatenate([abs_error(c) for c in cells])
    closure = -enc  # higher = better-tracked encounter

    # |G| and D per encounter from the case_id label (G<sign><mag>_D<..>_Y<..>)
    def parse_gd(cid: str) -> tuple[float, float]:
        g = float(cid.split("_D")[0].replace("G", ""))
        d = float(cid.split("_D")[1].split("_Y")[0])
        return abs(g), d

    absG = np.array([parse_gd(str(c))[0] for c in case])
    Dval = np.array([parse_gd(str(c))[1] for c in case])

    out: dict[str, dict] = {}
    for name, xvar in (("absG", absG), ("D", Dval)):
        perm = stats_lib.case_permutation_p(
            xvar,
            closure,
            case,
            spearman,
            n_perm=cfg.n_perm,
            rng=np.random.default_rng(cfg.perm_seed),
            alternative="two-sided",
        )
        out[name] = {
            "spearman_rho": perm["stat"],
            "perm_p": float(perm["p"]),
            "perm_n": int(perm["n_perm"]),
            "n_encounters": int(closure.size),
            "n_cases": int(len(set(case.tolist()))),
        }
    return {
        "family": LEAD_FAMILY,
        "metric": "per-encounter closure = -|y_pred - y_true| (representational wake, H=16)",
        "splits": "test_b + test_c pooled",
        "trends": out,
    }


# ------------------------------------------------------------------- numbers part (e)


def fmt_ci(ci: list[float]) -> str:
    """[lo, hi] -> '[lo, hi]' with 3 decimals for the report."""
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def build_numbers_part(holm: dict, verdicts: dict, gd: dict, fig8: dict) -> dict:
    """Assemble the eval_all.py numbers PART (macro-bound headline quantities).

    Macro-bound:
        NumPairedWakeRepr     primary paired difference (predictive minus Fukami,
                              representational wake, abs-error delta) + clustered CI;
        NumHolmSurvivors      count of the 12-test Holm family that survive the
                              pre-registered D165 procedure (encounter-level sign p).
    The stricter case-respecting survivor count and the GD-hardening differences are
    stored as non-macro records.
    """
    numbers: dict[str, dict] = {}

    prim = holm.get("tests", {}).get("repr/wake_enstrophy")
    if prim is not None:
        numbers["paired_repr_wake_pred_minus_fukami"] = {
            "macro": "NumPairedWakeRepr",
            "value": float(prim["enc_mean_delta"]),
            "fmt": "%.2f",
            "ci_lo": float(prim["case_clustered_ci"][0]),
            "ci_hi": float(prim["case_clustered_ci"][1]),
            "n": int(prim["n_encounters"]),
            "split": "test_b",
            "endpoint": "representational",
            "probe": PRIMARY_PROBE,
            "observable": PRIMARY_OBS,
            "horizon": PRIMARY_H,
            "run_tags": [PREDICTIVE_LEAD, RECON_FUKAMI_LEAD],
            "source": "stats_harvest.py",
            "note": (
                "paired abs-error delta = err(fukami) - err(jepa_tf_noc); >0 = predictive "
                f"better; case-clustered CI; enc_sign_p={prim['sign_p_one_sided']:.4g} "
                f"(holm_enc_survives={prim.get('holm_enc_survives')}), "
                f"case_sign_p={prim['case_level_sign_p']:.4g} "
                f"(holm_case_survives={prim.get('holm_case_survives')})"
            ),
        }

    if "n_survivors_enc" in holm:
        numbers["holm_family_survivors"] = {
            "macro": "NumHolmSurvivors",
            "value": int(holm["n_survivors_enc"]),
            "fmt": "%d",
            "n": int(holm["n_tests"]),
            "source": "stats_harvest.py",
            "note": (
                "Holm survivors of the 12-test family (encounter-level sign p, the "
                "pre-registered D165 procedure): "
                + (", ".join(holm["survivors_enc"]) or "(none)")
                + f"; stricter case-respecting (case-level sign p) survivors "
                f"[{holm['n_survivors_case']}/12]: "
                + (", ".join(holm["survivors_case"]) or "(none)")
            ),
        }
        numbers["holm_family_survivors_case_strict"] = {
            "value": int(holm["n_survivors_case"]),
            "n": int(holm["n_tests"]),
            "source": "stats_harvest.py",
            "note": (
                "stricter case-respecting (case-level one-sided sign p) Holm survivors: "
                + (", ".join(holm["survivors_case"]) or "(none)")
            ),
        }

    # GD-hardening: store the matched-head paired difference per nonlinear probe (s0 lead)
    for probe, blk in gd.get("per_probe", {}).items():
        ps = blk.get("per_seed", {})
        s0 = ps.get(f"{RECON_MATCHED_HEAD}_s0")
        if s0 is None:
            continue
        numbers[f"gd_hardening_{probe}_s0"] = {
            "value": float(s0["enc_mean_delta"]),
            "ci_lo": float(s0["case_clustered_ci"][0]),
            "ci_hi": float(s0["case_clustered_ci"][1]),
            "split": "test_b",
            "endpoint": "representational",
            "probe": probe,
            "observable": PRIMARY_OBS,
            "horizon": PRIMARY_H,
            "run_tags": [PREDICTIVE_LEAD, f"{RECON_MATCHED_HEAD}_s0"],
            "source": "stats_harvest.py",
            "note": (
                "paired abs-error delta = err(ctrl_recon_cnn_s0) - err(jepa_tf_noc); "
                f"ci_includes_zero={s0['ci_includes_zero']}"
            ),
        }

    numbers["gd_weak_branch_supported"] = {
        "value": 1 if gd.get("weak_branch_supported") else 0,
        "fmt": "%d",
        "source": "stats_harvest.py",
        "note": gd.get("interpretation", ""),
    }

    return {"part": "stats_harvest", "numbers": numbers}


def write_report(holm: dict, verdicts: dict, gd: dict, fig8: dict, out_path: Path) -> None:
    """Human-readable B6 report (no em-dashes; tables as plain text)."""
    L: list[str] = []
    L.append("# Session 28 B6 core statistics harvest\n")
    L.append(
        "All paired tests use the D165 convention: per-encounter absolute error, "
        "delta = err(reconstructive) - err(predictive), so delta > 0 means the "
        "predictive family is better. CIs are case-clustered (cluster unit = case, "
        "n = 10000). The HEADLINE Holm survivor count uses the encounter-level "
        "one-sided sign-test p (the exact pre-registered D165 holm.json procedure); "
        "the stricter case-RESPECTING case-level one-sided sign-test p (one value per "
        "case, honouring cluster_unit = case) is reported and Holm-corrected alongside "
        "as a conservative robustness companion.\n"
    )

    L.append(
        "## (a) The 12-test Holm family (predictive jepa_tf_noc vs reconstructive Fukami AE)\n"
    )
    L.append(f"Cell: {holm.get('cell')}; n_tests = {holm.get('n_tests')}.")
    if holm.get("missing"):
        L.append(f"Missing cells (no dump): {', '.join(holm['missing'])}.")
    L.append("")
    L.append(
        "  test                       n_enc  delta      clustered_CI            "
        "enc_p    holm_enc  surv_enc  case_p   holm_case surv_case"
    )
    for tkey in sorted(holm.get("tests", {})):
        t = holm["tests"][tkey]
        L.append(
            f"  {tkey:26s} {t['n_encounters']:4d}  {t['enc_mean_delta']:+8.3f}  "
            f"{fmt_ci(t['case_clustered_ci']):>22s}  {t['sign_p_one_sided']:.4g}"
            f"{'':2s} {t.get('holm_enc_p', float('nan')):.4g}"
            f"   {str(t.get('holm_enc_survives')):5s}  {t['case_level_sign_p']:.4g}"
            f"{'':2s} {t.get('holm_case_p', float('nan')):.4g}"
            f"   {str(t.get('holm_case_survives')):5s}"
        )
    L.append("")
    if "survivors_enc" in holm:
        L.append(
            f"Holm survivors (encounter-level sign p, HEADLINE / D165 procedure, "
            f"alpha=0.05): {holm['n_survivors_enc']}/12 -> "
            + (", ".join(holm["survivors_enc"]) or "(none)")
        )
        L.append(
            f"Holm survivors (case-level sign p, STRICTER case-respecting robustness): "
            f"{holm['n_survivors_case']}/12 -> " + (", ".join(holm["survivors_case"]) or "(none)")
        )
    L.append("")

    L.append("## (b) Per-family verdicts on the primary cell (predictive vs each baseline)\n")
    L.append(f"Cell: {verdicts.get('cell')}.")
    L.append("")
    L.append(
        "  family                 obj             seed_mean  sd     paired_delta  "
        "clustered_CI            sig?"
    )
    fs = verdicts.get("family_summary", {})
    for fam in sorted(verdicts.get("verdicts", {})):
        v = verdicts["verdicts"][fam]
        sm = fs.get(fam, {})
        delta = v.get("enc_mean_delta")
        ci = v.get("case_clustered_ci")
        L.append(
            f"  {fam:22s} {str(v.get('objective')):14s} "
            f"{sm.get('seed_mean', float('nan')):+8.3f}  {sm.get('seed_sd', 0.0):.3f}  "
            f"{(f'{delta:+8.3f}' if delta is not None else '     n/a'):>10s}  "
            f"{(fmt_ci(ci) if ci else '          n/a'):>22s}  "
            f"{v.get('significant_clustered')}"
        )
    L.append("")
    L.append(
        f"Predictive lead ({verdicts.get('predictive')}) seed_mean = "
        f"{fs.get(LEAD_FAMILY, {}).get('seed_mean', float('nan')):+.3f}.\n"
    )

    L.append("## (c) Gate-GD weak-branch hardening (matched-head reconstructive AE)\n")
    L.append(f"Cell: {gd.get('cell')}.")
    L.append(
        f"Predictive = {gd.get('predictive')}; "
        f"matched-head = {gd.get('matched_head_reconstructive')}."
    )
    L.append("")
    for probe, blk in gd.get("per_probe", {}).items():
        L.append(f"  probe {probe}:")
        for rtag, v in blk.get("per_seed", {}).items():
            L.append(
                f"    {rtag:20s} delta={v['enc_mean_delta']:+8.3f}  "
                f"CI={fmt_ci(v['case_clustered_ci'])}  ci_includes_zero={v['ci_includes_zero']}"
            )
        if "note" in blk:
            L.append(f"    {blk['note']}")
    L.append("")
    L.append(f"Weak branch statistically supported: {gd.get('weak_branch_supported')}")
    for r in gd.get("rationale", []):
        L.append(f"  - {r}")
    L.append(f"\n{gd.get('interpretation')}\n")

    L.append("## (d) Fig-8 trends: per-encounter wake closure vs |G| and D (lead family)\n")
    if "trends" in fig8:
        L.append(f"Family {fig8['family']}; metric: {fig8['metric']}; {fig8['splits']}.")
        L.append("")
        for name, t in fig8["trends"].items():
            L.append(
                f"  vs {name:5s}: Spearman rho = {t['spearman_rho']:+.3f}, "
                f"perm_p = {t['perm_p']:.4g} (n_enc={t['n_encounters']}, n_cases={t['n_cases']})"
            )
    else:
        L.append(fig8.get("note", "(no trend cells)"))
    L.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L))
    print(f"[harvest] wrote report -> {out_path}")


# ------------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI; defaults read the canonical closure-matrix outputs under the repo root."""
    p = argparse.ArgumentParser(description="Session 28 B6 statistics harvest.")
    p.add_argument(
        "--closure-dir",
        type=Path,
        default=REPO / "outputs/session28/closure_matrix",
        help="Directory holding matrix.npz and per_encounter/dump.npz.",
    )
    p.add_argument(
        "--numbers-part",
        type=Path,
        default=REPO / "outputs/session28/numbers_parts/stats_harvest.json",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=REPO / "outputs/session28/stats_harvest_report.md",
    )
    p.add_argument("--boot-seed", type=int, default=0, help="Case-clustered bootstrap seed.")
    p.add_argument("--perm-seed", type=int, default=0, help="Case-permutation seed.")
    p.add_argument("--n-perm", type=int, default=10000, help="Case-permutation draws.")
    p.add_argument("--check", action="store_true", help="Compute + print; do not write.")
    return p.parse_args(argv)


def run_harvest(cfg: argparse.Namespace) -> dict:
    """Compute all five harvest blocks; return the assembled dict (no writing)."""
    closure_dir = cfg.closure_dir
    dump_npz = closure_dir / "per_encounter" / "dump.npz"
    dump_index = closure_dir / "per_encounter" / "index.json"
    matrix_npz = closure_dir / "matrix.npz"
    if not dump_npz.exists():
        raise SystemExit(
            f"per-encounter dump not found at {dump_npz}; run closure_matrix.py "
            "--dump-per-encounter first (see module docstring)."
        )
    dump = PerEncounterDump(dump_npz, dump_index)
    m = load_matrix(matrix_npz)

    holm = harvest_holm_family(dump, cfg)
    verdicts = harvest_family_verdicts(dump, m, cfg)
    gd = harvest_gate_gd_hardening(dump, cfg)
    fig8 = harvest_fig8_trends(dump, cfg)
    return {"holm": holm, "verdicts": verdicts, "gd": gd, "fig8": fig8}


def main(argv: list[str] | None = None) -> None:
    """Run the harvest; write the numbers part + report (unless --check)."""
    cfg = parse_args(argv)
    res = run_harvest(cfg)
    holm, verdicts, gd, fig8 = res["holm"], res["verdicts"], res["gd"], res["fig8"]

    print(
        f"[harvest] Holm family: {holm.get('n_tests')} tests, "
        f"{holm.get('n_survivors_enc', 0)} survive (encounter-level sign p, D165 headline)"
    )
    print(f"[harvest] survivors (headline, enc-level): {holm.get('survivors_enc')}")
    print(f"[harvest] survivors (stricter, case-level): {holm.get('survivors_case')}")
    print(f"[harvest] Gate-GD weak branch supported: {gd.get('weak_branch_supported')}")
    prim = holm.get("tests", {}).get("repr/wake_enstrophy")
    if prim:
        print(
            "[harvest] primary paired delta (pred - fukami, repr wake): "
            f"{prim['enc_mean_delta']:+.3f} CI {fmt_ci(prim['case_clustered_ci'])} "
            f"(holm_enc_survives={prim.get('holm_enc_survives')}, "
            f"holm_case_survives={prim.get('holm_case_survives')})"
        )

    part = build_numbers_part(holm, verdicts, gd, fig8)
    if cfg.check:
        print("[harvest] check mode: not writing")
        print(json.dumps(part, indent=2))
        return
    cfg.numbers_part.parent.mkdir(parents=True, exist_ok=True)
    cfg.numbers_part.write_text(json.dumps(part, indent=2))
    print(f"[harvest] wrote numbers part -> {cfg.numbers_part}")
    # Also persist the full block dict next to the report for downstream figures.
    (cfg.report.parent / "stats_harvest_full.json").write_text(
        json.dumps(res, indent=2, default=str)
    )
    write_report(holm, verdicts, gd, fig8, cfg.report)


if __name__ == "__main__":
    main()

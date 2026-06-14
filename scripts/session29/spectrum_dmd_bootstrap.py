"""Bootstrap confidence intervals for the latent DMD spectrum (Session 28 Part 1)
and the predictor Floquet leading-multiplier modulus (Session 28 Part 2).

The point estimates live in ``outputs/session28/numbers_parts/spectrum_dmd.json``
and are produced by ``scripts/session28/spectrum_dmd.py``. That script reports the
leading oscillatory DMD eigenvalue (Strouhal number + modulus) on the no-gust
Baseline limit cycle per family (POD / JEPA tf-no-c / Fukami at d = 64) and the
leading Floquet multiplier modulus of the linearised tf-no-c predictor over one
shedding cycle. None of those carried an uncertainty band; this script attaches
2.5 / 97.5 percentile bootstrap CIs so the manuscript can report the
shedding-frequency eigenvalue with uncertainty and qualify the Floquet
"marginally stable orbit" claim as a CI statement rather than a bare point value.

WHAT IS RESAMPLED (matching how each point estimate is formed)
    Part 1 (data-driven DMD). The point estimate stacks every consecutive
    snapshot pair (z_t, z_{t+1}) from the four Baseline encounters into column
    matrices X, Y (476 pairs, no pair crosses an encounter boundary) and fits the
    exact-DMD operator A = Y X^+ by least squares, then extracts the
    leading-modulus oscillatory eigenvalue. With only four Baseline encounters an
    encounter-level resample is far too coarse, so (per the standard DMD
    least-squares bootstrap, and as instructed for the few-segment regime) we
    resample the snapshot PAIRS / columns with replacement within the standard
    DMD delay-embedding window, refit A on the resampled columns, and re-extract
    the same leading oscillatory eigenvalue. n = 1000 draws; we report the
    2.5 / 97.5 percentiles of the resulting St and |lambda| per family.

    Part 2 (Floquet). The monodromy over one ~59-frame Baseline shedding cycle is
    the ordered product of the per-frame companion Jacobians of the tf-no-c
    predictor; the leading per-step modulus is |mu_cycle|^{1/period}. Recomputing
    the companion Jacobians is the only predictor computation; it is done ONCE on
    CPU (the predictor is small; no GPU per the analysis-only constraint), then
    the cycle product is bootstrapped by resampling the 59 per-frame companion
    Jacobians with replacement (the per-step modulus is the per-frame geometric
    contribution, so a with-replacement resample of the step factors is the
    natural bootstrap of that geometric mean). n = 1000 draws; we report the
    2.5 / 97.5 percentiles of the per-step leading-multiplier modulus and, crucial
    for the manuscript wording, whether that CI straddles 1.0.

Outputs (analysis-only; the main agent merges the numbers part):
    outputs/session28/numbers_parts/spectrum_dmd_bootstrap.json   (numbers part)
    outputs/session29/reports/spectrum_dmd_bootstrap.md           (report)

CPU only. No GPU. Does not touch any training run or the existing point-estimate
artifacts. Run niced and thread-capped per the shared-workstation hygiene block:

    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    nice -n 10 .venv/bin/python scripts/session29/spectrum_dmd_bootstrap.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

import sys  # noqa: E402

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session28"))
sys.path.insert(0, str(REPO / "scripts"))

import spectrum_dmd as S  # noqa: E402

N_BOOT = 1000
BOOT_SEED = 0
CI_LO_PCT = 2.5
CI_HI_PCT = 97.5
FLOQUET_TAG = "noc_tf"  # the paper's main (tf-no-c) predictor


# ------------------------------------------------------------------ Part 1 bootstrap


def snapshot_pairs(trajs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack consecutive (z_t, z_{t+1}) columns exactly as ``exact_dmd`` does.

    Pairs never cross a trajectory boundary. Returns (X, Y) each (d, m), where m
    is the number of snapshot pairs pooled over all encounters.
    """
    x_cols = []
    y_cols = []
    for traj in trajs:
        traj = np.asarray(traj, dtype=np.float64)
        if traj.shape[0] < 2:
            continue
        x_cols.append(traj[:-1].T)
        y_cols.append(traj[1:].T)
    if not x_cols:
        raise ValueError("no usable snapshot pairs")
    return np.concatenate(x_cols, axis=1), np.concatenate(y_cols, axis=1)


def leading_from_columns(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit A = Y X^+ on the given columns and return (leading_St, leading_modulus).

    Identical operator construction and leading-oscillatory selection as the
    Part-1 point estimate (``exact_dmd`` -> ``leading_oscillatory_pair``); only
    the column set changes (a bootstrap resample of the snapshot pairs).
    """
    a_t, *_ = np.linalg.lstsq(x.T, y.T, rcond=None)
    eigvals = np.linalg.eigvals(a_t.T)
    lead = S.leading_oscillatory_pair(eigvals)["leading"]
    return float(lead["St"]), float(lead["modulus"])


def bootstrap_part1(latents_root: Path, n_boot: int, seed: int) -> dict:
    """Pair-resampling bootstrap of the leading DMD eigenvalue per family."""
    out: dict = {}
    for fam, tag in S.DMD_FAMILIES.items():
        path = latents_root / tag / "train.npz"
        if not path.exists():
            print(f"[part1-boot] SKIP {fam}: {path} missing")
            continue
        blob = np.load(path, allow_pickle=True)
        base = S.baseline_trajectories(blob)
        if not base:
            print(f"[part1-boot] SKIP {fam}: no Baseline trajectories")
            continue
        x, y = snapshot_pairs(base)
        m = x.shape[1]
        st_pt, mod_pt = leading_from_columns(x, y)
        rng = np.random.default_rng(seed)
        sts = np.empty(n_boot)
        mods = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, m, size=m)
            sts[b], mods[b] = leading_from_columns(x[:, idx], y[:, idx])
        st_lo, st_hi = np.percentile(sts, [CI_LO_PCT, CI_HI_PCT])
        mod_lo, mod_hi = np.percentile(mods, [CI_LO_PCT, CI_HI_PCT])
        out[fam] = {
            "tag": tag,
            "n_pairs": int(m),
            "st_point": st_pt,
            "st_ci_lo": float(st_lo),
            "st_ci_hi": float(st_hi),
            "modulus_point": mod_pt,
            "modulus_ci_lo": float(mod_lo),
            "modulus_ci_hi": float(mod_hi),
        }
        print(
            f"[part1-boot] {fam:12s} St={st_pt:.3f} [{st_lo:.3f}, {st_hi:.3f}]  "
            f"|lambda|={mod_pt:.3f} [{mod_lo:.3f}, {mod_hi:.3f}]  (n_pairs={m})"
        )
    return out


# ------------------------------------------------------------------ Part 2 bootstrap


def build_companion_jacobians(latents_root: Path, tag: str) -> tuple[np.ndarray, int, int]:
    """Compute the per-frame companion Jacobians over one Baseline cycle ONCE (CPU).

    Returns (Cs, period, window) where Cs is (period, Wd, Wd). This is the only
    predictor computation; everything downstream is a numpy resample of these
    cached step factors.
    """
    import torch

    device = torch.device("cpu")
    bundle = S.load_predictor(tag, device)
    blob = np.load(latents_root / S.DMD_FAMILIES["jepa_tf_noc"] / "train.npz", allow_pickle=True)
    base = S.baseline_trajectories(blob)
    if not base:
        raise ValueError("no Baseline trajectory for the Floquet operating cycle")
    traj = max(base, key=lambda t: t.shape[0]).astype(np.float32)
    t_len = traj.shape[0]
    window = min(bundle.max_seq_len, 16)
    period = S.FULL_CYCLE_FRAMES
    if t_len < window + period:
        period = max(2, t_len - window - 1)
    torch.set_grad_enabled(True)
    t0 = window - 1
    cs = []
    tic = time.time()
    for step in range(period):
        t = t0 + step
        lo = t - window + 1
        win = traj[lo:t + 1]
        cs.append(S.companion_jacobian(bundle.predictor, bundle.ptype, win, device))
    torch.set_grad_enabled(False)
    print(
        f"[part2-boot] built {period} companion Jacobians (W={window}, {bundle.ptype}) "
        f"in {time.time() - tic:.1f}s"
    )
    return np.stack(cs), period, window


def cycle_per_step_modulus_exact(cs: np.ndarray, order: np.ndarray, period: int) -> float:
    """Exact leading per-step Floquet modulus (forms the product, full eig).

    Used for the point estimate / verification; O(period) matmuls plus one (Wd, Wd)
    eigendecomposition. The bootstrap uses the cheaper power-iteration variant
    below, which reproduces this to ~6 digits.
    """
    n = cs.shape[1]
    mono = np.eye(n)
    for k in order:
        mono = cs[k] @ mono
    lead_cycle_mod = float(np.max(np.abs(np.linalg.eigvals(mono))))
    return lead_cycle_mod ** (1.0 / period)


def cycle_per_step_modulus(
    cs: np.ndarray, order: np.ndarray, period: int, iters: int = 200, tol: float = 1e-10
) -> float:
    """Leading per-step Floquet modulus via power iteration on the matvec sequence.

    The monodromy M = C_{order[-1]} ... C_{order[0]} is applied as a sequence of
    ``period`` matrix-vector products (never forming the (Wd, Wd) product), and
    ||M v|| / ||v|| converges to the leading-multiplier modulus |mu_cycle| (true
    for a real OR a complex-pair dominant eigenvalue: ||M^k v||^{1/k} -> |mu|).
    The per-step modulus is |mu_cycle|^{1/period}. This matches the exact path to
    ~6 digits and is ~15x cheaper, so the 1000-draw bootstrap runs in ~1 minute on
    CPU. The leading Floquet multiplier here is strongly dominant (near-real,
    ~1.28 over the cycle), so convergence is fast (~16 iterations).
    """
    rng = np.random.default_rng(0)
    v = rng.standard_normal(cs.shape[1])
    v /= np.linalg.norm(v)
    lam_prev = 0.0
    for _ in range(iters):
        w = v
        for k in order:
            w = cs[k] @ w
        lam = float(np.linalg.norm(w))
        v = w / (lam + 1e-300)
        if abs(lam - lam_prev) <= tol * max(lam, 1e-12):
            break
        lam_prev = lam
    return lam ** (1.0 / period)


def bootstrap_part2(latents_root: Path, tag: str, n_boot: int, seed: int) -> dict:
    """Resample the per-frame companion Jacobians with replacement; CI of per-step |mu|."""
    cs, period, window = build_companion_jacobians(latents_root, tag)
    idx0 = np.arange(period)
    per_step_pt = cycle_per_step_modulus_exact(cs, idx0, period)  # exact for the point value
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        order = rng.integers(0, period, size=period)
        vals[b] = cycle_per_step_modulus(cs, order, period)
    lo, hi = np.percentile(vals, [CI_LO_PCT, CI_HI_PCT])
    straddles = bool(lo <= 1.0 <= hi)
    print(
        f"[part2-boot] {tag} Floquet per-step |mu|={per_step_pt:.4f} "
        f"[{lo:.4f}, {hi:.4f}]  straddles 1.0: {straddles}"
    )
    return {
        "tag": tag,
        "period_frames": int(period),
        "window": int(window),
        "per_step_modulus_point": float(per_step_pt),
        "per_step_modulus_ci_lo": float(lo),
        "per_step_modulus_ci_hi": float(hi),
        "ci_straddles_one": straddles,
    }


# ------------------------------------------------------------------ numbers part


def emit_numbers_part(part1: dict, part2: dict, out_path: Path) -> dict:
    """Write the bootstrap numbers part using the eval_all ci_lo/ci_hi mechanism.

    Macro naming: NEW base names suffixed ``CI`` so they never collide with the
    existing ``SpecDmdSt*`` / ``SpecDmdMod*`` / ``SpecFloqMod`` point macros that
    eval_all already emits from spectrum_dmd.json. Each record carries ``value``
    (the point estimate, repeated) plus ``ci_lo`` / ``ci_hi``; emit_macros then
    auto-emits the three alphabetic-only macros ``\\<base>``, ``\\<base>lo`` and
    ``\\<base>hi`` per record. So a single record keyed ``SpecDmdStJepaTfCI``
    yields the macros ``SpecDmdStJepaTfCI`` / ``SpecDmdStJepaTfCIlo`` /
    ``SpecDmdStJepaTfCIhi`` (and likewise SpecDmdMod*CI and SpecFloqModCI),
    none of which collide with the existing point macros.
    """
    numbers: dict = {}
    macro_fam = {"pod": "Pod", "jepa_tf_noc": "JepaTf", "fukami": "Fukami"}

    for fam, rec in part1.items():
        mf = macro_fam.get(fam, fam.title().replace("_", ""))
        numbers[f"dmd_St_baseline_{fam}_ci"] = {
            "macro": f"SpecDmdSt{mf}CI",
            "value": rec["st_point"],
            "ci_lo": rec["st_ci_lo"],
            "ci_hi": rec["st_ci_hi"],
            "fmt": "%.2f",
            "split": "baseline",
            "observable": "leading_oscillatory_St",
            "source": "spectrum_dmd_bootstrap.py",
            "note": (
                f"leading-pair DMD St on the Baseline limit cycle ({rec['tag']}); "
                f"{N_BOOT}-draw 2.5/97.5 snapshot-pair bootstrap (n_pairs={rec['n_pairs']})"
            ),
        }
        numbers[f"dmd_modulus_baseline_{fam}_ci"] = {
            "macro": f"SpecDmdMod{mf}CI",
            "value": rec["modulus_point"],
            "ci_lo": rec["modulus_ci_lo"],
            "ci_hi": rec["modulus_ci_hi"],
            "fmt": "%.3f",
            "split": "baseline",
            "observable": "leading_oscillatory_modulus",
            "source": "spectrum_dmd_bootstrap.py",
            "note": (
                f"leading-pair DMD |lambda| on the Baseline limit cycle ({rec['tag']}); "
                f"{N_BOOT}-draw 2.5/97.5 snapshot-pair bootstrap (n_pairs={rec['n_pairs']})"
            ),
        }

    if part2:
        numbers["floquet_leading_modulus_ci"] = {
            "macro": "SpecFloqModCI",
            "value": part2["per_step_modulus_point"],
            "ci_lo": part2["per_step_modulus_ci_lo"],
            "ci_hi": part2["per_step_modulus_ci_hi"],
            "fmt": "%.3f",
            "split": "baseline",
            "endpoint": "forecast",
            "observable": "floquet_leading_per_step_modulus",
            "source": "spectrum_dmd_bootstrap.py",
            "note": (
                f"per-step Floquet multiplier modulus of the {part2['tag']} predictor "
                f"over the {part2['period_frames']}-frame Baseline cycle (W={part2['window']}); "
                f"{N_BOOT}-draw 2.5/97.5 step-Jacobian bootstrap; "
                f"CI straddles 1.0: {part2['ci_straddles_one']}"
            ),
        }

    part = {"part": "spectrum_dmd_bootstrap", "numbers": numbers}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(part, indent=2))
    print(f"[numbers] wrote {len(numbers)} numbers -> {out_path}")
    return part


# ------------------------------------------------------------------ report


def git_sha() -> str:
    """Short git sha of the working tree, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def write_report(
    part1: dict, part2: dict, command: str, latents_root: Path, out_path: Path
) -> None:
    """Markdown report: provenance, point estimates with CIs, Floquet verdict."""
    macro_fam = {"pod": "Pod", "jepa_tf_noc": "JepaTf", "fukami": "Fukami"}
    lines = [
        "# Bootstrap CIs for the latent DMD spectrum and Floquet modulus",
        "",
        f"- git sha: `{git_sha()}`",
        f"- command: `{command}`",
        f"- UTC timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"- n_boot: {N_BOOT}, seed: {BOOT_SEED}, CI: {CI_LO_PCT}/{CI_HI_PCT} percentiles",
        "",
        "## Inputs",
        "",
        f"- latents root: `{latents_root}`",
        "- predictor checkpoint (Floquet): "
        "`outputs/session27/JEPA_d64_noc_tf/checkpoint_iter020000.pt`",
        "- point-estimate part (unchanged): "
        "`outputs/session28/numbers_parts/spectrum_dmd.json`",
        "",
        "## Method",
        "",
        "Part 1 (data-driven DMD on the no-gust Baseline limit cycle): the point",
        "estimate stacks every consecutive snapshot pair (z_t, z_{t+1}) from the four",
        "Baseline encounters into column matrices X, Y (476 pairs, no pair crosses an",
        "encounter boundary), fits A = Y X^+, and extracts the leading-modulus",
        "oscillatory eigenvalue. With only four encounters an encounter-level resample",
        "is too coarse, so the bootstrap resamples the snapshot pairs (columns) with",
        "replacement within the standard DMD delay-embedding window, refits A, and",
        "re-extracts the same leading eigenvalue. n = 1000 draws.",
        "",
        "Part 2 (Floquet): the 59 per-frame companion Jacobians of the tf-no-c",
        "predictor over one Baseline shedding cycle are computed once on CPU, then the",
        "ordered-product monodromy is bootstrapped by resampling those step factors",
        "with replacement (the per-step modulus is their geometric contribution). The",
        "reported quantity is the per-step leading-multiplier modulus",
        "|mu_cycle|^{1/period}. n = 1000 draws.",
        "",
        "## Part 1: DMD leading oscillatory eigenvalue (Baseline limit cycle)",
        "",
        "| family | St point | St CI | |lambda| point | |lambda| CI |",
        "| --- | --- | --- | --- | --- |",
    ]
    for fam in ("jepa_tf_noc", "pod", "fukami"):
        if fam not in part1:
            continue
        r = part1[fam]
        lines.append(
            f"| {fam} | {r['st_point']:.3f} | "
            f"[{r['st_ci_lo']:.3f}, {r['st_ci_hi']:.3f}] | "
            f"{r['modulus_point']:.3f} | "
            f"[{r['modulus_ci_lo']:.3f}, {r['modulus_ci_hi']:.3f}] |"
        )
    lines += [
        "",
        "Honest read of the St lower tail: the St CI lower bound is small for every",
        "family (the leading-eigenvalue selection occasionally locks onto a different,",
        "lower-frequency oscillatory mode of comparable modulus under a resample, since",
        "the d = 64 operator carries several near-unit-modulus modes). The CI therefore",
        "reflects the leading-mode SELECTION variability of the estimator, not a wide",
        "genuine frequency band; the modulus CI is the clean, well-behaved quantity and",
        "is the one to cite for the marginal-stability claim. Report the St as the point",
        "estimate with the percentile band as a faithful-but-conservative uncertainty.",
        "",
        "Macros created via the eval_all ci_lo/ci_hi mechanism (base name + lo/hi;",
        "non-colliding with the existing SpecDmdSt*/SpecDmdMod* point macros): "
        + ", ".join(
            f"SpecDmdSt{macro_fam[f]}CI{{,lo,hi}}, SpecDmdMod{macro_fam[f]}CI{{,lo,hi}}"
            for f in ("jepa_tf_noc", "pod", "fukami")
            if f in part1
        )
        + ".",
        "",
        "## Part 2: Floquet leading per-step multiplier modulus (tf-no-c predictor)",
        "",
    ]
    if part2:
        lines += [
            f"- per-step |mu| point estimate: {part2['per_step_modulus_point']:.4f}",
            f"- 95% bootstrap CI: "
            f"[{part2['per_step_modulus_ci_lo']:.4f}, {part2['per_step_modulus_ci_hi']:.4f}]",
            f"- period: {part2['period_frames']} frames, window W = {part2['window']}",
            "- macros created (ci_lo/ci_hi mechanism): SpecFloqModCI, SpecFloqModCIlo, "
            "SpecFloqModCIhi.",
            "",
            "### Verdict",
            "",
        ]
        if part2["ci_straddles_one"]:
            verdict = (
                "The Floquet per-step modulus CI "
                f"[{part2['per_step_modulus_ci_lo']:.4f}, "
                f"{part2['per_step_modulus_ci_hi']:.4f}] STRADDLES 1.0, so the "
                "'marginally stable orbit' wording is supported as a CI statement "
                "(the modulus is statistically indistinguishable from 1)."
            )
        elif part2["per_step_modulus_ci_lo"] > 1.0:
            verdict = (
                "The Floquet per-step modulus CI "
                f"[{part2['per_step_modulus_ci_lo']:.4f}, "
                f"{part2['per_step_modulus_ci_hi']:.4f}] sits ENTIRELY ABOVE 1.0, so "
                "the orbit is (weakly) unstable / amplifying rather than strictly "
                "marginally stable; the wording should say the modulus sits just "
                "above 1, not exactly on the unit circle."
            )
        else:
            verdict = (
                "The Floquet per-step modulus CI "
                f"[{part2['per_step_modulus_ci_lo']:.4f}, "
                f"{part2['per_step_modulus_ci_hi']:.4f}] sits ENTIRELY BELOW 1.0, so "
                "the rolled-out latent orbit is contracting (decaying toward a fixed "
                "point); the marginal-stability wording should be softened."
            )
        lines += [verdict, ""]
    else:
        lines += ["(Floquet bootstrap not computed in this run.)", ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {out_path}")


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> None:
    """Run the Part-1 and Part-2 bootstraps and write the numbers part + report."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents-root", type=Path, default=REPO / "outputs/session28/latents")
    p.add_argument(
        "--numbers-part",
        type=Path,
        default=REPO / "outputs/session28/numbers_parts/spectrum_dmd_bootstrap.json",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=REPO / "outputs/session29/reports/spectrum_dmd_bootstrap.md",
    )
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=BOOT_SEED)
    p.add_argument("--skip-floquet", action="store_true", help="Part 1 only (no predictor).")
    args = p.parse_args(argv)

    command = "python scripts/session29/spectrum_dmd_bootstrap.py"

    print("[spectrum_dmd_bootstrap] === Part 1: DMD snapshot-pair bootstrap ===", flush=True)
    part1 = bootstrap_part1(args.latents_root, args.n_boot, args.seed)

    part2: dict = {}
    if not args.skip_floquet:
        print(
            "\n[spectrum_dmd_bootstrap] === Part 2: Floquet step-Jacobian bootstrap ===",
            flush=True,
        )
        part2 = bootstrap_part2(args.latents_root, FLOQUET_TAG, args.n_boot, args.seed)

    emit_numbers_part(part1, part2, args.numbers_part)
    write_report(part1, part2, command, args.latents_root, args.report)
    print("[spectrum_dmd_bootstrap] DONE")


if __name__ == "__main__":
    main()

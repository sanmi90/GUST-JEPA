"""Session 28 B2: the closure matrix, the centerpiece v2.1 evaluation (master plan B2).

PROBE REGIME DECLARATION (CLAUDE.md probe methodology): every probe in this script
is a STATE-DESCRIPTOR probe on PER-frame z. Probes are fitted on the family's TRAIN
per-frame latents (the ``z_full`` key, pooled over all frames) against per-frame DNS
observables, and read out at single frames ``impact + H``. No parameter (G, D, Y)
probe lives here; parameter probes use IMPACT-frame z elsewhere.

What it computes (frozen protocol: configs/eval_protocol_v2p1.yaml, D181)
    For every family x latent dim x encoder seed in the families manifest
    (scripts/session28/families_closure.yaml), x 6 observables x endpoint
    {representational, forecast} x probe class {ridge, krr_rbf, mlp_reg}:
    held-out R^2 = 1 - SSE/SST with SST about the held-out split's OWN mean,
    collected over per-encounter pairs (y_pred, y_true) at frame impact + H,
    H in {4, 8, 16, 32} (primary 16), on train + val (= the on-disk test_a)
    + test_b (interior / boundary tiers separately and pooled) + test_c.

Endpoints
    representational  probe applied to the family's encoder latents of held-out
                      DNS frames (``z_full`` in outputs/session28/latents/<tag>/).
    forecast          probe applied to the full-context autoregressive rollout
                      (``z_full`` in outputs/session28/rollouts/<key>/). z_markov
                      is RETIRED (protocol rollout.mode = full_context) and never
                      read. Forecast rows exist only where the families manifest
                      declares a matched predictor-on-top rollout key (B1 covers
                      the s42/canonical latents per family x d at first).

Probe protocol (faithful port of scripts/session18/physical_metrics_from_rollouts.py)
    fit_ridge / fit_krr_rbf / fit_mlp_reg / apply_probe are verbatim ports of the
    v2-era machinery (D129/D131) with type hints added. Probes are fitted on TRAIN
    per-frame latents only. The protocol's "5-fold case-level CV on the readout" is
    implemented as: 5 case-level folds on train (no case straddles folds); the
    train-split rows report OUT-OF-FOLD predictions (each train case is predicted
    by the probe fitted without it); held-out splits use the probe refitted on all
    train cases. Probe hyperparameters stay fixed at the v2 values (no CV search).

Uncertainty (through scripts/session28/stats_lib.py, D165 conventions)
    Per cell, the per-encounter normalised squared errors
    u_i = (y_pred_i - y_true_i)^2 / (SST/n) satisfy R^2 = 1 - mean(u), so the
    encounter bootstrap CI (n = 2000, stats_lib.encounter_bootstrap_ci) and, on
    every wake_enstrophy row, the case-clustered CI
    (stats_lib.case_cluster_bootstrap, pooled-encounter mean, n = 10000) map
    affinely onto R^2 CIs as [1 - hi(u), 1 - lo(u)]. The SST denominator (the
    held-out split's own variance) is held FIXED under resampling, per the
    protocol's wording that SST is about the held-out split's own mean; the
    bootstrap therefore captures SSE variability. Each cell draws from a fresh
    np.random.default_rng(boot_seed), so cell values are independent of which
    other artifacts exist (the script is re-run as the extraction chain lands;
    Holm multiplicity over the 12-test family is the B6 harvest, not done here).

Outputs (overwritten on every run; the run is idempotent given the artifacts)
    outputs/session28/closure_matrix/matrix.csv     every cell: family, tag,
        objective, d, seed (-1 = deterministic family), observable, endpoint,
        probe, split, tier, H, n, value, ci_lo, ci_hi, cc_ci_lo, cc_ci_hi
        (cc_* = case-clustered, wake rows only, NaN elsewhere)
    outputs/session28/closure_matrix/matrix.npz     same table, one array per column
    outputs/session28/closure_matrix/matrix.parquet same table (only if a parquet
        engine is installed; logged and skipped otherwise)
    outputs/session28/closure_matrix/matrix_meta.json   provenance (protocol +
        manifest sha256, git commit, config, skip log)
    outputs/session28/closure_matrix/gate_gd.json   Gate GD verdict block (also
        printed): is the family ordering on representational wake enstrophy
        preserved under all three probe classes at d = 64?
    outputs/session28/numbers_parts/closure_headline.json   macro-bound headline
        cells for eval_all.py: representational wake_enstrophy R^2 at H = 16,
        test_b pooled, ridge, per family at d = 64, seed mean +- sd
        (macros NumReprWake<Family>; lo/hi = case-clustered CI of the lead seed).

Usage (CPU-only: numpy + sklearn on NPZ latents; no torch, no GPU; run niced and
thread-capped per the 2026-06-12 shared-workstation hygiene block so the RTX 6000
training queue and other users keep interactive priority):

    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    nice -n 10 python scripts/session28/closure_matrix.py
    nice -n 10 python scripts/session28/closure_matrix.py \\
        --families jepa_tf_noc fukami pod --probe-classes ridge --splits test_b

    Full matrix with all three probe classes refits ~6 probes per (member,
    observable, class); the MLP class dominates runtime (hours for the full
    44-member matrix; ridge alone runs in minutes). Subset flags exist for
    incremental runs while artifacts land; the final paper run is one full pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stats_lib  # noqa: E402

ENDPOINTS = ("representational", "forecast")
# On-disk artifacts (latents NPZ filenames, DNS-metrics key prefixes) name the
# validation split by its v1 name test_a; the protocol reports it as val.
SPLIT_ALIASES = {"val": ("val", "test_a")}
DETERMINISTIC_SEED = -1  # recorded seed for seedless (deterministic) families


# --------------------------------------------------------------------------- config


@dataclass
class Config:
    """Resolved run configuration (protocol defaults + CLI overrides)."""

    families_manifest: Path
    protocol_path: Path
    dns_metrics: Path
    latents_root: Path
    rollouts_root: Path
    split_manifest: Path
    out_dir: Path
    numbers_part: Path
    observables: list[str]
    probe_classes: list[str]
    splits: list[str]
    horizons: list[int]
    test_b_tiers: list[str]
    primary_horizon: int = 16
    primary_observable: str = "wake_enstrophy"
    primary_probe: str = "ridge"
    headline_d: int = 64
    n_boot_enc: int = stats_lib.N_BOOT_ENC
    n_boot_case: int = stats_lib.N_BOOT_CASE
    n_folds: int = 5
    cv_seed: int = 0
    boot_seed: int = 0
    ridge_alpha: float = 1.0
    min_pairs: int = 3
    family_filter: list[str] | None = None
    dim_filter: list[int] | None = None


@dataclass
class Member:
    """One (family, d, seed) cell of the training matrix."""

    family: str
    objective: str
    macro_family: str
    d: int
    seed: int  # DETERMINISTIC_SEED for seedless families
    tag: str
    latents_dir: Path
    rollouts_dir: Path | None
    in_gate: bool = True


@dataclass
class SplitData:
    """One endpoint's latent trajectories for one split, DNS-matched."""

    z: np.ndarray  # (n, T, d) float32
    case_ids: np.ndarray  # (n,) str
    impact: np.ndarray  # (n,) int64
    y: dict[str, np.ndarray] = field(default_factory=dict)  # obs -> (n, T) float64


# ------------------------------------------------------------------- manifest loading


def load_protocol(path: Path) -> dict:
    """Read the frozen evaluation protocol YAML."""
    return yaml.safe_load(path.read_text())


def load_families(
    manifest_path: Path,
    latents_root: Path,
    rollouts_root: Path,
    family_filter: list[str] | None = None,
    dim_filter: list[int] | None = None,
) -> list[Member]:
    """Expand the families manifest into the flat member list.

    Args:
        manifest_path: families YAML (see scripts/session28/families_closure.yaml).
        latents_root: directory containing one latents dir per member tag.
        rollouts_root: directory containing one rollout dir per matched-predictor key.
        family_filter: optional subset of family names.
        dim_filter: optional subset of latent dims.

    Returns:
        Members in manifest order. Existence of the latents dir is NOT checked
        here; missing members are skipped (and logged) at evaluation time.
    """
    blob = yaml.safe_load(manifest_path.read_text())
    default_tmpl = blob.get("defaults", {}).get("tag_template", "{family}_d{d}_s{seed}")
    members: list[Member] = []
    for fam, spec in blob["families"].items():
        if family_filter and fam not in family_filter:
            continue
        tmpl = spec.get("tag_template", default_tmpl)
        roll = spec.get("rollouts") or {}
        roll_keys = {int(k): v for k, v in (roll.get("keys") or {}).items()}
        roll_seed = roll.get("seed")
        roll_seed = DETERMINISTIC_SEED if roll_seed is None else int(roll_seed)
        for d_key, seeds in spec["dims"].items():
            d = int(d_key)
            if dim_filter and d not in dim_filter:
                continue
            seed_list = [int(s) for s in seeds] if seeds else [DETERMINISTIC_SEED]
            for seed in seed_list:
                tag = tmpl.format(family=fam, d=d, seed=seed)
                rollouts_dir = None
                if d in roll_keys and seed == roll_seed:
                    rollouts_dir = rollouts_root / str(roll_keys[d])
                members.append(
                    Member(
                        family=fam,
                        objective=spec["objective"],
                        macro_family=spec["macro_family"],
                        d=d,
                        seed=seed,
                        tag=tag,
                        latents_dir=latents_root / tag,
                        rollouts_dir=rollouts_dir,
                        in_gate=bool(spec.get("gate", True)),
                    )
                )
    return members


def load_tier_map(split_manifest: Path) -> dict[str, str | None]:
    """Map case_id -> test_b tier from the split manifest's case annotations."""
    cases = json.loads(split_manifest.read_text())["cases"]
    return {cid: c.get("tier") for cid, c in cases.items()}


# ----------------------------------------------------------------------- NPZ loading


def npz_get(blob: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    """First present key among ``names`` (tolerates singular/plural conventions)."""
    for n in names:
        if n in blob.files:
            return blob[n]
    raise KeyError(f"none of {names} present in npz (keys: {sorted(blob.files)})")


def find_split_file(directory: Path, split: str) -> Path | None:
    """Resolve <split>.npz under ``directory``, tolerating the val/test_a alias."""
    for alias in SPLIT_ALIASES.get(split, (split,)):
        p = directory / f"{alias}.npz"
        if p.exists():
            return p
    return None


def dns_split_array(dns: np.lib.npyio.NpzFile, split: str, suffix: str) -> np.ndarray:
    """DNS-metrics array ``{split}_{suffix}``, tolerating the val/test_a alias."""
    for alias in SPLIT_ALIASES.get(split, (split,)):
        key = f"{alias}_{suffix}"
        if key in dns.files:
            return dns[key]
    raise KeyError(f"no '{split}_{suffix}' (or alias) in DNS metrics npz")


def match_index(
    cid_a: np.ndarray, ei_a: np.ndarray, cid_b: np.ndarray, ei_b: np.ndarray
) -> np.ndarray:
    """Index into (cid_b, ei_b) for each (cid_a[i], ei_a[i]); -1 where missing."""
    idx = {(str(c), int(e)): i for i, (c, e) in enumerate(zip(cid_b, ei_b))}
    return np.array([idx.get((str(c), int(e)), -1) for c, e in zip(cid_a, ei_a)], dtype=np.int64)


def load_split_data(
    directory: Path,
    split: str,
    dns: np.lib.npyio.NpzFile,
    observables: list[str],
    label: str,
) -> SplitData | None:
    """Load one split's latents/rollouts NPZ and align it with the DNS metrics.

    Reads the ``z_full`` key ONLY (representational latents and full-context
    rollouts share the key name; z_markov is retired and never read). Encounters
    without a matching DNS row are dropped with a logged count.

    Returns:
        SplitData, or None when the split file is absent.
    """
    path = find_split_file(directory, split)
    if path is None:
        return None
    blob = np.load(path, allow_pickle=True)
    z = blob["z_full"].astype(np.float32)
    cids = np.array([str(c) for c in npz_get(blob, "case_ids", "case_id").tolist()])
    eis = npz_get(blob, "encounter_indices", "encounter_index").astype(np.int64)
    impact = npz_get(blob, "impact_frame").astype(np.int64)

    di = match_index(
        cids,
        eis,
        dns_split_array(dns, split, "case_id"),
        dns_split_array(dns, split, "encounter_index"),
    )
    keep = di >= 0
    if not keep.all():
        print(f"[closure] {label} {split}: {int((~keep).sum())} encounters dropped (no DNS row)")
    z, cids, impact, di = z[keep], cids[keep], impact[keep], di[keep]

    dns_impact = dns_split_array(dns, split, "impact_frame").astype(np.int64)[di]
    n_mismatch = int((dns_impact != impact).sum())
    if n_mismatch:
        print(
            f"[closure] WARNING {label} {split}: impact_frame differs from DNS metrics "
            f"on {n_mismatch} encounters; using the latents' impact_frame "
            "(the rollout was seeded from it)"
        )
    y = {obs: dns_split_array(dns, split, obs)[di].astype(np.float64) for obs in observables}
    return SplitData(z=z, case_ids=cids, impact=impact, y=y)


# ------------------------------------------------------ probes (verbatim v2-era ports)


def fit_ridge(Z: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict:
    """Standardise Z, fit linear ridge (port of physical_metrics_from_rollouts)."""
    Z = Z.astype(np.float64)
    y = y.astype(np.float64)
    mu_z = Z.mean(axis=0)
    sigma_z = Z.std(axis=0).clip(min=1e-9)
    Zn = (Z - mu_z) / sigma_z
    yc = y - y.mean()
    A = Zn.T @ Zn + alpha * np.eye(Zn.shape[1])
    W = np.linalg.solve(A, Zn.T @ yc)
    return {"kind": "ridge", "W": W, "mu_z": mu_z, "sigma_z": sigma_z, "b": float(y.mean())}


def fit_krr_rbf(
    Z: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    gamma: float = 0.01,
    n_subsample: int = 4000,
) -> dict:
    """Kernel ridge (RBF), subsampled for bounded cost (verbatim v2 port)."""
    from sklearn.kernel_ridge import KernelRidge

    Z = Z.astype(np.float64)
    y = y.astype(np.float64)
    mu_z = Z.mean(axis=0)
    sigma_z = Z.std(axis=0).clip(min=1e-9)
    Zn = (Z - mu_z) / sigma_z
    if len(Zn) > n_subsample:
        idx = np.random.RandomState(0).choice(len(Zn), n_subsample, replace=False)
        Zn = Zn[idx]
        y = y[idx]
    model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    model.fit(Zn, y)
    return {"kind": "krr_rbf", "model": model, "mu_z": mu_z, "sigma_z": sigma_z}


def fit_mlp_reg(
    Z: np.ndarray,
    y: np.ndarray,
    hidden: int = 64,
    alpha: float = 1e-2,
    max_iter: int = 200,
) -> dict:
    """Small regularised MLP regressor (verbatim v2 port)."""
    from sklearn.neural_network import MLPRegressor

    Z = Z.astype(np.float64)
    y = y.astype(np.float64)
    mu_z = Z.mean(axis=0)
    sigma_z = Z.std(axis=0).clip(min=1e-9)
    Zn = (Z - mu_z) / sigma_z
    model = MLPRegressor(
        hidden_layer_sizes=(hidden, hidden),
        alpha=alpha,
        max_iter=max_iter,
        random_state=0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    model.fit(Zn, y)
    return {"kind": "mlp_reg", "model": model, "mu_z": mu_z, "sigma_z": sigma_z}


def fit_probe(Z: np.ndarray, y: np.ndarray, probe_class: str, ridge_alpha: float = 1.0) -> dict:
    """Dispatch to one of the three protocol probe classes."""
    if probe_class == "ridge":
        return fit_ridge(Z, y, alpha=ridge_alpha)
    if probe_class == "krr_rbf":
        return fit_krr_rbf(Z, y)
    if probe_class == "mlp_reg":
        return fit_mlp_reg(Z, y)
    raise ValueError(f"unknown probe class {probe_class!r}")


def apply_probe(z: np.ndarray, probe: dict) -> np.ndarray:
    """Apply a fitted probe to (n, d) latents -> (n,) predictions."""
    Zn = (z.astype(np.float64) - probe["mu_z"]) / probe["sigma_z"]
    if probe["kind"] == "ridge":
        return Zn @ probe["W"] + probe["b"]
    return probe["model"].predict(Zn)


# ----------------------------------------------------------- estimator + fold helpers


def r2_heldout(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Protocol R^2: 1 - SSE/SST with SST about the held-out sample's own mean."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    sse = float(((y_pred - y_true) ** 2).sum())
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-12)


def make_case_folds(case_ids: np.ndarray, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Assign each encounter to one of ``n_folds`` CASE-level folds.

    Unique cases are sorted lexicographically (stats_lib convention), shuffled
    with a seeded generator, and dealt round-robin into folds, so no case
    straddles folds and the assignment is deterministic for a given seed.

    Returns:
        (n_enc,) int64 fold id per encounter.
    """
    cids = [str(c) for c in np.asarray(case_ids).tolist()]
    uc = sorted(set(cids))
    order = np.random.default_rng(seed).permutation(len(uc))
    fold_of_case = {uc[j]: int(rank % n_folds) for rank, j in enumerate(order)}
    return np.array([fold_of_case[c] for c in cids], dtype=np.int64)


def gather_at_horizons(
    z: np.ndarray, impact: np.ndarray, horizons: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Slice per-encounter frames at exactly ``impact + H`` for each horizon.

    H counts frames PAST the impact frame: H = 16 reads frame impact + 16
    (off-by-one guard: never impact + H - 1). Encounters whose impact + H falls
    outside [0, T) are flagged invalid (and excluded from that horizon's cell).

    Args:
        z: (n, T, d) trajectories (latents or per-frame targets with d = 1).
        impact: (n,) per-encounter impact frames.
        horizons: list of horizons H.

    Returns:
        (z_h, valid): (n, n_H, d) gathered frames and (n, n_H) validity mask.
    """
    n, T, _ = z.shape
    h_arr = np.asarray(horizons, dtype=np.int64)
    frames = np.asarray(impact, dtype=np.int64)[:, None] + h_arr[None, :]
    valid = (frames >= 0) & (frames < T)
    safe = np.clip(frames, 0, T - 1)
    z_h = z[np.arange(n)[:, None], safe, :]
    return z_h, valid


def predict_at_horizons(z_h: np.ndarray, probe: dict) -> np.ndarray:
    """Apply a probe to (n, n_H, d) gathered frames -> (n, n_H) predictions."""
    n, n_h, d = z_h.shape
    return apply_probe(z_h.reshape(n * n_h, d), probe).reshape(n, n_h)


# ------------------------------------------------------------------- cell uncertainty


def cell_uncertainty(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    case_ids: np.ndarray,
    with_case_ci: bool,
    n_boot_enc: int,
    n_boot_case: int,
    boot_seed: int,
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """Held-out R^2 with encounter-bootstrap and (optional) case-clustered CIs.

    Uses stats_lib on the per-encounter normalised squared errors
    u_i = (y_pred_i - y_true_i)^2 / (SST/n), for which R^2 = 1 - mean(u): CIs of
    mean(u) map affinely onto R^2 CIs (see module docstring for the fixed-SST
    convention). A fresh default_rng(boot_seed) per cell keeps every cell
    reproducible independently of evaluation order.

    Returns:
        (r2, (ci_lo, ci_hi), (cc_ci_lo, cc_ci_hi)); cc bounds are NaN unless
        ``with_case_ci``.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    msd = max(sst / y_true.size, 1e-12)
    u = (y_pred - y_true) ** 2 / msd
    r2 = 1.0 - float(u.mean())

    rng = np.random.default_rng(boot_seed)
    lo_u, hi_u = stats_lib.encounter_bootstrap_ci(u, n_boot=n_boot_enc, rng=rng)
    ci = (1.0 - hi_u, 1.0 - lo_u)
    cc = (float("nan"), float("nan"))
    if with_case_ci:
        cb = stats_lib.case_cluster_bootstrap(u, case_ids, n_boot=n_boot_case, rng=rng)
        cc = (1.0 - cb["enc_mean_ci"][1], 1.0 - cb["enc_mean_ci"][0])
    return r2, ci, cc


# --------------------------------------------------------------------- member harvest


def fit_member_probes(
    train: SplitData, cfg: Config, need_oof: bool, label: str
) -> tuple[dict, dict[str, int]]:
    """Fit the probe suite on the member's TRAIN per-frame latents.

    Args:
        train: DNS-matched train split (z (n, T, d), per-frame targets).
        cfg: run configuration.
        need_oof: also fit the 5 per-fold probes (only needed for train rows).
        label: member tag for log lines.

    Returns:
        (probes, fold_of_case) where probes[obs][cls] = {"full": probe,
        "folds": {fold_id: probe} or {}} and fold_of_case maps case_id -> fold.
    """
    n, t_len, d = train.z.shape
    z_flat = train.z.reshape(n * t_len, d)
    fold_id = make_case_folds(train.case_ids, n_folds=cfg.n_folds, seed=cfg.cv_seed)
    fold_of_case = {str(c): int(f) for c, f in zip(train.case_ids, fold_id)}

    probes: dict[str, dict[str, dict]] = {}
    for obs in cfg.observables:
        y_flat = train.y[obs].reshape(n * t_len)
        probes[obs] = {}
        for cls in cfg.probe_classes:
            suite: dict = {"full": fit_probe(z_flat, y_flat, cls, cfg.ridge_alpha), "folds": {}}
            if need_oof:
                for k in sorted(set(fold_id.tolist())):
                    in_fit = fold_id != k
                    suite["folds"][k] = fit_probe(
                        train.z[in_fit].reshape(-1, d),
                        train.y[obs][in_fit].reshape(-1),
                        cls,
                        cfg.ridge_alpha,
                    )
            y_hat = apply_probe(z_flat, suite["full"])
            suite["fit_r2"] = r2_heldout(y_hat, y_flat)
            probes[obs][cls] = suite
        print(
            f"[closure] {label} fit {obs}: "
            + ", ".join(f"{c} R^2={probes[obs][c]['fit_r2']:.3f}" for c in cfg.probe_classes)
        )
    return probes, fold_of_case


def tier_iter(
    split: str, case_ids: np.ndarray, tier_map: dict[str, str | None], cfg: Config
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (tier_label, encounter mask) per the protocol's test_b tier reporting."""
    n = case_ids.size
    if split != "test_b":
        yield "all", np.ones(n, dtype=bool)
        return
    for tier in cfg.test_b_tiers:
        if tier == "pooled":
            yield "pooled", np.ones(n, dtype=bool)
        else:
            yield tier, np.array([tier_map.get(str(c)) == tier for c in case_ids])


def evaluate_member(
    member: Member,
    dns: np.lib.npyio.NpzFile,
    tier_map: dict[str, str | None],
    cfg: Config,
) -> tuple[list[dict], list[str]]:
    """All closure-matrix rows for one (family, d, seed) member.

    Returns:
        (rows, skip_log). Missing latents skip the member; missing rollout dirs
        or split files skip the affected endpoint/split only.
    """
    skips: list[str] = []
    train = load_split_data(member.latents_dir, "train", dns, cfg.observables, member.tag)
    if train is None:
        msg = f"SKIP member {member.tag}: no train latents under {member.latents_dir}"
        print(f"[closure] {msg}")
        return [], [msg]

    need_oof = "train" in cfg.splits
    probes, fold_of_case = fit_member_probes(train, cfg, need_oof, member.tag)

    rows: list[dict] = []
    for endpoint in ENDPOINTS:
        if endpoint == "forecast":
            if member.rollouts_dir is None:
                continue
            if not member.rollouts_dir.exists():
                msg = f"SKIP {member.tag} forecast: rollouts dir {member.rollouts_dir} missing"
                print(f"[closure] {msg}")
                skips.append(msg)
                continue
            source = member.rollouts_dir
        else:
            source = member.latents_dir

        for split in cfg.splits:
            data = load_split_data(source, split, dns, cfg.observables, member.tag)
            if data is None:
                msg = f"skip {member.tag} {endpoint}/{split}: no {split}.npz under {source}"
                print(f"[closure] {msg}")
                skips.append(msg)
                continue
            z_h, valid = gather_at_horizons(data.z, data.impact, cfg.horizons)
            for cls in cfg.probe_classes:
                for obs in cfg.observables:
                    y_h, _ = gather_at_horizons(data.y[obs][:, :, None], data.impact, cfg.horizons)
                    y_h = y_h[:, :, 0]
                    if split == "train":
                        preds = np.full(z_h.shape[:2], np.nan)
                        for k, probe_k in probes[obs][cls]["folds"].items():
                            m = np.array([fold_of_case.get(str(c)) == k for c in data.case_ids])
                            if m.any():
                                preds[m] = predict_at_horizons(z_h[m], probe_k)
                    else:
                        preds = predict_at_horizons(z_h, probes[obs][cls]["full"])
                    for hi, h in enumerate(cfg.horizons):
                        ok = valid[:, hi] & np.isfinite(preds[:, hi])
                        for tier, mask in tier_iter(split, data.case_ids, tier_map, cfg):
                            sel = ok & mask
                            n_pairs = int(sel.sum())
                            if n_pairs < cfg.min_pairs:
                                continue
                            r2, ci, cc = cell_uncertainty(
                                preds[sel, hi],
                                y_h[sel, hi],
                                data.case_ids[sel],
                                with_case_ci=(obs == cfg.primary_observable),
                                n_boot_enc=cfg.n_boot_enc,
                                n_boot_case=cfg.n_boot_case,
                                boot_seed=cfg.boot_seed,
                            )
                            rows.append(
                                {
                                    "family": member.family,
                                    "tag": member.tag,
                                    "objective": member.objective,
                                    "d": member.d,
                                    "seed": member.seed,
                                    "observable": obs,
                                    "endpoint": endpoint,
                                    "probe": cls,
                                    "split": split,
                                    "tier": tier,
                                    "H": int(h),
                                    "n": n_pairs,
                                    "value": r2,
                                    "ci_lo": ci[0],
                                    "ci_hi": ci[1],
                                    "cc_ci_lo": cc[0],
                                    "cc_ci_hi": cc[1],
                                }
                            )
            print(
                f"[closure] {member.tag} {endpoint}/{split}: "
                f"{sum(1 for r in rows if r['endpoint'] == endpoint and r['split'] == split)} rows"
            )
    return rows, skips


def run_matrix(cfg: Config) -> dict:
    """Compute every available closure-matrix cell.

    Returns:
        {"rows": list of cell dicts, "skips": skip-log lines,
         "members": list of evaluated member tags}.
    """
    members = load_families(
        cfg.families_manifest,
        cfg.latents_root,
        cfg.rollouts_root,
        cfg.family_filter,
        cfg.dim_filter,
    )
    tier_map = load_tier_map(cfg.split_manifest)
    dns = np.load(cfg.dns_metrics, allow_pickle=True)
    print(f"[closure] {len(members)} manifest members; DNS metrics: {cfg.dns_metrics}")

    rows: list[dict] = []
    skips: list[str] = []
    evaluated: list[str] = []
    gate_membership: dict[str, bool] = {}
    for member in members:
        m_rows, m_skips = evaluate_member(member, dns, tier_map, cfg)
        rows.extend(m_rows)
        skips.extend(m_skips)
        gate_membership[member.tag] = member.in_gate
        if m_rows:
            evaluated.append(member.tag)
    for r in rows:
        r["in_gate"] = gate_membership[r["tag"]]
    print(f"[closure] total {len(rows)} cells from {len(evaluated)} members")
    return {"rows": rows, "skips": skips, "members": evaluated}


# --------------------------------------------------------------------------- outputs

MATRIX_COLUMNS = (
    "family",
    "tag",
    "objective",
    "d",
    "seed",
    "observable",
    "endpoint",
    "probe",
    "split",
    "tier",
    "H",
    "n",
    "value",
    "ci_lo",
    "ci_hi",
    "cc_ci_lo",
    "cc_ci_hi",
)


def write_matrix(rows: list[dict], out_dir: Path) -> None:
    """Write matrix.csv + matrix.npz (+ matrix.parquet when an engine exists)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "matrix.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(MATRIX_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[closure] wrote {len(rows)} rows -> {csv_path}")

    arrays = {}
    for col in MATRIX_COLUMNS:
        vals = [r[col] for r in rows]
        if col in ("d", "seed", "H", "n"):
            arrays[col] = np.asarray(vals, dtype=np.int64)
        elif col in ("value", "ci_lo", "ci_hi", "cc_ci_lo", "cc_ci_hi"):
            arrays[col] = np.asarray(vals, dtype=np.float64)
        else:
            arrays[col] = np.asarray([str(v) for v in vals])
    np.savez(out_dir / "matrix.npz", **arrays)
    print(f"[closure] wrote {out_dir / 'matrix.npz'}")

    try:
        import pandas as pd

        pd.DataFrame(rows, columns=list(MATRIX_COLUMNS)).to_parquet(out_dir / "matrix.parquet")
        print(f"[closure] wrote {out_dir / 'matrix.parquet'}")
    except Exception as exc:  # no pandas / no parquet engine: npz + csv suffice
        print(f"[closure] parquet not written ({exc}); npz + csv are canonical")


def headline_rows(rows: list[dict], cfg: Config) -> list[dict]:
    """The macro-bound headline slice (primary endpoint cells at fixed d)."""
    return [
        r
        for r in rows
        if r["endpoint"] == "representational"
        and r["observable"] == cfg.primary_observable
        and r["H"] == cfg.primary_horizon
        and r["split"] == "test_b"
        and r["tier"] == "pooled"
        and r["probe"] == cfg.primary_probe
        and r["d"] == cfg.headline_d
    ]


def emit_headline_part(
    rows: list[dict], members: list[Member], cfg: Config, out_path: Path
) -> dict:
    """Write the closure_headline numbers part for eval_all.py.

    One record per family at d = headline_d: value = seed mean of the primary
    cell (representational wake_enstrophy R^2 at H = primary, test_b pooled,
    primary probe), seed_sd over encoder seeds (0 for single-member families),
    ci = case-clustered CI of the lead seed (42 when present). Macro name:
    NumReprWake<macro_family>.
    """
    cells = headline_rows(rows, cfg)
    macro_of_family = {m.family: m.macro_family for m in members}
    numbers: dict[str, dict] = {}
    for family in sorted({r["family"] for r in cells}):
        fam_cells = sorted(
            (r for r in cells if r["family"] == family),
            key=lambda r: (r["seed"] != 42, r["seed"]),  # lead seed 42 first
        )
        vals = np.array([r["value"] for r in fam_cells], dtype=np.float64)
        lead = fam_cells[0]
        seeds = [r["seed"] for r in fam_cells]
        rec = {
            "macro": f"NumReprWake{macro_of_family[family]}",
            "value": float(vals.mean()),
            "fmt": "%.2f",
            "seed_mean": float(vals.mean()),
            "seed_sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "n": int(vals.size),
            "split": "test_b",
            "endpoint": "representational",
            "probe": cfg.primary_probe,
            "observable": cfg.primary_observable,
            "horizon": int(cfg.primary_horizon),
            "run_tags": [r["tag"] for r in fam_cells],
            "source": "closure_matrix.py",
            "note": (
                f"d={cfg.headline_d}, tier pooled, seeds {seeds}, n_enc={lead['n']}; "
                f"ci = case-clustered CI of lead seed {lead['seed']}"
            ),
        }
        if np.isfinite(lead["cc_ci_lo"]):
            rec["ci_lo"] = float(lead["cc_ci_lo"])
            rec["ci_hi"] = float(lead["cc_ci_hi"])
        numbers[f"closure_repr_wake_H{cfg.primary_horizon}_testb_{family}"] = rec
    part = {"part": "closure_headline", "numbers": numbers}
    if numbers:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(part, indent=2))
        print(f"[closure] wrote {len(numbers)} headline numbers -> {out_path}")
    else:
        print("[closure] no headline cells available yet; numbers part NOT written")
    return part


def gate_gd_verdict(rows: list[dict], cfg: Config) -> dict:
    """Gate GD (master plan B2): probe-class robustness of the wake-closure ordering.

    Strong branch: on representational wake enstrophy (H = primary, test_b
    pooled, d = headline_d) every predictive family stays above every
    reconstructive family under ALL THREE probe classes, with non-overlapping
    case-clustered CIs (min predictive cc_ci_lo > max reconstructive cc_ci_hi,
    over member cells). Weak branch: a NONLINEAR probe (krr_rbf / mlp_reg)
    recovers the reconstructive wake to within 0.15 of the predictive one.
    Families flagged ``gate: false`` in the manifest (the conditioned reference)
    are excluded. Incomplete until all three probe classes have cells.
    """
    base = [
        r
        for r in rows
        if r["endpoint"] == "representational"
        and r["observable"] == cfg.primary_observable
        and r["H"] == cfg.primary_horizon
        and r["split"] == "test_b"
        and r["tier"] == "pooled"
        and r["d"] == cfg.headline_d
        and r.get("in_gate", True)
    ]
    per_probe: dict[str, dict] = {}
    for cls in sorted({r["probe"] for r in base}):
        cells = [r for r in base if r["probe"] == cls]
        fam_mean = {
            fam: float(np.mean([r["value"] for r in cells if r["family"] == fam]))
            for fam in sorted({r["family"] for r in cells})
        }
        obj_of = {r["family"]: r["objective"] for r in cells}
        pred = {f: v for f, v in fam_mean.items() if obj_of[f] == "predictive"}
        recon = {f: v for f, v in fam_mean.items() if obj_of[f] == "reconstructive"}
        ranking = sorted(fam_mean, key=fam_mean.get, reverse=True)
        entry: dict = {
            "ranking": ranking,
            "family_seed_mean": fam_mean,
            "n_predictive": len(pred),
            "n_reconstructive": len(recon),
        }
        if pred and recon:
            entry["predictive_floor"] = min(pred.values())
            entry["reconstructive_ceiling"] = max(recon.values())
            entry["predictive_above_reconstructive"] = (
                entry["predictive_floor"] > entry["reconstructive_ceiling"]
            )
            cc_lo_pred = [
                r["cc_ci_lo"]
                for r in cells
                if r["objective"] == "predictive" and np.isfinite(r["cc_ci_lo"])
            ]
            cc_hi_recon = [
                r["cc_ci_hi"]
                for r in cells
                if r["objective"] == "reconstructive" and np.isfinite(r["cc_ci_hi"])
            ]
            entry["case_ci_separated"] = bool(
                cc_lo_pred and cc_hi_recon and min(cc_lo_pred) > max(cc_hi_recon)
            )
            if cls in ("krr_rbf", "mlp_reg"):
                entry["weak_trigger"] = bool(
                    entry["reconstructive_ceiling"] >= entry["predictive_floor"] - 0.15
                )
        per_probe[cls] = entry

    expected = set(cfg.probe_classes)
    have = {c for c, e in per_probe.items() if "predictive_above_reconstructive" in e}
    rankings = [tuple(e["ranking"]) for e in per_probe.values() if e["ranking"]]
    verdict: dict = {
        "gate": "GD",
        "cell": (
            f"representational {cfg.primary_observable} R^2, H={cfg.primary_horizon}, "
            f"test_b pooled, d={cfg.headline_d}, seed means"
        ),
        "probe_classes_evaluated": sorted(per_probe),
        "probe_classes_required": sorted(expected),
        "rankings_identical": bool(rankings) and len(set(rankings)) == 1,
        "per_probe": per_probe,
        "required_rewrite_either_branch": (
            "rewrite 'no probe can recover information the latent does not carry' as "
            "'no probe in the evaluated class recovers ...' (B2, D183)"
        ),
    }
    weak = any(e.get("weak_trigger", False) for e in per_probe.values())
    if have < {"ridge", "krr_rbf", "mlp_reg"} or have < expected:
        verdict["branch"] = "incomplete"
        verdict["reason"] = "not all three probe classes have both objectives evaluated"
    elif weak:
        verdict["branch"] = "weak"
        verdict["reason"] = (
            "a nonlinear probe recovers the reconstructive wake within 0.15 of the "
            "predictive one: rewrite carries/encodes claims as LINEAR DECODABILITY"
        )
    elif all(
        e.get("predictive_above_reconstructive", False) and e.get("case_ci_separated", False)
        for e in per_probe.values()
    ):
        verdict["branch"] = "strong"
        verdict["reason"] = (
            "family ordering preserved under all three probe classes with "
            "non-overlapping case-clustered CIs at d=64"
        )
    else:
        verdict["branch"] = "mixed"
        verdict["reason"] = (
            "ordering or CI separation fails under at least one probe class without "
            "triggering the weak branch; review per_probe before writing D183"
        )
    return verdict


def print_gate(verdict: dict) -> None:
    """Human-readable Gate GD block on stdout."""
    print("\n[closure] ================ GATE GD VERDICT ================")
    print(f"[closure] cell    : {verdict['cell']}")
    print(f"[closure] branch  : {verdict['branch'].upper()} ({verdict['reason']})")
    for cls, e in verdict["per_probe"].items():
        if "predictive_floor" in e:
            print(
                f"[closure]   {cls:8s}: pred_floor={e['predictive_floor']:+.3f} "
                f"recon_ceiling={e['reconstructive_ceiling']:+.3f} "
                f"ordered={e['predictive_above_reconstructive']} "
                f"cc_ci_separated={e.get('case_ci_separated')}"
            )
        ranked = ", ".join(f"{f}={e['family_seed_mean'][f]:+.3f}" for f in e["ranking"])
        print(f"[closure]   {cls:8s} ranking: {ranked}")
    print(f"[closure] rankings identical across probes: {verdict['rankings_identical']}")
    print("[closure] ==================================================\n")


# ------------------------------------------------------------------------------- CLI


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file (provenance stamping)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    """Current git commit of the source tree (UNKNOWN outside a repo)."""
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "UNKNOWN"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI; defaults come from the frozen protocol where it specifies them."""
    p = argparse.ArgumentParser(
        description="Session 28 B2 closure matrix (see module docstring).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--protocol", type=Path, default=REPO / "configs/eval_protocol_v2p1.yaml")
    p.add_argument(
        "--families-manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "families_closure.yaml",
    )
    p.add_argument(
        "--dns-metrics",
        type=Path,
        default=REPO / "outputs/session28/exp2/dns_physical_metrics.npz",
    )
    p.add_argument("--latents-root", type=Path, default=REPO / "outputs/session28/latents")
    p.add_argument("--rollouts-root", type=Path, default=REPO / "outputs/session28/rollouts")
    p.add_argument("--split-manifest", type=Path, default=REPO / "configs/splits/split_v2p1.json")
    p.add_argument("--out-dir", type=Path, default=REPO / "outputs/session28/closure_matrix")
    p.add_argument(
        "--numbers-part",
        type=Path,
        default=REPO / "outputs/session28/numbers_parts/closure_headline.json",
    )
    p.add_argument("--families", nargs="+", default=None, help="Subset of family names.")
    p.add_argument("--dims", nargs="+", type=int, default=None, help="Subset of latent dims.")
    p.add_argument(
        "--probe-classes", nargs="+", default=None, help="Subset of protocol probe classes."
    )
    p.add_argument("--splits", nargs="+", default=None, help="Subset of protocol splits.")
    p.add_argument("--horizons", nargs="+", type=int, default=None)
    p.add_argument("--n-boot-enc", type=int, default=None, help="Encounter bootstrap draws.")
    p.add_argument("--n-boot-case", type=int, default=None, help="Case-clustered draws.")
    p.add_argument("--cv-seed", type=int, default=0, help="Case-fold construction seed.")
    p.add_argument("--boot-seed", type=int, default=0, help="Per-cell bootstrap seed.")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    """Merge the frozen protocol with CLI overrides into a Config."""
    proto = load_protocol(args.protocol)
    observables = list(proto["observables"]["list"])
    probe_classes = args.probe_classes or list(proto["probes"]["classes"])
    splits = args.splits or list(proto["data"]["splits_reported"])
    horizons = args.horizons or [int(h) for h in proto["rollout"]["horizons_reported"]]
    n_boot_enc = args.n_boot_enc or int(proto["uncertainty"]["bootstrap_n"])
    unknown = set(probe_classes) - set(proto["probes"]["classes"])
    if unknown:
        raise SystemExit(f"unknown probe classes {sorted(unknown)} (not in the protocol)")
    return Config(
        families_manifest=args.families_manifest,
        protocol_path=args.protocol,
        dns_metrics=args.dns_metrics,
        latents_root=args.latents_root,
        rollouts_root=args.rollouts_root,
        split_manifest=args.split_manifest,
        out_dir=args.out_dir,
        numbers_part=args.numbers_part,
        observables=observables,
        probe_classes=probe_classes,
        splits=splits,
        horizons=horizons,
        test_b_tiers=list(proto["data"]["test_b_tiers"]),
        primary_horizon=int(proto["rollout"]["primary_horizon"]),
        primary_observable=str(proto["observables"]["primary"]),
        n_boot_enc=n_boot_enc,
        n_boot_case=args.n_boot_case or stats_lib.N_BOOT_CASE,
        cv_seed=args.cv_seed,
        boot_seed=args.boot_seed,
        ridge_alpha=args.ridge_alpha,
        family_filter=args.families,
        dim_filter=args.dims,
    )


def main(argv: list[str] | None = None) -> None:
    """Compute, write, and summarise the closure matrix."""
    args = parse_args(argv)
    cfg = build_config(args)
    result = run_matrix(cfg)
    rows = result["rows"]
    if not rows:
        print("[closure] no cells computed (no artifacts on disk yet); nothing written")
        return

    write_matrix(rows, cfg.out_dir)

    members = load_families(
        cfg.families_manifest,
        cfg.latents_root,
        cfg.rollouts_root,
        cfg.family_filter,
        cfg.dim_filter,
    )
    emit_headline_part(rows, members, cfg, cfg.numbers_part)

    verdict = gate_gd_verdict(rows, cfg)
    print_gate(verdict)
    gate_path = cfg.out_dir / "gate_gd.json"
    gate_path.write_text(json.dumps(verdict, indent=2))
    print(f"[closure] wrote {gate_path}")

    meta = {
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "protocol_file": str(cfg.protocol_path),
        "protocol_sha256": sha256_file(cfg.protocol_path),
        "families_manifest": str(cfg.families_manifest),
        "families_manifest_sha256": sha256_file(cfg.families_manifest),
        "dns_metrics": str(cfg.dns_metrics),
        "split_manifest": str(cfg.split_manifest),
        "n_rows": len(rows),
        "members_evaluated": result["members"],
        "skips": result["skips"],
        "config": {
            "observables": cfg.observables,
            "probe_classes": cfg.probe_classes,
            "splits": cfg.splits,
            "horizons": cfg.horizons,
            "n_boot_enc": cfg.n_boot_enc,
            "n_boot_case": cfg.n_boot_case,
            "n_folds": cfg.n_folds,
            "cv_seed": cfg.cv_seed,
            "boot_seed": cfg.boot_seed,
            "ridge_alpha": cfg.ridge_alpha,
        },
    }
    (cfg.out_dir / "matrix_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[closure] wrote {cfg.out_dir / 'matrix_meta.json'}")


if __name__ == "__main__":
    main()

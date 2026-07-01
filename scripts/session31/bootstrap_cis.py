"""Case-clustered bootstrap CIs for the Session 31 ROM headline cells.

The q*.json payloads hold only AGGREGATE metrics; a case-clustered CI (the gate-F
requirement) needs each metric recomputed per Test B CASE. We reuse the EXACT Q1/Q3
pipelines on the cached latents/fields/pressure (``outputs/session31/q1_latents``),
fit each cheap estimator ONCE on TRAIN, dump the per-row Test B contributions with
their ``case_id``, and bootstrap by resampling the 10 Test B cases with replacement
(:mod:`src.evaluation.report_session31`). Results accrete into
``outputs/session31/ci_dump.json`` keyed by the SAME headline number names that
``assemble_report.py`` uses.

Two paths:

* SCALAR (default, CPU): Q1 representation probes (linear + MLP R^2 per model,target)
  and Q3 pressure -> observable R^2 (per model,target + mean-obs). These recompute
  from the cached GAP latents / pressure with no GPU.
* FIELD (``--with-field``, GPU): Q1 decode-floor VRMSE + SSIM and Q3 pressure -> field
  VRMSE, each needing a one-time SpatialLatentFieldDecoder fit on the cached TRAIN
  latents (the encoder is already applied, so no encoder forward runs). Fit once per
  model, decode Test B once, dump per-row num/den + SSIM, then bootstrap.

Confine to a core subset per the shared-box rule, e.g.::

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 0-7 \\
        python scripts/session31/bootstrap_cis.py            # scalar (CPU)
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 0-7 \\
        python scripts/session31/bootstrap_cis.py --with-field --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.report_session31 import (  # noqa: E402
    CANONICAL_MODELS,
    TARGETS,
    agg_mean,
    agg_r2,
    agg_vrmse,
    bootstrap_ci,
    case_clustered_bootstrap,
    name_and_macro,
    percentile_ci,
    r2_contribs,
    three_curve_field_contribs,
    vrmse_contribs,
)

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "outputs" / "session31" / "q1_latents"
CI_DUMP = REPO / "outputs" / "session31" / "ci_dump.json"
Q2_TEMPORAL = REPO / "outputs" / "session31" / "q2_temporal.json"


def _load_ci_dump(path: Path) -> dict:
    if path.exists():
        blob = json.loads(path.read_text())
        return blob.get("numbers", blob)
    return {}


def _agg_mean_r2(c, targets):
    """Mean over targets of the aggregated R^2 (each 1 - sum ss_res / sum ss_tot)."""
    vals = []
    for t in targets:
        sst = float(c[f"ss_tot_{t}"].sum())
        sse = float(c[f"ss_res_{t}"].sum())
        vals.append(1.0 - sse / sst if sst > 0 else np.nan)
    return float(np.nanmean(vals))


# --------------------------------------------------------------------------- scalar (CPU)
def scalar_cis(models, *, n_boot: int, seed: int, verbose: bool = True) -> dict:
    """Q1 probe + Q3 pressure->obs case-clustered CIs from the cached GAP latents."""
    from src.evaluation.pressure_infer import (
        _row_groups,
        build_pressure_cache,
        fit_pressure_estimator,
        load_gap_split,
    )
    from src.evaluation.represent import fit_linear_probe, fit_mlp_probe

    pressure = build_pressure_cache(CACHE, partition="v2p2", verbose=verbose)
    p_tr, p_tb = pressure["train"], pressure["test_b"]

    out: dict[str, dict] = {}
    for m in models:
        gap_tr = load_gap_split(CACHE / f"latents_{m}_train.npz")
        gap_tb = load_gap_split(CACHE / f"latents_{m}_test_b.npz")
        mask = gap_tb["window_mask"]
        case_w = gap_tb["case_id"][mask]
        z_tr, z_tb = gap_tr["z_gap"], gap_tb["z_gap"]

        # --- Q1 probes: fit on all train, predict windowed test_b ---
        for t in TARGETS:
            y_tr, y_tb = gap_tr["targets"][t], gap_tb["targets"][t]
            y_w = y_tb[mask]
            lin = fit_linear_probe(z_tr, y_tr)
            mlp = fit_mlp_probe(z_tr, y_tr, seed=seed)
            for kind, model in (("repr_lin", lin), ("repr_mlp", mlp)):
                yhat = model.predict(z_tb[mask])
                c = r2_contribs(y_w, yhat)
                rec = bootstrap_ci(case_w, c, agg_r2, n_boot=n_boot, seed=seed)
                name, _ = name_and_macro(kind, m, target=t)
                rec["recompute_value"] = rec.pop("value")
                out[name] = rec

        # --- Q3 pressure -> observable (linear) ---
        groups = _row_groups(gap_tr["case_id"], gap_tr["encounter_index"])
        est = fit_pressure_estimator(p_tr, z_tr, n_components=400, seed=seed, groups=groups)
        z_tb_from_p = est.predict(p_tb)[mask]

        # --- Q3 pressure -> GAP latent aggregated multi-output R^2 (per-row over dims) ---
        z_true_w = z_tb[mask]
        mean_col = z_true_w.mean(axis=0, keepdims=True)
        lat_contribs = {
            "ss_res": ((z_tb_from_p - z_true_w) ** 2).sum(axis=1),
            "ss_tot": ((z_true_w - mean_col) ** 2).sum(axis=1),
        }
        rec = bootstrap_ci(case_w, lat_contribs, agg_r2, n_boot=n_boot, seed=seed)
        rec["recompute_value"] = rec.pop("value")
        out[name_and_macro("pres_latent_r2", m)[0]] = rec

        contribs_all: dict[str, np.ndarray] = {}
        for t in TARGETS:
            plin = fit_linear_probe(z_tr, gap_tr["targets"][t])
            y_w = gap_tb["targets"][t][mask]
            yhat = plin.predict(z_tb_from_p)
            c = r2_contribs(y_w, yhat)
            rec = bootstrap_ci(case_w, c, agg_r2, n_boot=n_boot, seed=seed)
            name, _ = name_and_macro("pres_obs", m, target=t)
            rec["recompute_value"] = rec.pop("value")
            out[name] = rec
            contribs_all[f"ss_res_{t}"] = c["ss_res"]
            contribs_all[f"ss_tot_{t}"] = c["ss_tot"]
        # mean-obs pressure R^2 (mean of the five aggregated R^2 per resample)
        agg = partial(_agg_mean_r2, targets=TARGETS)
        point = agg(contribs_all)
        samples = case_clustered_bootstrap(case_w, contribs_all, agg, n_boot=n_boot, seed=seed)
        lo, hi = percentile_ci(samples)
        name, _ = name_and_macro("pres_obs_mean", m)
        out[name] = {
            "recompute_value": point,
            "ci_low": lo,
            "ci_high": hi,
            "n": int(case_w.shape[0]),
        }

        if verbose:
            cl = out[name_and_macro("repr_lin", m, target="C_L")[0]]
            print(
                f"[ci-scalar] {m}: C_L lin R2 "
                f"{cl['recompute_value']:.3f} [{cl['ci_low']:.3f},{cl['ci_high']:.3f}] | "
                f"pres mean-obs {point:.3f} [{lo:.3f},{hi:.3f}]",
                flush=True,
            )
        del gap_tr, gap_tb
    return out


# --------------------------------------------------------------------------- field (GPU)
def _ssim_per_row(pred, true, data_range):
    from skimage.metrics import structural_similarity

    return np.array(
        [
            float(structural_similarity(true[i], pred[i], data_range=float(data_range)))
            for i in range(pred.shape[0])
        ],
        dtype=np.float64,
    )


def field_cis(
    models,
    *,
    n_boot: int,
    seed: int,
    gpu: int,
    device_override,
    decoder_steps: int,
    verbose: bool = True,
) -> dict:
    """Q1 decode-floor + Q3 pressure->field CIs via a one-time decoder fit per model."""
    import torch

    from src.data.omega_pipeline import ssim_data_range
    from src.evaluation.pressure_infer import (
        _row_groups,
        build_pressure_cache,
        fit_pressure_estimator,
    )
    from src.evaluation.represent import decode_fields, fit_decode_floor_decoder
    from src.evaluation.rollout import load_field_cache, load_latent_split

    if device_override is not None:
        device = torch.device(device_override)
        gpu_name = device_override
    else:
        from src.utils.device import require_rtx6000

        device = require_rtx6000(gpu_index=gpu)
        gpu_name = torch.cuda.get_device_name(device.index)
        if "RTX" not in gpu_name or "6000" not in gpu_name:
            raise RuntimeError(f"hardware policy: gpu_name={gpu_name!r} is not an RTX 6000")

    ssim_L = float(ssim_data_range("outputs/data_pipeline/v2p2/manifest.json"))
    pressure = build_pressure_cache(CACHE, partition="v2p2", verbose=verbose)
    p_tr, p_tb = pressure["train"], pressure["test_b"]
    omega_tr, *_ = load_field_cache(CACHE / "fields_train.npz")
    omega_tb, *_ = load_field_cache(CACHE / "fields_test_b.npz")

    out: dict[str, dict] = {}
    print(f"[ci-field] device={device} ({gpu_name}) ssim_L={ssim_L:.3f}", flush=True)
    for m in models:
        tr = load_latent_split(CACHE / f"latents_{m}_train.npz")
        tb = load_latent_split(CACHE / f"latents_{m}_test_b.npz")
        grid = tr.latent_grid
        d = int(tr.z_spatial.shape[1])
        mask = tb.window_mask
        case_w = tb.case_id[mask]

        decoder = fit_decode_floor_decoder(
            tr.z_spatial,
            omega_tr,
            grid,
            device=device,
            steps=decoder_steps,
            batch=64,
            seed=seed,
            verbose=verbose,
        )
        # --- decode floor: decode the TRUE spatial latent ---
        floor_pred = decode_fields(decoder, tb.z_spatial, device)[mask]
        true_w = np.asarray(omega_tb, dtype=np.float32)[mask]
        cv = vrmse_contribs(floor_pred, true_w)
        rec = bootstrap_ci(case_w, cv, agg_vrmse, n_boot=n_boot, seed=seed)
        name, _ = name_and_macro("repr_floor_vrmse", m)
        rec["recompute_value"] = rec.pop("value")
        out[name] = rec
        ssim_rows = _ssim_per_row(floor_pred, true_w, ssim_L)
        recs = bootstrap_ci(case_w, {"val": ssim_rows}, agg_mean, n_boot=n_boot, seed=seed)
        name, _ = name_and_macro("repr_floor_ssim", m)
        recs["recompute_value"] = recs.pop("value")
        out[name] = recs

        # --- Q3 pressure -> field: decode the pressure-estimated spatial latent ---
        groups = _row_groups(tr.case_id, tr.encounter_index)
        est = fit_pressure_estimator(
            p_tr,
            tr.z_spatial.reshape(tr.z_spatial.shape[0], -1),
            n_components=400,
            seed=seed,
            groups=groups,
        )
        n_tb = tb.z_spatial.shape[0]
        z_pred = np.asarray(est.predict(p_tb), dtype=np.float32).reshape(n_tb, d, grid[0], grid[1])
        pres_pred = decode_fields(decoder, z_pred, device)[mask]
        cvp = vrmse_contribs(pres_pred, true_w)
        recp = bootstrap_ci(case_w, cvp, agg_vrmse, n_boot=n_boot, seed=seed)
        name, _ = name_and_macro("pres_field_vrmse", m)
        recp["recompute_value"] = recp.pop("value")
        out[name] = recp

        if verbose:
            fv = out[name_and_macro("repr_floor_vrmse", m)[0]]
            print(
                f"[ci-field] {m}: floor VRMSE {fv['recompute_value']:.3f} "
                f"[{fv['ci_low']:.3f},{fv['ci_high']:.3f}] | pres field VRMSE "
                f"{recp['recompute_value']:.3f} [{recp['ci_low']:.3f},{recp['ci_high']:.3f}]",
                flush=True,
            )
        del decoder, est, tr, tb
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def closure_cis(
    models,
    *,
    horizon: int,
    n_boot: int,
    seed: int,
    gpu: int,
    device_override,
    predictor_steps: int,
    verbose: bool = True,
) -> dict:
    """Q2 observable-closure CIs at one horizon via a fresh matched-ResUNet rollout.

    Fits the matched predictor on TRAIN latents (the Q2 matched-predictor protocol),
    rolls every in-window Test B anchor to ``horizon``, reads the five observables
    from the rolled GAP latent through the frozen Q1 MLP probes, and bootstraps the
    per-observable R^2 and the mean-obs ROM figure of merit over Test B cases. The
    rolled latent is not cached, so this is a one-time GPU rollout (no field decode).
    """
    import torch

    from src.evaluation.represent import fit_mlp_probe
    from src.evaluation.rollout import (
        _gather_spatial,
        fit_matched_resunet,
        group_encounters,
        load_latent_split,
    )

    if device_override is not None:
        device = torch.device(device_override)
        gpu_name = device_override
    else:
        from src.utils.device import require_rtx6000

        device = require_rtx6000(gpu_index=gpu)
        gpu_name = torch.cuda.get_device_name(device.index)
        if "RTX" not in gpu_name or "6000" not in gpu_name:
            raise RuntimeError(f"hardware policy: gpu_name={gpu_name!r} is not an RTX 6000")

    context_length = 2  # Q2 params: ResUNetPredictor(context_length=2)
    print(f"[ci-closure] device={device} ({gpu_name}) horizon={horizon}", flush=True)
    out: dict[str, dict] = {}
    for m in models:
        tr = load_latent_split(CACHE / f"latents_{m}_train.npz")
        tb = load_latent_split(CACHE / f"latents_{m}_test_b.npz")
        trajs = [enc.z_spatial for enc in group_encounters(tr)]
        predictor = fit_matched_resunet(
            trajs,
            device=device,
            horizon_train=8,
            context_length=context_length,
            steps=predictor_steps,
            seed=seed,
            verbose=verbose,
        )
        buckets = _gather_spatial(
            predictor, group_encounters(tb), [horizon], context_length, device
        )
        b = buckets[int(horizon)]
        rolled_gap = np.concatenate(b.rolled_gap)  # (A, d)
        target_row = np.concatenate(b.target_row)  # (A,) rows into the flat cache
        case_w = tb.case_id[target_row]

        contribs_all: dict[str, np.ndarray] = {}
        for t in TARGETS:
            mlp = fit_mlp_probe(tr.z_gap, tr.targets[t], seed=seed)
            y_true = tb.targets[t][target_row]
            y_pred = mlp.predict(rolled_gap)
            c = r2_contribs(y_true, y_pred)
            rec = bootstrap_ci(case_w, c, agg_r2, n_boot=n_boot, seed=seed)
            rec["recompute_value"] = rec.pop("value")
            out[name_and_macro("fore_clos", m, target=t, h=horizon)[0]] = rec
            contribs_all[f"ss_res_{t}"] = c["ss_res"]
            contribs_all[f"ss_tot_{t}"] = c["ss_tot"]
        agg = partial(_agg_mean_r2, targets=TARGETS)
        point = agg(contribs_all)
        samples = case_clustered_bootstrap(case_w, contribs_all, agg, n_boot=n_boot, seed=seed)
        lo, hi = percentile_ci(samples)
        out[name_and_macro("fore_merit", m, h=horizon)[0]] = {
            "recompute_value": point,
            "ci_low": lo,
            "ci_high": hi,
            "n": int(case_w.shape[0]),
        }
        if verbose:
            print(
                f"[ci-closure] {m}: mean-obs MLP R2 h{horizon} "
                f"{point:.3f} [{lo:.3f},{hi:.3f}] (n={case_w.shape[0]})",
                flush=True,
            )
        del predictor, tr, tb, buckets
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _reported_field_vrmse(model: str, horizon: int) -> dict[str, float]:
    """run_q2 reported field-VRMSE (model/floor/persistence) at ``horizon`` for ``model``."""
    q2 = json.loads(Q2_TEMPORAL.read_text())
    fv = q2["models"][model]["predictors"]["resunet_matched"]["field_vrmse"]
    i = list(fv["horizons"]).index(int(horizon))
    return {which: float(fv[which][i]) for which in ("model", "floor", "persistence")}


def field_forecast_cis(
    models,
    *,
    horizon: int = 8,
    n_boot: int,
    seed: int,
    gpu: int,
    device_override,
    predictor_steps: int,
    decoder_steps: int,
    match_tol: float = 0.005,
    verbose: bool = True,
) -> dict:
    """Q2 FIELD-VRMSE three-curve (model/floor/persistence) CIs at one horizon.

    Mirrors run_q2's field-VRMSE recompute EXACTLY: per model, fit the matched
    ResUNet predictor and the decode-floor decoder on TRAIN latents, roll every
    in-window Test B anchor over ``1..DEFAULT_HORIZON`` (so ``_needed_anchors`` uses
    H=16 and the ``horizon`` sample set matches run_q2 frame-for-frame), read the
    ``horizon`` bucket, then form the three curves: ``model = decode(rolled z)``,
    ``floor = decode(true z_{t+h})``, ``persistence = held anchor DNS field``, each
    scored against the DNS field at ``t + h``. The aggregated VRMSE is a
    ratio-of-sums, so the bootstrap resamples per-CASE (num, den) contributions and
    recomputes ``sqrt(sum num / sum den)`` per resample.

    Before bootstrapping, the recomputed point VRMSE per curve is asserted against
    the run_q2 reported value within ``match_tol`` (bf16 decode/rollout
    nondeterminism); a gross mismatch (the mis-indexed-horizon bug) raises so no CI
    that fails to bracket the reported point is ever emitted.
    """
    import torch

    from src.evaluation.represent import decode_fields, fit_decode_floor_decoder
    from src.evaluation.rollout import (
        DEFAULT_HORIZON,
        _gather_spatial,
        fit_matched_resunet,
        group_encounters,
        load_field_cache,
        load_latent_split,
    )

    if device_override is not None:
        device = torch.device(device_override)
        gpu_name = device_override
    else:
        from src.utils.device import require_rtx6000

        device = require_rtx6000(gpu_index=gpu)
        gpu_name = torch.cuda.get_device_name(device.index)
        if "RTX" not in gpu_name or "6000" not in gpu_name:
            raise RuntimeError(f"hardware policy: gpu_name={gpu_name!r} is not an RTX 6000")

    context_length = 2  # Q2 params: ResUNetPredictor(context_length=2)
    horizons = list(range(1, DEFAULT_HORIZON + 1))  # H=16 so the h-bucket matches run_q2
    omega_tr, *_ = load_field_cache(CACHE / "fields_train.npz")
    omega_tb, *_ = load_field_cache(CACHE / "fields_test_b.npz")
    omega_tb = np.asarray(omega_tb, dtype=np.float32)

    print(
        f"[ci-field-fore] device={device} ({gpu_name}) horizon={horizon} "
        f"rollout_H={DEFAULT_HORIZON}",
        flush=True,
    )
    out: dict[str, dict] = {}
    for m in models:
        tr = load_latent_split(CACHE / f"latents_{m}_train.npz")
        tb = load_latent_split(CACHE / f"latents_{m}_test_b.npz")
        grid = tr.latent_grid
        trajs = [enc.z_spatial for enc in group_encounters(tr)]

        decoder = fit_decode_floor_decoder(
            tr.z_spatial,
            omega_tr,
            grid,
            device=device,
            steps=decoder_steps,
            batch=64,
            seed=seed,
            verbose=verbose,
        )
        predictor = fit_matched_resunet(
            trajs,
            device=device,
            horizon_train=8,
            context_length=context_length,
            steps=predictor_steps,
            seed=seed,
            verbose=verbose,
        )
        buckets = _gather_spatial(predictor, group_encounters(tb), horizons, context_length, device)
        b = buckets[int(horizon)]
        rolled_sp = np.concatenate(b.rolled_sp).astype(np.float32)
        true_sp = np.concatenate(b.true_sp).astype(np.float32)
        target_row = np.concatenate(b.target_row)
        anchor_row = np.concatenate(b.anchor_row)
        case_w = tb.case_id[target_row]

        model_field = decode_fields(decoder, rolled_sp, device)
        floor_field = decode_fields(decoder, true_sp, device)
        true_field = omega_tb[target_row]
        persist_field = omega_tb[anchor_row]

        curves = three_curve_field_contribs(model_field, floor_field, persist_field, true_field)
        reported = _reported_field_vrmse(m, horizon)
        # MANDATORY: recompute must match run_q2 reported before any CI is emitted.
        deltas = {which: abs(agg_vrmse(c) - reported[which]) for which, c in curves.items()}
        worst = max(deltas.values())
        if verbose:
            print(
                f"[ci-field-fore] {m} h{horizon} recompute-vs-reported deltas "
                + " ".join(f"{w}={deltas[w]:.5f}" for w in ("model", "floor", "persistence"))
                + f" (max {worst:.5f})",
                flush=True,
            )
        if worst > match_tol:
            raise RuntimeError(
                f"field-VRMSE recompute for {m} at h{horizon} disagrees with run_q2 "
                f"reported by {worst:.5f} > tol {match_tol} (deltas {deltas}). "
                f"STOP: not emitting CIs that may not bracket the reported point."
            )

        for which, c in curves.items():
            rec = bootstrap_ci(case_w, c, agg_vrmse, n_boot=n_boot, seed=seed)
            rec["recompute_value"] = rec.pop("value")
            out[name_and_macro("fore_field", m, which=which, h=horizon)[0]] = rec

        if verbose:
            for which in ("model", "floor", "persistence"):
                rec = out[name_and_macro("fore_field", m, which=which, h=horizon)[0]]
                print(
                    f"[ci-field-fore] {m}: {which} VRMSE h{horizon} "
                    f"{rec['recompute_value']:.3f} [{rec['ci_low']:.3f},{rec['ci_high']:.3f}] "
                    f"(reported {reported[which]:.3f}, n={rec['n']})",
                    flush=True,
                )
        del decoder, predictor, buckets, tr, tb
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(CANONICAL_MODELS))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with-field", action="store_true", help="Also compute GPU field CIs.")
    p.add_argument("--field-only", action="store_true", help="Skip scalar; field CIs only.")
    p.add_argument(
        "--with-closure",
        action="store_true",
        help="Also compute GPU Q2 observable-closure CIs (matched-predictor rollout).",
    )
    p.add_argument("--closure-only", action="store_true", help="Skip scalar; closure CIs only.")
    p.add_argument("--closure-horizon", type=int, default=8)
    p.add_argument(
        "--with-field-forecast",
        action="store_true",
        help="Also compute GPU Q2 field-VRMSE three-curve CIs (matched-predictor rollout).",
    )
    p.add_argument(
        "--field-forecast-only",
        action="store_true",
        help="Skip scalar; field-VRMSE forecast CIs only.",
    )
    p.add_argument("--field-forecast-horizon", type=int, default=8)
    p.add_argument("--predictor-steps", type=int, default=4000)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None, help="Explicit device (bypasses require_rtx6000).")
    p.add_argument("--decoder-steps", type=int, default=6000)
    p.add_argument("--out", default=str(CI_DUMP))
    args = p.parse_args(argv)

    numbers = _load_ci_dump(Path(args.out))
    scalar_skipped = args.field_only or args.closure_only or args.field_forecast_only
    if not scalar_skipped:
        numbers.update(scalar_cis(args.models, n_boot=args.n_boot, seed=args.seed))
    if args.with_field or args.field_only:
        numbers.update(
            field_cis(
                args.models,
                n_boot=args.n_boot,
                seed=args.seed,
                gpu=args.gpu,
                device_override=args.device,
                decoder_steps=args.decoder_steps,
            )
        )
    if args.with_closure or args.closure_only:
        numbers.update(
            closure_cis(
                args.models,
                horizon=args.closure_horizon,
                n_boot=args.n_boot,
                seed=args.seed,
                gpu=args.gpu,
                device_override=args.device,
                predictor_steps=args.predictor_steps,
            )
        )
    if args.with_field_forecast or args.field_forecast_only:
        numbers.update(
            field_forecast_cis(
                args.models,
                horizon=args.field_forecast_horizon,
                n_boot=args.n_boot,
                seed=args.seed,
                gpu=args.gpu,
                device_override=args.device,
                predictor_steps=args.predictor_steps,
                decoder_steps=args.decoder_steps,
            )
        )
    blob = {"n_boot": args.n_boot, "seed": args.seed, "numbers": numbers}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(blob, indent=2, sort_keys=True))
    print(f"[ci] wrote {args.out} ({len(numbers)} CI records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

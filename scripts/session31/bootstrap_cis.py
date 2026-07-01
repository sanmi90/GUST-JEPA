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
    vrmse_contribs,
)

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "outputs" / "session31" / "q1_latents"
CI_DUMP = REPO / "outputs" / "session31" / "ci_dump.json"


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(CANONICAL_MODELS))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with-field", action="store_true", help="Also compute GPU field CIs.")
    p.add_argument("--field-only", action="store_true", help="Skip scalar; field CIs only.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None, help="Explicit device (bypasses require_rtx6000).")
    p.add_argument("--decoder-steps", type=int, default=6000)
    p.add_argument("--out", default=str(CI_DUMP))
    args = p.parse_args(argv)

    numbers = _load_ci_dump(Path(args.out))
    if not args.field_only:
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
    blob = {"n_boot": args.n_boot, "seed": args.seed, "numbers": numbers}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(blob, indent=2, sort_keys=True))
    print(f"[ci] wrote {args.out} ({len(numbers)} CI records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

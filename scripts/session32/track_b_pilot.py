"""Session 32 Track B pilot: predict-correct filter smoke on test_a.

Runs the EnKF (:mod:`src.estimation`) end-to-end on a few ``test_a`` encounters for
``jepa_pool`` (plus a ``POD`` sanity row), with the FIELD-FREE init and a single
inflation value ``rho=1.02``. This is a SMOKE: it confirms the pipeline runs and
produces sane numbers (innovation whiteness + analysis-vs-open-loop closure on E_w
and C_L). It does NOT freeze inflation or warning thresholds (deferred), and it
never touches Test B/C.

State  = frozen pooled latent (d=32), advanced by a fresh MATCHED AR-transformer
         predictor fit on TRAIN pooled latents (per family).
Measure = K=8 qDEIM wall-pressure taps through the frozen H (pressure-only).
Init    = O1-style windowed p_wall -> state estimator on TRAIN (field-free).
Truth observables (E_w, C_L) are read from prior/analysis STATES via the frozen Q1
probes as eval-only readouts; they never enter the innovation.

Run (RTX 6000 + CPU cap):
    taskset -c 24-31 python -m scripts.session32.track_b_pilot --gpu 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.estimation import enkf as ek  # noqa: E402
from src.estimation import metrics as mt  # noqa: E402
from src.estimation.obs_operator import (  # noqa: E402
    fit_observation_operator,
    load_or_compute_taps,
    load_osp_taps,
)
from src.evaluation.pressure_infer import (  # noqa: E402
    fit_pressure_estimator,
    load_pressure_for_alignment,
)
from src.evaluation.represent import fit_linear_probe  # noqa: E402
from src.evaluation.rollout import fit_matched_transformer, frame_phase_ids  # noqa: E402
from src.evaluation.rom_eval import (  # noqa: E402
    load_frozen_model,
    load_reference_model,
    physical_observables,
)

DEFAULT_CASES = (
    "G+1.00_D0.50_Y+0.10",
    "G+2.00_D1.00_Y+0.10",
    "G+1.00_D1.00_Y+0.10",
)
INIT_WINDOW = 12  # frames of pre-impact pressure for the field-free initialiser
PRE = 24  # assimilate from t_imp - PRE
POST = 48  # ... to t_imp + POST


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------- data
def _cache_dir(partition: str) -> Path:
    import os

    env = os.environ.get("VORTEX_JEPA_CACHE")
    if env:
        return Path(env) / partition
    prevent = os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT"))
    return Path(prevent) / "data" / "processed" / "vortex-jepa" / partition


def windowed_features(p: np.ndarray, W: int) -> np.ndarray:
    """Per-frame flattened window of the last ``W`` pressure frames (edge-padded)."""
    p = np.asarray(p, dtype=np.float32)
    T, S = p.shape
    out = np.empty((T, W * S), dtype=np.float32)
    for t in range(T):
        lo = t - W + 1
        if lo < 0:
            pad = np.repeat(p[0:1], -lo, axis=0)
            win = np.concatenate([pad, p[0 : t + 1]], axis=0)
        else:
            win = p[lo : t + 1]
        out[t] = win.reshape(-1)
    return out


@np.errstate(all="ignore")
def encode_encounters(frozen, encounters, pipe, cache_dir, windows, device):
    """Encode a small list of (case, k) into pooled-state + pressure + truth obs."""
    from src.evaluation.rom_eval import _encode_frames

    out = []
    pooled = getattr(frozen.encoder, "latent_mode", "spatial") == "pooled"
    for cid, k in encounters:
        path = cache_dir / cid / f"encounter_{int(k):02d}.h5"
        with h5py.File(path, "r") as g:
            omega_raw = np.asarray(g["omega_z"], dtype=np.float32)
            cl = np.asarray(g["C_L"], dtype=np.float64)
            p_wall = np.asarray(g["p_wall"], dtype=np.float32)
        n_frames = omega_raw.shape[0]
        omega_clean = pipe.preprocess_raw(omega_raw, cid, int(k))
        omega_norm = np.asarray(pipe.normalize(omega_clean), dtype=np.float32)
        z = _encode_frames(frozen.encoder, omega_norm, device, 40, True)  # (T,d) or (T,d,h,w)
        if not pooled:
            z = z.mean(axis=(2, 3))
        z_gap = np.asarray(z, dtype=np.float32)  # (T, d)
        phys = physical_observables(omega_clean)
        entry = windows.get(f"{cid}/{int(k):02d}")
        t_imp = int(entry["impact"][0]) if entry else int(np.argmax(np.abs(cl)))
        out.append(
            {
                "case_id": cid,
                "encounter_index": int(k),
                "z_gap": z_gap,
                "p_wall": p_wall,
                "C_L": cl,
                "wake_enstrophy": phys["wake_enstrophy"],
                "t_impact": t_imp,
                "n_frames": n_frames,
                "phase_ids": frame_phase_ids(entry, n_frames),
            }
        )
    return out


def group_gap_trajs(z_gap, case_id, encounter_index, frame):
    keys = np.array([f"{c}/{int(e):02d}" for c, e in zip(case_id.astype(str), encounter_index)])
    trajs = []
    for key in sorted(np.unique(keys)):
        rows = np.where(keys == key)[0]
        rows = rows[np.argsort(frame[rows])]
        trajs.append(z_gap[rows].astype(np.float32))
    return trajs


def q_from_teacher_forcing(predictor, gap_trajs, device, *, n_samples=256, length=32, seed=0):
    """One-step predictor residual covariance Q (D222) via teacher-forced forward."""
    import torch

    rng = np.random.default_rng(seed)
    valid = [t for t in gap_trajs if t.shape[0] > length]
    d = gap_trajs[0].shape[1]
    resids = []
    predictor.eval()
    for _ in range(n_samples):
        z = valid[int(rng.integers(0, len(valid)))]
        s = int(rng.integers(0, z.shape[0] - length))
        sub = z[s : s + length][None, :, :]
        zb = torch.from_numpy(np.ascontiguousarray(sub)).to(device)
        with torch.no_grad():
            if getattr(device, "type", "cpu") == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    zh = predictor(zb, None)
            else:
                zh = predictor(zb, None)
        zh = zh.float().cpu().numpy()[0]  # (length, d)
        # z_hat[t] predicts z[t+1]; residual over t = 0..length-2
        resids.append(sub[0, 1:] - zh[:-1])
    R = np.concatenate(resids, axis=0)  # (M, d)
    Q = np.cov(R, rowvar=False) + 1e-6 * np.eye(d)
    return Q


# --------------------------------------------------------------------------- context
def build_context(name, run_dir, is_reference, device, args, train_cache_tmpl=None):
    train_cache_tmpl = train_cache_tmpl or args.train_cache
    cache = np.load(_resolve(train_cache_tmpl % name), allow_pickle=True)
    z_gap_tr = np.asarray(cache["z_gap"], dtype=np.float32)
    cid_tr = np.asarray(cache["case_id"]).astype(str)
    enc_tr = np.asarray(cache["encounter_index"])
    frame_tr = np.asarray(cache["frame"])
    y_cl = np.asarray(cache["target_C_L"], dtype=np.float64)
    y_ew = np.asarray(cache["target_wake_enstrophy"], dtype=np.float64)

    # TRAIN wall pressure aligned to the train rows.
    p_tr = load_pressure_for_alignment(cid_tr, enc_tr, frame_tr, partition=args.partition)

    # matched AR-transformer predictor on the pooled TRAIN latent.
    gap_trajs = group_gap_trajs(z_gap_tr, cid_tr, enc_tr, frame_tr)
    fake_spatial = [t[:, :, None, None] for t in gap_trajs]  # (L,d,1,1) -> GAP=pooled
    predictor = fit_matched_transformer(
        fake_spatial,
        device=device,
        latent_dim=z_gap_tr.shape[1],
        steps=args.predictor_steps,
        seed=args.seed,
        verbose=False,
    )
    fmodel = ek.TransformerForecast(predictor, device)

    # Observation taps. Track O reworked to MODEL-CONDITIONED OSP (per-model TCSI
    # taps, HANDOFF D232): each model's H senses at THAT model's own OSP K8 taps.
    #   osp_per_model (primary): osp[<model>]["K8"]
    #   qdeim_shared  (baseline): osp["qdeim_shared"]["K8"] (old target-blind array)
    if args.taps_mode == "osp_per_model":
        taps, tap_prov = load_osp_taps(name, k=args.k, osp_path=args.osp_taps)
    elif args.taps_mode == "qdeim_shared":
        taps, tap_prov = load_osp_taps("qdeim_shared", k=args.k, osp_path=args.osp_taps)
    else:  # legacy self-contained fallback
        taps, tap_prov = load_or_compute_taps(k=args.k, taps_path=args.taps, p_train=p_tr)
    # encounter-grouped held-out split for the R (obs-noise) estimate.
    groups = np.array([f"{c}/{int(e)}" for c, e in zip(cid_tr, enc_tr)])
    ug = np.unique(groups)
    rng = np.random.default_rng(args.seed)
    val_g = set(rng.choice(ug, size=max(1, len(ug) // 5), replace=False).tolist())
    vmask = np.array([g in val_g for g in groups])
    H = fit_observation_operator(
        z_gap_tr[~vmask],
        p_tr[~vmask],
        taps,
        kind="linear",
        z_val=z_gap_tr[vmask],
        p_val=p_tr[vmask],
        provenance=tap_prov,
    )

    # frozen Q1 observable probes (eval-only readouts; NEVER an innovation).
    probe_cl = fit_linear_probe(z_gap_tr, y_cl)
    probe_ew = fit_linear_probe(z_gap_tr, y_ew)

    # process noise Q (D222).
    Q = q_from_teacher_forcing(predictor, gap_trajs, device, seed=args.seed)

    # field-free init estimator: windowed p_wall -> pooled state.
    # subsample train frames (grouped) for the windowed Nystroem fit.
    sub = rng.choice(len(z_gap_tr), size=min(9000, len(z_gap_tr)), replace=False)
    win_tr = windowed_features(p_tr, INIT_WINDOW)  # (N, W*S)
    est = fit_pressure_estimator(
        win_tr[sub], z_gap_tr[sub], n_components=300, seed=args.seed, groups=groups[sub]
    )
    resid = z_gap_tr[sub] - est.predict(win_tr[sub])
    init_cov = np.cov(resid, rowvar=False) + 1e-6 * np.eye(z_gap_tr.shape[1])
    init = ek.FieldFreeInit(mean_fn=est.predict, cov=init_cov)

    return {
        "name": name,
        "fmodel": fmodel,
        "H": H,
        "Q": Q,
        "init": init,
        "probe_cl": probe_cl,
        "probe_ew": probe_ew,
        "taps": (taps, tap_prov),
    }


# --------------------------------------------------------------------------- run one
def run_encounter(ctx, enc, args, rho=None):
    rho = args.rho if rho is None else float(rho)
    H = ctx["H"]
    d = int(enc["z_gap"].shape[1])
    t_imp = enc["t_impact"]
    n_frames = enc["n_frames"]
    f0 = max(0, t_imp - PRE)
    f1 = min(n_frames - 1, t_imp + POST)
    frames = np.arange(f0, f1 + 1)
    T = len(frames)

    # measurements: K taps of span-averaged wall pressure at each assimilation frame.
    y_series = H.select_taps(enc["p_wall"][frames])  # (T, K)

    # field-free init at f0.
    rng = np.random.default_rng(args.seed + 1)
    win = windowed_features(enc["p_wall"], INIT_WINDOW)[f0]
    init_ens = ctx["init"].sample(win, args.n_members, rng)

    filt = ek.EnsembleKalmanFilter(
        ctx["fmodel"],
        H,
        ctx["Q"],
        n_members=args.n_members,
        inflation=rho,
        mode=args.mode,
        seed=args.seed + 2,
    )
    res = filt.run(y_series, init_ens, frames, meta={"case": enc["case_id"]})

    # open-loop free run (same forecast, no measurement update).
    ol = ek.EnsembleKalmanFilter(
        ctx["fmodel"],
        H,
        ctx["Q"],
        n_members=args.n_members,
        inflation=rho,
        mode=args.mode,
        seed=args.seed + 3,
    )
    ol_states = np.empty((T, args.n_members, d))
    ol_states[0] = init_ens
    buf = init_ens[:, None, :]
    for t in range(1, T):
        prior = ol.forecast(buf)
        ol_states[t] = prior
        buf = np.concatenate([buf, prior[:, None, :]], axis=1)
        if buf.shape[1] > filt.max_context:
            buf = buf[:, -filt.max_context :, :]

    # observable readouts from STATE means (frozen probes; eval-only).
    truth_cl = enc["C_L"][frames]
    truth_ew = enc["wake_enstrophy"][frames]
    phase_ids = enc["phase_ids"][frames]

    def readout(states_mean, probe):
        return np.asarray(probe.predict(states_mean), dtype=np.float64).reshape(-1)

    a_cl = readout(res.analysis_mean, ctx["probe_cl"])
    a_ew = readout(res.analysis_mean, ctx["probe_ew"])
    p_cl = readout(res.prior_mean, ctx["probe_cl"])
    p_ew = readout(res.prior_mean, ctx["probe_ew"])
    ol_cl = readout(ol_states.mean(axis=1), ctx["probe_cl"])
    ol_ew = readout(ol_states.mean(axis=1), ctx["probe_ew"])

    # per-frame ensemble readouts for CRPS / spread-skill.
    ens_cl = np.stack([ctx["probe_cl"].predict(res.analysis_ens[t]) for t in range(T)])
    ens_ew = np.stack([ctx["probe_ew"].predict(res.analysis_ens[t]) for t in range(T)])

    clo_a_cl = mt.closure_by_phase(a_cl, truth_cl, phase_ids)
    clo_a_ew = mt.closure_by_phase(a_ew, truth_ew, phase_ids)
    clo_ol_cl = mt.closure_by_phase(ol_cl, truth_cl, phase_ids)
    clo_ol_ew = mt.closure_by_phase(ol_ew, truth_ew, phase_ids)

    white = mt.lag1_autocorr(res.innovation)
    nis = mt.nis_coverage(res.nis, dof=H.output_dim)
    div = mt.divergence_flag(res.nis, dof=H.output_dim)

    return {
        "case_id": enc["case_id"],
        "encounter_index": enc["encounter_index"],
        "t_impact": int(t_imp),
        "frames": [int(f0), int(f1)],
        "n_assim": int(T),
        "closure": {
            "C_L": {
                "analysis": clo_a_cl,
                "prior": mt.closure_by_phase(p_cl, truth_cl, phase_ids),
                "open_loop": clo_ol_cl,
                "gain_vs_open_loop": mt.closure_gain(clo_a_cl, clo_ol_cl),
            },
            "wake_enstrophy": {
                "analysis": clo_a_ew,
                "prior": mt.closure_by_phase(p_ew, truth_ew, phase_ids),
                "open_loop": clo_ol_ew,
                "gain_vs_open_loop": mt.closure_gain(clo_a_ew, clo_ol_ew),
            },
        },
        "innovation_whiteness": white,
        "nis_coverage": nis,
        "divergence": div,
        "crps": {
            "C_L": mt.crps_ensemble(ens_cl, truth_cl),
            "wake_enstrophy": mt.crps_ensemble(ens_ew, truth_ew),
        },
        "spread_skill": {
            "C_L": mt.spread_skill(ens_cl, truth_cl),
            "wake_enstrophy": mt.spread_skill(ens_ew, truth_ew),
        },
    }


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Session 32 Track B EnKF pilot (test_a smoke)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None, help="explicit device (bypass require_rtx6000)")
    p.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    p.add_argument("--encounter", type=int, default=4, help="val encounter index to use")
    p.add_argument("--rho", type=float, default=1.02)
    p.add_argument("--mode", default="stochastic", choices=["stochastic", "sqrt"])
    p.add_argument("--n-members", type=int, default=64)
    p.add_argument("--k", type=int, default=8)
    p.add_argument(
        "--taps-mode",
        default="osp_per_model",
        choices=["osp_per_model", "qdeim_shared", "legacy"],
        help="osp_per_model (primary): each model's own OSP K8 taps; "
        "qdeim_shared: the target-blind shared array; legacy: self-contained qDEIM.",
    )
    p.add_argument("--osp-taps", default="outputs/session32/osp_taps_v2p2.json")
    p.add_argument("--taps", default="outputs/session32/qdeim_taps_v2p2.json")
    p.add_argument("--partition", default="v2p2")
    p.add_argument("--pipeline-manifest", default="outputs/data_pipeline/v2p2/manifest.json")
    p.add_argument("--split", default="configs/splits/split_v2p2.json")
    p.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    p.add_argument("--checkpoint", default="checkpoint_iter010000.pt")
    p.add_argument(
        "--train-cache",
        default="outputs/session31/q1_latents_ablation/latents_%s_train.npz",
        help="printf-style train latents path; %%s = model name",
    )
    p.add_argument("--jepa-run", default="outputs/runs/session31/jepa_pool")
    p.add_argument("--pod-run", default="outputs/runs/session31/pod")
    p.add_argument("--predictor-steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/session32/track_b_pilot.json")
    p.add_argument("--no-pod", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    import torch

    args = parse_args(argv)
    if args.device is not None:
        device = torch.device(args.device)
        gpu_name = args.device
    else:
        from src.utils.device import require_rtx6000

        device = require_rtx6000(gpu_index=args.gpu)
        gpu_name = torch.cuda.get_device_name(device.index)
        if "RTX" not in gpu_name or "6000" not in gpu_name:
            raise RuntimeError(f"hardware policy: gpu_name={gpu_name!r} is not an RTX 6000")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    windows = json.loads(_resolve(args.windows).read_text())["windows"]
    pipe = OmegaPipeline.from_manifest(_resolve(args.pipeline_manifest))
    encounters = [(c, args.encounter) for c in args.cases]

    print(
        f"[track-b] device={device} ({gpu_name}) cases={args.cases} rho={args.rho} "
        f"N={args.n_members} mode={args.mode}",
        flush=True,
    )

    payload = {
        "task": "SESSION 32 Track B -- predict-correct EnKF pilot (test_a smoke)",
        "params": {
            "gpu_name": gpu_name,
            "partition": args.partition,
            "inflation_rho": args.rho,
            "mode": args.mode,
            "n_members": args.n_members,
            "K": args.k,
            "taps_mode": args.taps_mode,
            "osp_taps_file": args.osp_taps,
            "init": "field_free_O1_windowed_pressure",
            "assim_window": {"pre": PRE, "post": POST, "cadence": "every_frame"},
            "seed": args.seed,
            "note": (
                "test_a pilot. State = pooled latent; measurement = K wall-pressure "
                "taps (pressure-only, per taps_mode); E_w / C_L are eval-only "
                "readouts, never innovations. Per-model OSP taps = HANDOFF D232."
            ),
        },
        "models": {},
    }

    abl = "outputs/session31/q1_latents_ablation/latents_%s_train.npz"
    canon = "outputs/session31/q1_latents/latents_%s_train.npz"
    model_specs = [("jepa_pool", args.jepa_run, False, abl)]
    if not args.no_pod:
        model_specs.append(("pod", args.pod_run, True, canon))

    for name, run_dir, is_ref, cache_tmpl in model_specs:
        print(f"\n[track-b] === building {name} ({args.taps_mode}) ===", flush=True)
        ctx = build_context(
            name,
            run_dir,
            is_ref,
            device,
            args,
            train_cache_tmpl=cache_tmpl,
        )
        frozen = (load_reference_model if is_ref else load_frozen_model)(
            _resolve(run_dir), args.checkpoint, device
        )
        encs = encode_encounters(
            frozen, encounters, pipe, _cache_dir(args.partition), windows, device
        )
        rows = [run_encounter(ctx, e, args) for e in encs]
        payload["models"][name] = {
            "taps": [int(x) for x in ctx["taps"][0]],
            "taps_provenance": ctx["taps"][1],
            "R_source": ctx["H"].provenance.get("R_source"),
            "encounters": rows,
        }
        _print_model(name, rows)
        out_path = _resolve(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=float))

    print(f"\n[track-b] wrote {_resolve(args.out)}", flush=True)
    return 0


def _phase(clo, phase, key="rmse"):
    return clo.get(phase, {}).get(key, float("nan"))


def _print_model(name, rows):
    print(f"[track-b] ---- {name}: analysis vs open-loop closure (impact phase) ----")
    for r in rows:
        cl = r["closure"]["C_L"]
        ew = r["closure"]["wake_enstrophy"]
        cl_r2_a = _phase(cl["analysis"], "impact", "r2")
        cl_r2_o = _phase(cl["open_loop"], "impact", "r2")
        print(
            f"  {r['case_id']} t_imp={r['t_impact']} n={r['n_assim']} | "
            f"C_L RMSE a/ol={_phase(cl['analysis'],'impact'):.3f}/"
            f"{_phase(cl['open_loop'],'impact'):.3f} R2 a/ol={cl_r2_a:.2f}/{cl_r2_o:.2f} | "
            f"E_w RMSE a/ol={_phase(ew['analysis'],'impact'):.3f}/"
            f"{_phase(ew['open_loop'],'impact'):.3f} | "
            f"innov|lag1|={r['innovation_whiteness']['mean_abs_lag1']:.2f} "
            f"NIS={r['nis_coverage']['mean_nis']:.1f}"
            f"(K={r['nis_coverage']['expected_nis']:.0f}) "
            f"div={r['divergence']['diverged']}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())

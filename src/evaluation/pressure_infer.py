"""Session 31 Track D Q3: state inference from wall pressure through frozen latents.

The last of the three ROM questions. For every canonical model's FROZEN encoder
latent (the Q1 cache) we ask: can a sparse wall-pressure sensor vector infer the
flow state *through that model's latent*, and is the predictive (JEPA) latent or
the reconstruction (AE) latent the better estimation target?

Protocol (Test B, restricted to the Track 0.B impact + relaxation windows baked
into the Q1 cache; anchored_local primary):

1. Fit a pressure -> latent estimator on ALL TRAIN frames (identical-capacity /
   all-train-fit / window-eval, matching Q1/Q2). Input = current-frame wall
   pressure ``p_wall (192,)``; target = the frozen encoder latent (detached). The
   estimator is a scalable RBF kernel-ridge (StandardScaler -> Nystroem(RBF) ->
   Ridge, alpha by a small held-out search), the same KernelRidge(RBF) idea the
   Session-21 pressure work used, made to scale to per-frame train. Two dedicated
   estimators per model: pressure -> GAP latent ``(d,)`` for the observable
   readout (the core), and pressure -> spatial latent ``(d, h, w)`` for the field
   readout (secondary).

2. On Test B, predict the latent from pressure, then read out through the FROZEN
   Q1 heads, scored on the impact + relaxation windows only:
   - OBSERVABLES: the five state observables read from the pressure-estimated GAP
     latent via the Q1 frozen linear + MLP probes -> R^2 vs the true observable.
     The *ceiling* is the same probe read from the TRUE latent (the Q1 windowed
     probe number).
   - FIELD: decode the pressure-estimated spatial latent via the Q1 identical
     capacity :class:`~src.models.decoder.SpatialLatentFieldDecoder` -> field
     VRMSE + SSIM vs DNS. The *floor* decodes the TRUE spatial latent.

3. COMPARISON: which model's frozen latent is the best pressure-estimation target
   (highest observable R^2 / lowest field VRMSE from pressure)? That answers
   whether the predictive or the reconstruction latent is the better state
   inference target from sparse wall sensors.

The pure helpers (:func:`align_pressure_to_rows`, :func:`latent_estimation_r2`,
:func:`readout_observable_from_pressure`, :func:`readout_field_from_pressure`) are
unit-tested in ``tests/test_pressure.py``. The heavy GPU decode floor + kernel
estimator fit live in :func:`run_q3`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_MODELS = (
    "jepa_nowake",
    "jepa_wake",
    "ae_nowake",
    "ae_wake",
    "supervised_only",
    "regAE",
)

OBSERVABLES = (
    "C_L",
    "C_D",
    "wake_enstrophy",
    "circulation_pos",
    "circulation_neg",
)

DEFAULT_ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
N_SURFACE = 192


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# =========================================================================== pure
def align_pressure_to_rows(
    pwall_by_key: dict[str, np.ndarray],
    case_id: np.ndarray,
    encounter_index: np.ndarray,
    frame: np.ndarray,
    n_surface: int = N_SURFACE,
) -> np.ndarray:
    """Join per-encounter wall pressure onto the flat latent rows by (case, enc, frame).

    Args:
        pwall_by_key: ``"{case_id}/{enc:02d}" -> p_wall (T, n_surface)`` per encounter.
        case_id: ``(N,)`` per-row case id.
        encounter_index: ``(N,)`` per-row encounter index.
        frame: ``(N,)`` per-row frame index into that encounter's ``p_wall``.
        n_surface: Number of wall taps.

    Returns:
        ``(N, n_surface)`` float32, row ``i`` = ``pwall_by_key[key_i][frame_i]``.

    Raises:
        KeyError: if a row references a ``(case, enc)`` absent from ``pwall_by_key``.
    """
    case_id = np.asarray(case_id).astype(str)
    encounter_index = np.asarray(encounter_index)
    frame = np.asarray(frame)
    n = case_id.shape[0]
    out = np.zeros((n, int(n_surface)), dtype=np.float32)
    keys = np.array([f"{c}/{int(e):02d}" for c, e in zip(case_id, encounter_index)])
    for key in np.unique(keys):
        if key not in pwall_by_key:
            raise KeyError(f"no wall pressure for encounter {key!r}")
        rows = np.where(keys == key)[0]
        pw = np.asarray(pwall_by_key[key], dtype=np.float32)
        out[rows] = pw[frame[rows]]
    return out


def latent_estimation_r2(z_pred: np.ndarray, z_true: np.ndarray, mask: np.ndarray | None) -> float:
    """Aggregated multi-output R^2 of a latent estimate, optionally windowed.

    ``1 - sum||z_pred - z_true||^2 / sum||z_true - mean_col(z_true)||^2`` where the
    column mean is taken over the (windowed) rows. Aggregated over all rows and
    latent dimensions before the ratio.
    """
    z_pred = np.asarray(z_pred, dtype=np.float64)
    z_true = np.asarray(z_true, dtype=np.float64)
    z_pred = z_pred.reshape(z_pred.shape[0], -1)
    z_true = z_true.reshape(z_true.shape[0], -1)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        z_pred = z_pred[mask]
        z_true = z_true[mask]
    mean_col = z_true.mean(axis=0, keepdims=True)
    num = float(((z_pred - z_true) ** 2).sum())
    den = float(((z_true - mean_col) ** 2).sum())
    if den <= 0.0:
        return float("nan")
    return float(1.0 - num / den)


class IdentityEstimator:
    """A pressure -> latent estimator whose ``predict`` returns its input unchanged.

    Used to pin the readout plumbing: feeding the TRUE latent as the "pressure
    estimate" must reproduce the direct-probe / decode-floor numbers exactly.
    """

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)


def readout_observable_from_pressure(
    estimator,
    probe,
    p_eval: np.ndarray,
    y_eval: np.ndarray,
    mask: np.ndarray | None,
) -> dict[str, float]:
    """Predict the latent from pressure, read an observable through a frozen probe.

    Returns ``{"pressure_r2", "n"}`` on the windowed eval rows. With an
    :class:`IdentityEstimator` and ``p_eval`` = the true latent this equals the
    direct probe R^2 (the ceiling).
    """
    from src.evaluation.represent import r2_score_np, select_window

    z_pred = estimator.predict(p_eval)
    if mask is not None:
        z_pred, y_eval = select_window(z_pred, y_eval, mask)
    y_hat = probe.predict(z_pred)
    return {"pressure_r2": r2_score_np(y_eval, y_hat), "n": int(np.asarray(y_eval).shape[0])}


def readout_field_from_pressure(
    estimator,
    decode_fn: Callable[[np.ndarray], np.ndarray],
    p_eval: np.ndarray,
    *,
    z_true_flat: np.ndarray,
    field_true: np.ndarray,
    latent_grid: tuple[int, int],
    latent_dim: int,
    mask: np.ndarray | None,
    ssim_L: float,
) -> dict[str, float]:
    """Decode the pressure-estimated spatial latent and score field VRMSE + SSIM.

    ``decode_fn`` maps a ``(N, d, h, w)`` latent batch to ``(N, Ho, Wo)`` fields
    (the shared decode-floor decoder). Reports the pressure-estimate curve and the
    floor (decode of the TRUE latent) on the windowed rows; with an
    :class:`IdentityEstimator` the two curves coincide.
    """
    from src.evaluation.represent import aggregated_vrmse, frame_ssim

    h, w = int(latent_grid[0]), int(latent_grid[1])
    d = int(latent_dim)
    n = p_eval.shape[0]
    z_pred = np.asarray(estimator.predict(p_eval), dtype=np.float32).reshape(n, d, h, w)
    z_true = np.asarray(z_true_flat, dtype=np.float32).reshape(n, d, h, w)
    pred_field = np.asarray(decode_fn(z_pred), dtype=np.float32)
    floor_field = np.asarray(decode_fn(z_true), dtype=np.float32)
    true = np.asarray(field_true, dtype=np.float32)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        pred_field = pred_field[mask]
        floor_field = floor_field[mask]
        true = true[mask]
    return {
        "pressure_vrmse": aggregated_vrmse(pred_field, true),
        "pressure_ssim": frame_ssim(pred_field, true, ssim_L),
        "floor_vrmse": aggregated_vrmse(floor_field, true),
        "floor_ssim": frame_ssim(floor_field, true, ssim_L),
        "n": int(true.shape[0]),
    }


# =========================================================================== estimator
@dataclass
class PressureEstimator:
    """A fitted StandardScaler -> Nystroem(RBF) -> Ridge pressure -> latent map."""

    scaler: object
    nystroem: object
    ridge: object
    alpha: float
    n_components: int

    def predict(self, p: np.ndarray) -> np.ndarray:
        feats = self.nystroem.transform(self.scaler.transform(np.asarray(p, dtype=np.float64)))
        return self.ridge.predict(feats)


def _select_alpha(
    feats: np.ndarray,
    targets: np.ndarray,
    alphas: Sequence[float],
    seed: int,
    holdout: float,
    groups: np.ndarray | None = None,
) -> float:
    """Pick the Ridge alpha minimising held-out latent MSE (multi-target safe).

    When ``groups`` is given (per-row encounter id), whole groups are held out so
    the alpha generalises across encounters rather than across temporally adjacent
    frames of the same encounter (which would leak and under-regularise).
    """
    from sklearn.linear_model import Ridge

    alphas = list(alphas)
    if len(alphas) == 1:
        return float(alphas[0])
    rng = np.random.default_rng(seed)
    if groups is None:
        n = feats.shape[0]
        idx = rng.permutation(n)
        nval = max(1, int(round(holdout * n)))
        val_mask = np.zeros(n, dtype=bool)
        val_mask[idx[:nval]] = True
    else:
        groups = np.asarray(groups)
        ug = np.unique(groups)
        rng.shuffle(ug)
        nval = max(1, int(round(holdout * len(ug))))
        val_groups = set(ug[:nval].tolist())
        val_mask = np.array([g in val_groups for g in groups])
    tr_mask = ~val_mask
    best_a, best_err = float(alphas[0]), np.inf
    for a in alphas:
        reg = Ridge(alpha=float(a)).fit(feats[tr_mask], targets[tr_mask])
        err = float(((reg.predict(feats[val_mask]) - targets[val_mask]) ** 2).mean())
        if err < best_err:
            best_err, best_a = err, float(a)
    return best_a


def fit_pressure_estimator(
    p_train: np.ndarray,
    z_train: np.ndarray,
    *,
    n_components: int = 400,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    seed: int = 0,
    holdout: float = 0.2,
    groups: np.ndarray | None = None,
) -> PressureEstimator:
    """Fit pressure -> latent as a scalable RBF kernel-ridge (Nystroem approximation).

    ``z_train`` may be ``(N, d)`` (GAP) or ``(N, d, h, w)`` (spatial); it is
    flattened to 2D. StandardScaler on pressure, Nystroem RBF feature map
    (default gamma ``1/n_features``), Ridge with alpha by a held-out search over
    ``alphas`` (encounter-grouped hold-out when ``groups`` is given, so alpha is
    not under-regularised by temporal leakage). This is the Session-21
    KernelRidge(RBF) idea made to scale to per-frame train.
    """
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    p_train = np.asarray(p_train, dtype=np.float64)
    z_train = np.asarray(z_train, dtype=np.float32)
    z_train = z_train.reshape(z_train.shape[0], -1)
    ncomp = int(min(n_components, p_train.shape[0]))
    scaler = StandardScaler().fit(p_train)
    ps = scaler.transform(p_train)
    nystroem = Nystroem(kernel="rbf", n_components=ncomp, random_state=seed).fit(ps)
    feats = nystroem.transform(ps)
    alpha = _select_alpha(feats, z_train, alphas, seed, holdout, groups=groups)
    ridge = Ridge(alpha=alpha).fit(feats, z_train)
    return PressureEstimator(scaler, nystroem, ridge, float(alpha), ncomp)


# =========================================================================== data io
def _cache_dir_for(partition: str, cache_root: str | Path | None) -> Path:
    if cache_root is None:
        env = os.environ.get("VORTEX_JEPA_CACHE")
        if env:
            cache_root = Path(env)
        else:
            prevent = os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT"))
            cache_root = Path(prevent) / "data" / "processed" / "vortex-jepa"
    return Path(cache_root) / partition


def _load_alignment(field_cache_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(case_id, encounter_index, frame)`` from a field cache npz."""
    import h5py  # noqa: F401  (ensure h5py present for the reader below)

    d = np.load(_resolve(field_cache_path), allow_pickle=True)
    return (
        np.asarray(d["case_id"]).astype(str),
        np.asarray(d["encounter_index"]),
        np.asarray(d["frame"]),
    )


def load_pressure_for_alignment(
    case_id: np.ndarray,
    encounter_index: np.ndarray,
    frame: np.ndarray,
    *,
    partition: str = "v2p2",
    cache_root: str | Path | None = None,
    n_surface: int = N_SURFACE,
) -> np.ndarray:
    """Read ``p_wall`` from the per-encounter cache and align it to the flat rows."""
    import h5py

    cache_dir = _cache_dir_for(partition, cache_root)
    case_id = np.asarray(case_id).astype(str)
    encounter_index = np.asarray(encounter_index)
    keys = np.array([f"{c}/{int(e):02d}" for c, e in zip(case_id, encounter_index)])
    pwall_by_key: dict[str, np.ndarray] = {}
    for key in np.unique(keys):
        cid, k = key.rsplit("/", 1)
        path = cache_dir / cid / f"encounter_{int(k):02d}.h5"
        with h5py.File(path, "r") as g:
            pwall_by_key[key] = np.asarray(g["p_wall"], dtype=np.float32)
    return align_pressure_to_rows(pwall_by_key, case_id, encounter_index, frame, n_surface)


def build_pressure_cache(
    cache_dir: str | Path,
    *,
    partition: str = "v2p2",
    cache_root: str | Path | None = None,
    n_surface: int = N_SURFACE,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Build (or load) the model-independent aligned pressure arrays for train/test_b."""
    cache_dir = _resolve(cache_dir)
    out: dict[str, np.ndarray] = {}
    for split in ("train", "test_b"):
        pcache = cache_dir / f"pressure_{split}.npz"
        align = _load_alignment(cache_dir / f"fields_{split}.npz")
        if pcache.exists():
            blob = np.load(pcache, allow_pickle=True)
            # sanity: alignment must match what the cache was built against.
            if np.array_equal(blob["case_id"].astype(str), align[0]) and np.array_equal(
                blob["frame"], align[2]
            ):
                out[split] = np.asarray(blob["p_wall"], dtype=np.float32)
                if verbose:
                    print(f"[q3] reuse pressure cache {pcache} {out[split].shape}", flush=True)
                continue
        p = load_pressure_for_alignment(
            *align, partition=partition, cache_root=cache_root, n_surface=n_surface
        )
        np.savez(
            pcache,
            p_wall=p.astype(np.float32),
            case_id=align[0].astype(str),
            encounter_index=align[1],
            frame=align[2],
        )
        out[split] = p
        if verbose:
            print(f"[q3] wrote pressure cache {pcache} {p.shape}", flush=True)
    return out


def load_gap_split(path: str | Path) -> dict:
    """Load only the GAP latent + alignment + targets from a Q1 latents npz (no z_spatial)."""
    d = np.load(_resolve(path), allow_pickle=True)
    return {
        "z_gap": np.asarray(d["z_gap"], dtype=np.float32),
        "case_id": np.asarray(d["case_id"]).astype(str),
        "encounter_index": np.asarray(d["encounter_index"]),
        "frame": np.asarray(d["frame"]),
        "window_mask": np.asarray(d["window_mask"], dtype=bool),
        "latent_grid": tuple(int(x) for x in d["latent_grid"]),
        "targets": {name: np.asarray(d[f"target_{name}"]) for name in OBSERVABLES},
    }


# =========================================================================== runner
def _row_groups(case_id: np.ndarray, encounter_index: np.ndarray) -> np.ndarray:
    """Per-row encounter group id ``"{case_id}/{enc}"`` for grouped alpha hold-out."""
    return np.array(
        [f"{c}/{int(e)}" for c, e in zip(np.asarray(case_id).astype(str), encounter_index)]
    )


def _subsample_taps(p: np.ndarray, k: int, n_surface: int = N_SURFACE) -> tuple[np.ndarray, list]:
    """Uniformly subsample ``k`` of ``n_surface`` taps (deterministic evenly spaced)."""
    if k >= n_surface:
        return p, list(range(n_surface))
    idx = np.linspace(0, n_surface - 1, k).round().astype(int)
    idx = np.unique(idx)
    return p[:, idx], idx.tolist()


def _observable_block(
    gap_tr: dict,
    gap_tb: dict,
    p_tr: np.ndarray,
    p_tb: np.ndarray,
    *,
    n_components: int,
    seed: int,
) -> dict:
    """Fit pressure->GAP estimator, read the five observables through the Q1 probes."""
    from src.evaluation.represent import fit_linear_probe, fit_mlp_probe

    mask = gap_tb["window_mask"]
    groups = _row_groups(gap_tr["case_id"], gap_tr["encounter_index"])
    est = fit_pressure_estimator(
        p_tr, gap_tr["z_gap"], n_components=n_components, seed=seed, groups=groups
    )
    z_gap_pred = est.predict(p_tb)
    lat_r2 = latent_estimation_r2(z_gap_pred, gap_tb["z_gap"], mask)

    out: dict = {
        "latent_r2": lat_r2,
        "estimator_alpha": est.alpha,
        "n_components": est.n_components,
    }
    per_obs: dict = {}
    for obs in OBSERVABLES:
        y_tr = gap_tr["targets"][obs]
        y_tb = gap_tb["targets"][obs]
        plin = fit_linear_probe(gap_tr["z_gap"], y_tr)
        pmlp = fit_mlp_probe(gap_tr["z_gap"], y_tr, seed=seed)
        p_lin = readout_observable_from_pressure(est, plin, p_tb, y_tb, mask)
        p_mlp = readout_observable_from_pressure(est, pmlp, p_tb, y_tb, mask)
        c_lin = readout_observable_from_pressure(
            IdentityEstimator(), plin, gap_tb["z_gap"], y_tb, mask
        )
        c_mlp = readout_observable_from_pressure(
            IdentityEstimator(), pmlp, gap_tb["z_gap"], y_tb, mask
        )
        per_obs[obs] = {
            "pressure_linear_r2": p_lin["pressure_r2"],
            "pressure_mlp_r2": p_mlp["pressure_r2"],
            "ceiling_linear_r2": c_lin["pressure_r2"],
            "ceiling_mlp_r2": c_mlp["pressure_r2"],
            "n": p_lin["n"],
        }
    out["per_observable"] = per_obs
    return out


def _sensor_sweep_block(
    gap_tr: dict,
    gap_tb: dict,
    p_tr: np.ndarray,
    p_tb: np.ndarray,
    ks: Sequence[int],
    *,
    n_components: int,
    seed: int,
) -> dict:
    """Cheap sensor-count sweep on the observable path (linear-probe C_L + mean R^2)."""
    from src.evaluation.represent import fit_linear_probe

    mask = gap_tb["window_mask"]
    groups = _row_groups(gap_tr["case_id"], gap_tr["encounter_index"])
    probes = {obs: fit_linear_probe(gap_tr["z_gap"], gap_tr["targets"][obs]) for obs in OBSERVABLES}
    sweep: dict = {}
    for k in ks:
        p_tr_k, idx = _subsample_taps(p_tr, int(k))
        p_tb_k, _ = _subsample_taps(p_tb, int(k))
        est = fit_pressure_estimator(
            p_tr_k, gap_tr["z_gap"], n_components=n_components, seed=seed, groups=groups
        )
        r2s = {}
        for obs in OBSERVABLES:
            r = readout_observable_from_pressure(
                est, probes[obs], p_tb_k, gap_tb["targets"][obs], mask
            )
            r2s[obs] = r["pressure_r2"]
        sweep[str(int(k))] = {
            "n_taps": len(idx),
            "linear_r2": r2s,
            "mean_linear_r2": float(np.mean(list(r2s.values()))),
        }
    return sweep


def run_q3(
    models: Sequence[str] = CANONICAL_MODELS,
    *,
    cache_dir: str | Path = "outputs/session31/q1_latents",
    out_json: str | Path = "outputs/session31/q3_pressure.json",
    pipeline_manifest: str | Path = "outputs/data_pipeline/v2p2/manifest.json",
    partition: str = "v2p2",
    cache_root: str | Path | None = None,
    n_components: int = 400,
    seed: int = 0,
    do_field: bool = True,
    sensor_sweep: Sequence[int] | None = None,
    decoder_steps: int = 6000,
    decoder_batch: int = 64,
    gpu: int = 0,
    device_override: str | None = None,
) -> dict:
    """Q3 orchestration: pressure -> latent -> {observables, field} for all models.

    The observable path (the core comparison) is pure sklearn and runs first,
    writing incrementally; the field path (secondary) needs the GPU decode floor.
    """
    from src.data.omega_pipeline import ssim_data_range

    cache_dir = _resolve(cache_dir)
    out_path = _resolve(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ssim_L = float(ssim_data_range(str(_resolve(pipeline_manifest))))

    pressure = build_pressure_cache(cache_dir, partition=partition, cache_root=cache_root)
    p_tr, p_tb = pressure["train"], pressure["test_b"]

    payload: dict = {
        "task": "SESSION 31 Track D Q3 -- state inference from wall pressure",
        "params": {
            "partition": partition,
            "pipeline_manifest": str(pipeline_manifest),
            "n_taps": int(p_tr.shape[1]),
            "window_definition": (
                "Track 0.B impact union relaxation frames (anchored_local) baked into "
                "the Q1 latent cache window_mask; eval on Test B restricted to these"
            ),
            "estimator": (
                "pressure -> latent = StandardScaler -> Nystroem(RBF, "
                f"n_components={n_components}) -> Ridge(alpha by 20% encounter-grouped "
                f"held-out search over {list(DEFAULT_ALPHAS)}); scalable KernelRidge(RBF), "
                "Session-21 idea. Current-frame p_wall, fit on ALL train frames."
            ),
            "observable_readout": (
                "Q1 frozen GAP-latent linear + MLP probes read from the pressure-"
                "estimated GAP latent; ceiling = same probe on the TRUE latent"
            ),
            "field_readout": (
                "Q1 identical-capacity SpatialLatentFieldDecoder (refit on TRAIN "
                "latents) decoding the pressure-estimated spatial latent; floor = "
                "decode of the TRUE spatial latent"
            ),
            "ssim_data_range_L": ssim_L,
            "observables": list(OBSERVABLES),
            "seed": seed,
            "estimator_target_note": (
                "dedicated pressure->GAP estimator for observables; dedicated "
                "pressure->spatial estimator for the field decode"
            ),
        },
        "models": {},
    }

    def _write():
        out_path.write_text(json.dumps(payload, indent=2))

    _write()

    # ---------------- OBSERVABLE PATH (core, CPU) -------------------------------
    print(f"[q3] observable path: {len(models)} models, n_taps={p_tr.shape[1]}", flush=True)
    for name in models:
        gap_tr = load_gap_split(cache_dir / f"latents_{name}_train.npz")
        gap_tb = load_gap_split(cache_dir / f"latents_{name}_test_b.npz")
        obs = _observable_block(gap_tr, gap_tb, p_tr, p_tb, n_components=n_components, seed=seed)
        entry: dict = {"pressure_to_obs": obs}
        if sensor_sweep:
            entry["sensor_sweep"] = _sensor_sweep_block(
                gap_tr, gap_tb, p_tr, p_tb, sensor_sweep, n_components=n_components, seed=seed
            )
        payload["models"][name] = entry
        _write()
        _print_obs_row(name, obs)
        del gap_tr, gap_tb

    # ---------------- FIELD PATH (secondary, GPU) -------------------------------
    if do_field:
        _run_field_path(
            models,
            cache_dir,
            p_tr,
            p_tb,
            payload,
            _write,
            ssim_L=ssim_L,
            n_components=n_components,
            seed=seed,
            decoder_steps=decoder_steps,
            decoder_batch=decoder_batch,
            gpu=gpu,
            device_override=device_override,
        )

    _write()
    print(f"\n[q3] wrote {out_path}", flush=True)
    _print_summary(payload)
    return payload


def _make_decode(decoder, decode_fields, device) -> Callable[[np.ndarray], np.ndarray]:
    """Bind a trained decoder + device into a ``(N, d, h, w) -> (N, Ho, Wo)`` callable."""

    def decode(z_sp: np.ndarray) -> np.ndarray:
        return decode_fields(decoder, z_sp, device)

    return decode


def _run_field_path(
    models,
    cache_dir,
    p_tr,
    p_tb,
    payload,
    write_fn,
    *,
    ssim_L,
    n_components,
    seed,
    decoder_steps,
    decoder_batch,
    gpu,
    device_override,
) -> None:
    import torch

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
    payload["params"]["gpu_name"] = gpu_name
    payload["params"]["decoder_steps"] = decoder_steps

    omega_tr, *_ = load_field_cache(cache_dir / "fields_train.npz")
    omega_tb, *_ = load_field_cache(cache_dir / "fields_test_b.npz")

    print(f"[q3] field path: device={device} ({gpu_name})", flush=True)
    for name in models:
        tr = load_latent_split(cache_dir / f"latents_{name}_train.npz")
        tb = load_latent_split(cache_dir / f"latents_{name}_test_b.npz")
        grid = tr.latent_grid
        d = int(tr.z_spatial.shape[1])

        groups = _row_groups(tr.case_id, tr.encounter_index)
        est = fit_pressure_estimator(
            p_tr,
            tr.z_spatial.reshape(tr.z_spatial.shape[0], -1),
            n_components=n_components,
            seed=seed,
            groups=groups,
        )
        decoder = fit_decode_floor_decoder(
            tr.z_spatial,
            omega_tr,
            grid,
            device=device,
            steps=decoder_steps,
            batch=decoder_batch,
            seed=seed,
            verbose=True,
        )
        decode = _make_decode(decoder, decode_fields, device)

        field = readout_field_from_pressure(
            est,
            decode,
            p_tb,
            z_true_flat=tb.z_spatial.reshape(tb.z_spatial.shape[0], -1),
            field_true=omega_tb,
            latent_grid=grid,
            latent_dim=d,
            mask=tb.window_mask,
            ssim_L=ssim_L,
        )
        field["estimator_alpha"] = est.alpha
        payload["models"].setdefault(name, {})["pressure_to_field"] = field
        write_fn()
        _print_field_row(name, field)
        del est, decoder, tr, tb
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _print_obs_row(name: str, obs: dict) -> None:
    per = obs["per_observable"]
    cl = per["C_L"]
    mean_lin = float(np.mean([per[o]["pressure_linear_r2"] for o in OBSERVABLES]))
    print(
        f"[q3] {name}: latent_r2={obs['latent_r2']:.3f} | "
        f"C_L press lin/ceil={cl['pressure_linear_r2']:.3f}/{cl['ceiling_linear_r2']:.3f} | "
        f"mean-obs press lin R2={mean_lin:.3f}",
        flush=True,
    )


def _print_field_row(name: str, field: dict) -> None:
    print(
        f"[q3] {name}: field VRMSE press/floor={field['pressure_vrmse']:.3f}/"
        f"{field['floor_vrmse']:.3f}  SSIM press/floor={field['pressure_ssim']:.3f}/"
        f"{field['floor_ssim']:.3f}",
        flush=True,
    )


def _print_summary(payload: dict) -> None:
    print("\n[q3] ===== Q3 SUMMARY: pressure -> latent -> state (Test B, in-window) =====")
    print(
        f"{'model':<16}{'latR2':>7}{'C_L(p/c)':>14}{'meanObs(p/c)':>16}"
        f"{'fieldVRMSE(p/f)':>18}{'fieldSSIM(p/f)':>16}"
    )
    for name, m in payload["models"].items():
        obs = m.get("pressure_to_obs")
        if obs:
            per = obs["per_observable"]
            cl = per["C_L"]
            mean_p = float(np.mean([per[o]["pressure_linear_r2"] for o in OBSERVABLES]))
            mean_c = float(np.mean([per[o]["ceiling_linear_r2"] for o in OBSERVABLES]))
            lat = f"{obs['latent_r2']:.2f}"
            cls = f"{cl['pressure_linear_r2']:.2f}/{cl['ceiling_linear_r2']:.2f}"
            means = f"{mean_p:.2f}/{mean_c:.2f}"
        else:
            lat, cls, means = "-", "-", "-"
        fld = m.get("pressure_to_field")
        if fld:
            fv = f"{fld['pressure_vrmse']:.2f}/{fld['floor_vrmse']:.2f}"
            fs = f"{fld['pressure_ssim']:.2f}/{fld['floor_ssim']:.2f}"
        else:
            fv = fs = "-"
        print(f"{name:<16}{lat:>7}{cls:>14}{means:>16}{fv:>18}{fs:>16}")
    print("[q3] ==================================================================\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 31 Track D Q3 pressure -> state inference")
    p.add_argument("--models", nargs="+", default=list(CANONICAL_MODELS))
    p.add_argument("--cache-dir", default="outputs/session31/q1_latents")
    p.add_argument("--out", default="outputs/session31/q3_pressure.json")
    p.add_argument("--pipeline-manifest", default="outputs/data_pipeline/v2p2/manifest.json")
    p.add_argument("--partition", default="v2p2")
    p.add_argument("--n-components", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-field", action="store_true", help="Skip the GPU field-decode path.")
    p.add_argument(
        "--sensor-sweep",
        nargs="+",
        type=int,
        default=None,
        help="Optional tap-count sweep on the observable path, e.g. --sensor-sweep 8 16 192.",
    )
    p.add_argument("--decoder-steps", type=int, default=6000)
    p.add_argument("--decoder-batch", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None, help="Explicit device (bypasses require_rtx6000).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device is None and os.environ.get("VORTEX_JEPA_ROM_DEVICE"):
        args.device = os.environ["VORTEX_JEPA_ROM_DEVICE"]
    run_q3(
        models=args.models,
        cache_dir=args.cache_dir,
        out_json=args.out,
        pipeline_manifest=args.pipeline_manifest,
        partition=args.partition,
        n_components=args.n_components,
        seed=args.seed,
        do_field=not args.no_field,
        sensor_sweep=args.sensor_sweep,
        decoder_steps=args.decoder_steps,
        decoder_batch=args.decoder_batch,
        gpu=args.gpu,
        device_override=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

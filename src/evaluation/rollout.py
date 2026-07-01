"""Session 31 Track D Q2: temporal forecast ROM eval over the six canonical models.

The decisive part of the AE-versus-JEPA comparison: does the predictive latent
forecast the gust impact and relaxation better than the reconstruction latent?

For every canonical model's FROZEN latent (Q1 cache) we fit a FRESH matched
predictor (encoder frozen, latents detached) on TRAIN latents and evaluate on
Test B, restricted to the Track 0.B impact + relaxation windows. This isolates
encoder geometry, not how each model trained (the matched-predictor protocol).

Predictor variants per model:
- ``resunet_matched`` (PRIMARY): :class:`~src.models.resunet_predictor.ResUNetPredictor`
  (context_length=2), autoregressive roll from context ``(z_{t-1}, z_t)`` up to H.
- ``transformer_matched`` (SECONDARY): :class:`~src.models.predictor.AutoregressivePredictor`
  on the GAP-pooled latent (the v2.1 winner, for the record). No field decode.
- ``native`` (SECONDARY, jepa_* only): the co-trained ResUNet in the checkpoint,
  the as-built ROM.

Metrics (Test B, in-window targets, per horizon h=1..H):
1. FIELD VRMSE three-curve floor < model < persistence. ``floor`` decodes the TRUE
   future latent, ``model`` decodes the ROLLED latent, ``persistence`` holds the
   last-context DNS field. All scored against the DNS field at t+h with the shared
   :class:`~src.models.decoder.SpatialLatentFieldDecoder` (identical capacity to
   the Q1 decode floor, refit here on TRAIN latents).
2. OBSERVABLE FORWARD CLOSURE (the ROM figure of merit): read the five observables
   from the rolled latent through the Q1 frozen GAP-latent probes; report R^2 and
   VRMSE vs the true observable at t+h, with floor (readout on the true latent) and
   persistence (readout on the held anchor latent) for context.
3. ON-MANIFOLD DRIFT: aggregated latent rel-L2 ``||z_rolled - z_true|| / ||z_true||``
   per horizon.

The pure helpers (:func:`roll_autoregressive`, :func:`assemble_three_curve`,
:func:`drift_rel_l2`, :func:`window_horizon_samples`) are unit-tested in
``tests/test_rollout.py``. The heavy GPU orchestration lives in :func:`run_q2`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
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

# Which models carry a co-trained (native) ResUNet predictor in the checkpoint.
NATIVE_PREDICTOR_MODELS = ("jepa_nowake", "jepa_wake")

OBSERVABLES = (
    "C_L",
    "C_D",
    "wake_enstrophy",
    "circulation_pos",
    "circulation_neg",
)

CONTEXT_LENGTH = 2
DEFAULT_HORIZON = 16


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# =========================================================================== pure
def roll_autoregressive(
    predict_fn: Callable[[list], object],
    context: Sequence,
    steps: int,
) -> list:
    """Open-loop autoregressive rollout, framework-agnostic.

    From an initial ``context`` list of ``context_length`` frames, calls
    ``predict_fn(context)`` to produce the next frame, appends it and slides the
    window, for ``steps`` iterations. Matches the training-time
    :func:`src.training.canonical_model.assemble_spatial_rollout` feedback exactly.

    Args:
        predict_fn: Maps a list of ``context_length`` frames to the next frame.
            The concatenation convention (channel stacking for the ResUNet,
            sequence growth for the transformer) is the caller's responsibility.
        context: Initial ``context_length`` frames (arrays or tensors).
        steps: Number of rollout steps to produce.

    Returns:
        A list of ``steps`` predicted frames, in horizon order (h = 1 .. steps).
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    window = list(context)
    preds: list = []
    for _ in range(steps):
        nxt = predict_fn(window)
        preds.append(nxt)
        window = window[1:] + [nxt]
    return preds


def assemble_three_curve(
    rolled: np.ndarray,
    z_true: np.ndarray,
    anchor_frames: np.ndarray,
    horizons: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the (model, floor, persistence) latent curves + target indices.

    Horizon indexing: the target frame at ``(anchor a, horizon h)`` is ``a + h``.

    Args:
        rolled: ``(A, H, ...)`` rolled latents (model prediction).
        z_true: ``(L, ...)`` true latent trajectory of the encounter.
        anchor_frames: ``(A,)`` local frame index of the last context frame.
        horizons: length-``H`` sequence of horizon offsets (e.g. ``1..H``).

    Returns:
        ``(model, floor, persist, target)``. ``model`` is ``rolled`` unchanged;
        ``floor[a, h] = z_true[a + horizon_h]``; ``persist[a, h] = z_true[a]``
        (last-context latent held); ``target[a, h] = a + horizon_h``.
    """
    anchor_frames = np.asarray(anchor_frames)
    horizons_arr = np.asarray(horizons)
    target = anchor_frames[:, None] + horizons_arr[None, :]  # (A, H)
    floor = z_true[target]  # (A, H, ...)
    anchor_vals = z_true[anchor_frames]  # (A, ...)
    persist = np.broadcast_to(
        anchor_vals[:, None], (anchor_vals.shape[0], len(horizons_arr)) + anchor_vals.shape[1:]
    ).copy()
    return rolled, floor, persist, target


def drift_rel_l2(rolled: np.ndarray, true: np.ndarray) -> float:
    """Aggregated on-manifold drift ``sqrt(sum||rolled-true||^2 / sum||true||^2)``.

    Aggregated over the sample set (numerator and denominator summed before the
    ratio), matching the VRMSE aggregation convention.
    """
    rolled = np.asarray(rolled, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if rolled.shape != true.shape:
        raise ValueError(f"shape mismatch: rolled {rolled.shape} vs true {true.shape}")
    num = float(((rolled - true) ** 2).sum())
    den = float((true**2).sum())
    if den <= 0.0:
        return float("nan")
    return float(np.sqrt(num / den))


def window_horizon_samples(
    window_mask: np.ndarray,
    anchor_frames: np.ndarray,
    horizons: Sequence[int],
    context_length: int,
    n_frames: int,
) -> dict[int, np.ndarray]:
    """Per horizon, the anchor-array indices whose target frame is a valid in-window sample.

    A sample ``(anchor a, horizon h)`` is kept when the context fits
    (``a - (context_length - 1) >= 0``), the target ``a + h`` is in range
    (``0 <= a + h < n_frames``), and the target frame is inside the impact +
    relaxation window (``window_mask[a + h]``).

    Returns:
        ``{h: indices}`` where ``indices`` selects into ``anchor_frames``.
    """
    anchor_frames = np.asarray(anchor_frames)
    window_mask = np.asarray(window_mask, dtype=bool)
    ctx_ok = anchor_frames - (context_length - 1) >= 0
    out: dict[int, np.ndarray] = {}
    for h in horizons:
        tgt = anchor_frames + int(h)
        in_range = (tgt >= 0) & (tgt < int(n_frames))
        inwin = np.zeros(anchor_frames.shape[0], dtype=bool)
        inwin[in_range] = window_mask[tgt[in_range]]
        out[int(h)] = np.where(ctx_ok & in_range & inwin)[0]
    return out


# =========================================================================== data
@dataclass
class LatentSplit:
    """Flat per-frame latents + alignment + targets for one split, one model."""

    z_spatial: np.ndarray  # (N, d, h, w) float32
    z_gap: np.ndarray  # (N, d) float32
    case_id: np.ndarray  # (N,) str
    encounter_index: np.ndarray  # (N,) int
    frame: np.ndarray  # (N,) int
    window_mask: np.ndarray  # (N,) bool
    targets: dict[str, np.ndarray]  # name -> (N,)
    latent_grid: tuple[int, int]


@dataclass
class Encounter:
    """One encounter's contiguous latent trajectory + alignment into the flat cache."""

    key: str
    z_spatial: np.ndarray  # (L, d, h, w)
    z_gap: np.ndarray  # (L, d)
    window_mask: np.ndarray  # (L,)
    global_rows: np.ndarray  # (L,) row index into the flat cache (for fields/targets)
    frames: np.ndarray  # (L,)


def load_latent_split(path: str | Path) -> LatentSplit:
    """Load a Q1 ``latents_<model>_<split>.npz`` into a :class:`LatentSplit`."""
    d = np.load(_resolve(path), allow_pickle=True)
    targets = {name: np.asarray(d[f"target_{name}"]) for name in OBSERVABLES}
    grid = tuple(int(x) for x in d["latent_grid"])
    return LatentSplit(
        z_spatial=np.asarray(d["z_spatial"], dtype=np.float32),
        z_gap=np.asarray(d["z_gap"], dtype=np.float32),
        case_id=np.asarray(d["case_id"]).astype(str),
        encounter_index=np.asarray(d["encounter_index"]),
        frame=np.asarray(d["frame"]),
        window_mask=np.asarray(d["window_mask"], dtype=bool),
        targets=targets,
        latent_grid=grid,
    )


def group_encounters(split: LatentSplit) -> list[Encounter]:
    """Group a flat :class:`LatentSplit` into per-encounter trajectories (frame-sorted)."""
    keys = np.array([f"{c}/{int(e):02d}" for c, e in zip(split.case_id, split.encounter_index)])
    out: list[Encounter] = []
    for key in sorted(np.unique(keys)):
        rows = np.where(keys == key)[0]
        order = np.argsort(split.frame[rows])
        rows = rows[order]
        out.append(
            Encounter(
                key=key,
                z_spatial=split.z_spatial[rows],
                z_gap=split.z_gap[rows],
                window_mask=split.window_mask[rows],
                global_rows=rows,
                frames=split.frame[rows],
            )
        )
    return out


def encounter_trajectories(split: LatentSplit) -> list[np.ndarray]:
    """Per-encounter ``(L, d, h, w)`` train trajectories for predictor fitting."""
    return [enc.z_spatial for enc in group_encounters(split)]


def load_field_cache(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ``fields_<split>.npz`` -> ``(omega_norm, case_id, encounter_index, frame)``.

    ``omega_norm`` is returned as float32 aligned row-for-row with the latent split
    (both were written from the same encode pass so the rows correspond).
    """
    d = np.load(_resolve(path), allow_pickle=True)
    return (
        np.asarray(d["omega_norm"], dtype=np.float32),
        np.asarray(d["case_id"]).astype(str),
        np.asarray(d["encounter_index"]),
        np.asarray(d["frame"]),
    )


# =========================================================================== predictors
def _sample_subtraj_batch(
    trajs: list[np.ndarray], length: int, batch: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample a ``(batch, length, ...)`` block of contiguous sub-trajectories."""
    out = np.empty((batch,) + (length,) + trajs[0].shape[1:], dtype=np.float32)
    n = len(trajs)
    for b in range(batch):
        ti = int(rng.integers(0, n))
        z = trajs[ti]
        s = int(rng.integers(0, z.shape[0] - length + 1))
        e = s + length
        out[b] = z[s:e]
    return out


def fit_matched_resunet(
    trajs: list[np.ndarray],
    *,
    device,
    horizon_train: int = 8,
    context_length: int = CONTEXT_LENGTH,
    steps: int = 4000,
    batch: int = 64,
    lr: float = 5e-4,
    seed: int = 0,
    log_every: int = 1000,
    verbose: bool = True,
):
    """Fit a fresh ResUNet predictor on frozen TRAIN latents by multi-step rollout MSE.

    Same open-loop rollout objective the native predictor trains under (kit
    ``pred`` term, ``mse_seq``): from the first ``context_length`` frames roll
    ``horizon_train`` steps and regress against the DETACHED true future latents.
    The encoder is frozen (latents are precomputed), so no anti-collapse is needed
    (the regression targets are fixed).
    """
    import torch
    from torch.optim import AdamW

    from src.models.resunet_predictor import ResUNetPredictor
    from src.training.canonical_model import assemble_spatial_rollout

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    latent_dim = int(trajs[0].shape[1])
    predictor = ResUNetPredictor(latent_dim=latent_dim, context_length=context_length).to(device)
    predictor.train()
    opt = AdamW(predictor.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    length = context_length + horizon_train
    use_cuda = getattr(device, "type", str(device)) == "cuda"

    for step in range(steps):
        blk = _sample_subtraj_batch(trajs, length, batch, rng)  # (B, T, d, h, w)
        zb = torch.from_numpy(blk).to(device)
        if use_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_seq, target_seq = assemble_spatial_rollout(predictor, zb, horizon_train)
                loss = torch.nn.functional.mse_loss(pred_seq.float(), target_seq.float())
        else:
            pred_seq, target_seq = assemble_spatial_rollout(predictor, zb, horizon_train)
            loss = torch.nn.functional.mse_loss(pred_seq, target_seq)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        opt.step()
        if verbose and (step % log_every == 0 or step == steps - 1):
            print(f"    [matched-resunet] step {step}/{steps} mse={loss.item():.5f}", flush=True)

    predictor.eval()
    return predictor


def fit_matched_transformer(
    trajs: list[np.ndarray],
    *,
    device,
    latent_dim: int = 32,
    hidden_dim: int = 256,
    depth: int = 4,
    heads: int = 8,
    subtraj_len: int = 12,
    steps: int = 4000,
    batch: int = 64,
    lr: float = 5e-4,
    seed: int = 0,
    log_every: int = 1000,
    verbose: bool = True,
):
    """Fit a fresh unconditioned transformer on the GAP-pooled TRAIN latent (teacher-forced).

    Secondary readout of the same frozen encoder geometry through the v2.1 winner
    architecture. ``trajs`` are the spatial trajectories; they are GAP-pooled to
    ``(L, d)`` here. Trained teacher-forced next-step (the transformer's native
    objective).
    """
    import torch
    from torch.optim import AdamW

    from src.models.predictor import AutoregressivePredictor

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gap_trajs = [t.mean(axis=(2, 3)).astype(np.float32) for t in trajs]  # (L, d)
    predictor = AutoregressivePredictor(
        latent_dim=latent_dim,
        cond_dim=0,
        hidden_dim=hidden_dim,
        depth=depth,
        heads=heads,
        max_seq_len=max(32, subtraj_len + 2),
    ).to(device)
    predictor.train()
    opt = AdamW(predictor.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    use_cuda = getattr(device, "type", str(device)) == "cuda"

    for step in range(steps):
        blk = _sample_subtraj_batch(gap_trajs, subtraj_len, batch, rng)  # (B, T, d)
        zb = torch.from_numpy(blk).to(device)
        if use_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_hat = predictor(zb, None)  # (B, T, d): z_hat[:, t] predicts z[:, t+1]
                loss = torch.nn.functional.mse_loss(z_hat[:, :-1].float(), zb[:, 1:].float())
        else:
            z_hat = predictor(zb, None)
            loss = torch.nn.functional.mse_loss(z_hat[:, :-1], zb[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        opt.step()
        if verbose and (step % log_every == 0 or step == steps - 1):
            print(
                f"    [matched-transformer] step {step}/{steps} mse={loss.item():.5f}", flush=True
            )

    predictor.eval()
    return predictor


def load_native_predictor(
    run_dir: str | Path,
    checkpoint_name: str,
    latent_dim: int,
    device,
):
    """Rebuild the co-trained ResUNet predictor from a canonical checkpoint (frozen)."""
    import torch

    from src.config.kit_config import load_model_config
    from src.training.canonical_model import CanonicalModel

    run_path = Path(run_dir)
    ckpt_path = run_path if run_path.is_file() else run_path / checkpoint_name
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob.get("args", {})
    run_config = blob.get("run_config", {})
    config_path = args.get("config") or run_config.get("config_path")
    cfg = load_model_config(_resolve(config_path))
    projection_norm = args.get("projection_norm", "batchnorm")
    model = CanonicalModel(cfg, latent_dim=int(latent_dim), projection_norm=projection_norm)
    model.load_state_dict(blob["model_state_dict"], strict=True)
    if model.predictor is None:
        raise ValueError(f"{run_dir} has no native predictor (objective={cfg.objective})")
    predictor = model.predictor
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)
    predictor.to(device)
    return predictor


# =========================================================================== rolling
def _roll_resunet_batch(predictor, ctx_frames: list, steps: int):
    """Roll a batch with a ResUNet predictor. ``ctx_frames``: cl tensors (B, d, h, w)."""
    import torch

    def predict(window):
        return predictor(torch.cat(window, dim=1))

    preds = roll_autoregressive(predict, ctx_frames, steps)  # list of (B, d, h, w)
    return torch.stack(preds, dim=1)  # (B, steps, d, h, w)


def _roll_transformer_batch(predictor, z_init, steps: int):
    """Roll a batch with the transformer. ``z_init``: (B, cl, d). Returns (B, steps, d)."""
    z_full = predictor.rollout(z_init, None, steps)  # (B, cl + steps, d)
    cl = z_init.shape[1]
    return z_full[:, cl:]  # drop the seed context frames


# =========================================================================== eval
@dataclass
class HorizonBucket:
    rolled_sp: list = field(default_factory=list)
    true_sp: list = field(default_factory=list)
    rolled_gap: list = field(default_factory=list)
    true_gap: list = field(default_factory=list)
    anchor_gap: list = field(default_factory=list)
    target_row: list = field(default_factory=list)
    anchor_row: list = field(default_factory=list)


def _new_buckets(horizons):
    return {int(h): HorizonBucket() for h in horizons}


def _needed_anchors(
    window_mask: np.ndarray, context_length: int, horizon: int, L: int
) -> np.ndarray:
    """Anchors (last-context frames) with at least one in-window target within ``horizon``."""
    anchors = []
    for t in range(context_length - 1, L - 1):
        lo = t + 1
        hi = min(L, t + horizon + 1)
        if window_mask[lo:hi].any():
            anchors.append(t)
    return np.asarray(anchors, dtype=int)


def _gather_spatial(
    predictor,
    encounters: list[Encounter],
    horizons: Sequence[int],
    context_length: int,
    device,
) -> dict[int, HorizonBucket]:
    """Roll every needed anchor of every encounter with a spatial (ResUNet) predictor."""
    import torch

    H = max(horizons)
    buckets = _new_buckets(horizons)
    for enc in encounters:
        Z = enc.z_spatial  # (L, d, h, w)
        L = Z.shape[0]
        wm = enc.window_mask
        anchors = _needed_anchors(wm, context_length, H, L)
        if anchors.size == 0:
            continue
        ctx = [
            torch.from_numpy(Z[anchors - (context_length - 1) + j]).to(device)
            for j in range(context_length)
        ]
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    rolled = _roll_resunet_batch(predictor, ctx, H)
            else:
                rolled = _roll_resunet_batch(predictor, ctx, H)
        rolled_np = rolled.float().cpu().numpy()  # (A, H, d, h, w)
        rolled_gap = rolled_np.mean(axis=(3, 4))  # (A, H, d)
        sel = window_horizon_samples(wm, anchors, horizons, context_length, L)
        for h in horizons:
            idx = sel[int(h)]
            if idx.size == 0:
                continue
            a = anchors[idx]
            tgt = a + int(h)
            step = int(h) - 1  # rolled tensor holds steps 1..H at indices 0..H-1
            b = buckets[int(h)]
            b.rolled_sp.append(rolled_np[idx, step].astype(np.float16))
            b.true_sp.append(Z[tgt].astype(np.float16))
            b.rolled_gap.append(rolled_gap[idx, step].astype(np.float32))
            b.true_gap.append(Z[tgt].mean(axis=(2, 3)).astype(np.float32))
            b.anchor_gap.append(Z[a].mean(axis=(2, 3)).astype(np.float32))
            b.target_row.append(enc.global_rows[tgt])
            b.anchor_row.append(enc.global_rows[a])
    return buckets


def _gather_gap(
    predictor,
    encounters: list[Encounter],
    horizons: Sequence[int],
    context_length: int,
    device,
) -> dict[int, HorizonBucket]:
    """Roll every needed anchor with the transformer (GAP-latent) predictor."""
    import torch

    H = max(horizons)
    buckets = _new_buckets(horizons)
    for enc in encounters:
        Zg = enc.z_gap  # (L, d)
        L = Zg.shape[0]
        wm = enc.window_mask
        anchors = _needed_anchors(wm, context_length, H, L)
        if anchors.size == 0:
            continue
        # seed = the cl context frames per anchor -> (A, cl, d)
        seed = np.stack(
            [Zg[anchors - (context_length - 1) + j] for j in range(context_length)], axis=1
        )
        z_init = torch.from_numpy(seed.astype(np.float32)).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    rolled = _roll_transformer_batch(predictor, z_init, H)
            else:
                rolled = _roll_transformer_batch(predictor, z_init, H)
        rolled_np = rolled.float().cpu().numpy()  # (A, H, d)
        sel = window_horizon_samples(wm, anchors, horizons, context_length, L)
        for h in horizons:
            idx = sel[int(h)]
            if idx.size == 0:
                continue
            a = anchors[idx]
            tgt = a + int(h)
            step = int(h) - 1  # rolled tensor holds steps 1..H at indices 0..H-1
            b = buckets[int(h)]
            b.rolled_gap.append(rolled_np[idx, step].astype(np.float32))
            b.true_gap.append(Zg[tgt].astype(np.float32))
            b.anchor_gap.append(Zg[a].astype(np.float32))
            b.target_row.append(enc.global_rows[tgt])
            b.anchor_row.append(enc.global_rows[a])
    return buckets


def _field_curves(
    buckets: dict[int, HorizonBucket],
    decoder,
    omega: np.ndarray,
    horizons: Sequence[int],
    device,
) -> dict:
    """Three-curve field VRMSE (model/floor/persistence) per horizon."""
    from src.evaluation.represent import aggregated_vrmse, decode_fields

    hs, model, floor, persist, ns = [], [], [], [], []
    for h in horizons:
        b = buckets[int(h)]
        if not b.rolled_sp:
            continue
        rolled_sp = np.concatenate(b.rolled_sp).astype(np.float32)
        true_sp = np.concatenate(b.true_sp).astype(np.float32)
        target_row = np.concatenate(b.target_row)
        anchor_row = np.concatenate(b.anchor_row)
        model_field = decode_fields(decoder, rolled_sp, device)
        floor_field = decode_fields(decoder, true_sp, device)
        true_field = omega[target_row]
        persist_field = omega[anchor_row]
        hs.append(int(h))
        model.append(aggregated_vrmse(model_field, true_field))
        floor.append(aggregated_vrmse(floor_field, true_field))
        persist.append(aggregated_vrmse(persist_field, true_field))
        ns.append(int(true_field.shape[0]))
    return {"horizons": hs, "model": model, "floor": floor, "persistence": persist, "n": ns}


def _observable_curves(
    buckets: dict[int, HorizonBucket],
    probes: dict[str, dict],
    flat_targets: dict[str, np.ndarray],
    horizons: Sequence[int],
) -> dict:
    """Observable forward-closure curves (R^2 + VRMSE) per observable per horizon."""
    from src.evaluation.represent import aggregated_vrmse, r2_score_np

    out: dict[str, dict] = {}
    for obs in OBSERVABLES:
        lin = probes[obs]["linear"]
        mlp = probes[obs]["mlp"]
        hs, m_lin, m_mlp, fl, per, vr, ns = [], [], [], [], [], [], []
        for h in horizons:
            b = buckets[int(h)]
            if not b.rolled_gap:
                continue
            rolled_gap = np.concatenate(b.rolled_gap)
            true_gap = np.concatenate(b.true_gap)
            anchor_gap = np.concatenate(b.anchor_gap)
            target_row = np.concatenate(b.target_row)
            y_true = flat_targets[obs][target_row]
            y_model_lin = lin.predict(rolled_gap)
            y_model_mlp = mlp.predict(rolled_gap)
            y_floor = lin.predict(true_gap)
            y_persist = lin.predict(anchor_gap)
            hs.append(int(h))
            m_lin.append(r2_score_np(y_true, y_model_lin))
            m_mlp.append(r2_score_np(y_true, y_model_mlp))
            fl.append(r2_score_np(y_true, y_floor))
            per.append(r2_score_np(y_true, y_persist))
            vr.append(aggregated_vrmse(y_model_lin, y_true))
            ns.append(int(y_true.shape[0]))
        out[obs] = {
            "horizons": hs,
            "model_linear_r2": m_lin,
            "model_mlp_r2": m_mlp,
            "floor_linear_r2": fl,
            "persistence_linear_r2": per,
            "model_linear_vrmse": vr,
            "n": ns,
        }
    return out


def _drift_curve(buckets: dict[int, HorizonBucket], horizons: Sequence[int], spatial: bool) -> dict:
    """Aggregated on-manifold latent drift rel-L2 per horizon."""
    hs, vals, ns = [], [], []
    for h in horizons:
        b = buckets[int(h)]
        if spatial and b.rolled_sp:
            rolled = np.concatenate(b.rolled_sp).astype(np.float32)
            true = np.concatenate(b.true_sp).astype(np.float32)
        elif (not spatial) and b.rolled_gap:
            rolled = np.concatenate(b.rolled_gap)
            true = np.concatenate(b.true_gap)
        else:
            continue
        rolled = rolled.reshape(rolled.shape[0], -1)
        true = true.reshape(true.shape[0], -1)
        hs.append(int(h))
        vals.append(drift_rel_l2(rolled, true))
        ns.append(int(true.shape[0]))
    return {"horizons": hs, "rel_l2": vals, "n": ns}


def fit_observable_probes(z_gap_train: np.ndarray, targets_train: dict[str, np.ndarray]) -> dict:
    """Fit the Q1 frozen GAP-latent readout (linear + MLP) for each observable."""
    from src.evaluation.represent import fit_linear_probe, fit_mlp_probe

    probes: dict[str, dict] = {}
    for obs in OBSERVABLES:
        y = targets_train[obs]
        probes[obs] = {
            "linear": fit_linear_probe(z_gap_train, y),
            "mlp": fit_mlp_probe(z_gap_train, y, seed=0),
        }
    return probes


# =========================================================================== runner
def _resunet_param_block(horizon_train, steps, batch, lr):
    return {
        "arch": "ResUNetPredictor(context_length=2, base_channels=64, depth=2)",
        "objective": "multi-step open-loop rollout MSE (mse_seq), detached targets",
        "horizon_train": horizon_train,
        "steps": steps,
        "batch": batch,
        "lr": lr,
    }


def run_q2(
    models: Sequence[str] = CANONICAL_MODELS,
    *,
    runs_base: str | Path = "outputs/runs/session31",
    checkpoint_name: str = "checkpoint_iter010000.pt",
    cache_dir: str | Path = "outputs/session31/q1_latents",
    out_json: str | Path = "outputs/session31/q2_temporal.json",
    horizon: int = DEFAULT_HORIZON,
    horizon_train: int = 8,
    predictor_steps: int = 4000,
    predictor_batch: int = 64,
    predictor_lr: float = 5e-4,
    do_transformer: bool = True,
    do_native: bool = True,
    windows_tag: str = "anchored_local",
    windows_path: str | Path = "outputs/session31/windows_v2p2.json",
    gpu: int = 0,
    device_override: str | None = None,
    decoder_steps: int = 6000,
    decoder_batch: int = 64,
    seed: int = 0,
) -> dict:
    """Q2 orchestration: matched (+ native/transformer) rollout eval for all models."""
    import torch

    from src.evaluation.represent import fit_decode_floor_decoder

    if device_override is not None:
        device = torch.device(device_override)
        gpu_name = device_override
    else:
        from src.utils.device import require_rtx6000

        device = require_rtx6000(gpu_index=gpu)
        gpu_name = torch.cuda.get_device_name(device.index)
        if "RTX" not in gpu_name or "6000" not in gpu_name:
            raise RuntimeError(f"hardware policy: gpu_name={gpu_name!r} is not an RTX 6000")

    horizons = list(range(1, horizon + 1))
    cache_dir = _resolve(cache_dir)
    out_path = _resolve(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Model-independent Test B fields (shared across models).
    omega_tb, f_cid, f_enc, f_frame = load_field_cache(cache_dir / "fields_test_b.npz")
    omega_tr, *_ = load_field_cache(cache_dir / "fields_train.npz")

    print(
        f"[q2] device={device} ({gpu_name}) horizon={horizon} windows={windows_tag} "
        f"models={list(models)}",
        flush=True,
    )

    payload: dict = {
        "task": "SESSION 31 Track D Q2 -- temporal forecast ROM eval",
        "params": {
            "checkpoint": checkpoint_name,
            "gpu_name": gpu_name,
            "horizon": horizon,
            "context_length": CONTEXT_LENGTH,
            "windows_tag": windows_tag,
            "windows_path": str(windows_path),
            "matched_predictor_protocol": (
                "fresh predictor fit on frozen (detached) TRAIN latents per model; "
                "eval Test B in-window targets. Isolates encoder geometry."
            ),
            "field_vrmse_definition": (
                "aggregated_vrmse(pred, DNS at t+h) with the shared "
                "SpatialLatentFieldDecoder; floor=decode(true z_{t+h}), "
                "model=decode(rolled), persistence=held last-context DNS field"
            ),
            "observable_closure": (
                "Q1 frozen GAP-latent probe (linear+MLP) read from the rolled latent; "
                "R^2 and VRMSE vs true observable at t+h; floor=readout on true "
                "z_{t+h}, persistence=readout on held anchor latent"
            ),
            "drift_definition": "aggregated sqrt(sum||z_rolled-z_true||^2/sum||z_true||^2)",
            "window_scope": (
                "Test B eval restricted to (anchor, horizon) samples whose target "
                "frame t+h is in the impact union relaxation window"
            ),
            "observables": list(OBSERVABLES),
            "resunet_matched": _resunet_param_block(
                horizon_train, predictor_steps, predictor_batch, predictor_lr
            ),
            "transformer_matched": {
                "arch": (
                    "AutoregressivePredictor(cond_dim=0, hidden=256, depth=4, heads=8) "
                    "on GAP latent"
                ),
                "objective": "teacher-forced next-step MSE",
                "steps": predictor_steps,
                "note": "secondary; observable-closure + drift only (no field decode)",
            },
            "decode_floor_decoder": {
                "arch": "SpatialLatentFieldDecoder (Q1 identical-capacity), refit on TRAIN latents",
                "steps": decoder_steps,
                "batch": decoder_batch,
            },
            "seed": seed,
        },
        "models": {},
    }

    def _write():
        out_path.write_text(json.dumps(payload, indent=2))

    _write()

    for name in models:
        print(f"\n[q2] === {name} ===", flush=True)
        tr = load_latent_split(cache_dir / f"latents_{name}_train.npz")
        tb = load_latent_split(cache_dir / f"latents_{name}_test_b.npz")
        trajs = encounter_trajectories(tr)
        encounters_tb = group_encounters(tb)
        latent_grid = tr.latent_grid

        # decode-floor decoder (identical capacity to Q1; refit on TRAIN latents).
        decoder = fit_decode_floor_decoder(
            tr.z_spatial,
            omega_tr,
            latent_grid,
            device=device,
            steps=decoder_steps,
            batch=decoder_batch,
            seed=seed,
            verbose=True,
        )
        # frozen GAP-latent observable readout (Q1 probe).
        probes = fit_observable_probes(tr.z_gap, tr.targets)

        model_out: dict = {}

        # ---- PRIMARY: matched ResUNet ----
        pred_rn = fit_matched_resunet(
            trajs,
            device=device,
            horizon_train=horizon_train,
            steps=predictor_steps,
            batch=predictor_batch,
            lr=predictor_lr,
            seed=seed,
        )
        buckets = _gather_spatial(pred_rn, encounters_tb, horizons, CONTEXT_LENGTH, device)
        model_out["resunet_matched"] = {
            "field_vrmse": _field_curves(buckets, decoder, omega_tb, horizons, device),
            "observable_closure": _observable_curves(buckets, probes, tb.targets, horizons),
            "drift": _drift_curve(buckets, horizons, spatial=True),
            "params": _resunet_param_block(
                horizon_train, predictor_steps, predictor_batch, predictor_lr
            ),
        }
        del pred_rn, buckets
        if device.type == "cuda":
            torch.cuda.empty_cache()
        payload["models"][name] = {
            "objective": _objective_of(runs_base, name, checkpoint_name),
            "predictors": model_out,
        }
        _write()
        _print_row(name, "resunet_matched", model_out["resunet_matched"])

        # ---- SECONDARY: native ResUNet (jepa_* only) ----
        if do_native and name in NATIVE_PREDICTOR_MODELS:
            pred_nat = load_native_predictor(
                _resolve(runs_base) / name, checkpoint_name, int(tr.z_spatial.shape[1]), device
            )
            buckets = _gather_spatial(pred_nat, encounters_tb, horizons, CONTEXT_LENGTH, device)
            model_out["native"] = {
                "field_vrmse": _field_curves(buckets, decoder, omega_tb, horizons, device),
                "observable_closure": _observable_curves(buckets, probes, tb.targets, horizons),
                "drift": _drift_curve(buckets, horizons, spatial=True),
                "params": {
                    "arch": "co-trained ResUNetPredictor from the checkpoint (as-built ROM)"
                },
            }
            del pred_nat, buckets
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _write()
            _print_row(name, "native", model_out["native"])

        # ---- SECONDARY: matched transformer on GAP latent ----
        if do_transformer:
            pred_tf = fit_matched_transformer(
                trajs,
                device=device,
                latent_dim=int(tr.z_spatial.shape[1]),
                steps=predictor_steps,
                batch=predictor_batch,
                lr=predictor_lr,
                seed=seed,
            )
            buckets = _gather_gap(pred_tf, encounters_tb, horizons, CONTEXT_LENGTH, device)
            model_out["transformer_matched"] = {
                "observable_closure": _observable_curves(buckets, probes, tb.targets, horizons),
                "drift": _drift_curve(buckets, horizons, spatial=False),
                "params": {
                    "arch": "AutoregressivePredictor(cond_dim=0) on GAP latent",
                    "note": "no field VRMSE (GAP latent cannot feed the spatial decoder)",
                },
            }
            del pred_tf, buckets
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _write()
            _print_row(name, "transformer_matched", model_out["transformer_matched"])

        del decoder, tr, tb, trajs, encounters_tb
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write()
    print(f"\n[q2] wrote {out_path}", flush=True)
    _print_summary(payload)
    return payload


def _objective_of(runs_base, name, checkpoint_name) -> str:
    try:
        import torch

        from src.config.kit_config import load_model_config

        run_path = _resolve(runs_base) / name
        ckpt = run_path if run_path.is_file() else run_path / checkpoint_name
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        args = blob.get("args", {})
        config_path = args.get("config") or blob.get("run_config", {}).get("config_path")
        return load_model_config(_resolve(config_path)).objective
    except Exception:
        return "unknown"


def _at(curve: dict, key: str, h: int):
    hs = curve.get("horizons", [])
    if h in hs:
        return curve[key][hs.index(h)]
    return float("nan")


def _print_row(name: str, variant: str, out: dict) -> None:
    fv = out.get("field_vrmse")
    parts = [f"[q2] {name}/{variant}:"]
    if fv:
        parts.append(
            f"field VRMSE h1 m/f/p={_at(fv, 'model', 1):.3f}/{_at(fv, 'floor', 1):.3f}/"
            f"{_at(fv, 'persistence', 1):.3f}  h8 m/f/p={_at(fv, 'model', 8):.3f}/"
            f"{_at(fv, 'floor', 8):.3f}/{_at(fv, 'persistence', 8):.3f}"
        )
    oc = out.get("observable_closure", {})
    if "C_L" in oc:
        parts.append(
            f"| C_L closure R2 h1/h8={_at(oc['C_L'], 'model_linear_r2', 1):.2f}/"
            f"{_at(oc['C_L'], 'model_linear_r2', 8):.2f}"
        )
    print(" ".join(parts), flush=True)


def _print_summary(payload: dict) -> None:
    print(
        "\n[q2] ===== Q2 SUMMARY: field VRMSE model/floor/persist (h1,h8), "
        "C_L closure R2 (h8) ====="
    )
    print(
        f"{'model':<16}{'variant':<20}{'VRMSE h1(m/f/p)':>20}"
        f"{'VRMSE h8(m/f/p)':>20}{'CLr2 h8':>9}"
    )
    for name, m in payload["models"].items():
        for variant, out in m["predictors"].items():
            fv = out.get("field_vrmse")
            oc = out.get("observable_closure", {})
            if fv:
                h1 = (
                    f"{_at(fv,'model',1):.2f}/{_at(fv,'floor',1):.2f}/{_at(fv,'persistence',1):.2f}"
                )
                h8 = (
                    f"{_at(fv,'model',8):.2f}/{_at(fv,'floor',8):.2f}/{_at(fv,'persistence',8):.2f}"
                )
            else:
                h1 = h8 = "  -  "
            clr = _at(oc.get("C_L", {}), "model_linear_r2", 8) if oc else float("nan")
            print(f"{name:<16}{variant:<20}{h1:>22}{h8:>22}{clr:>9.2f}")
    print("[q2] ============================================================================\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 31 Track D Q2 temporal forecast eval")
    p.add_argument("--models", nargs="+", default=list(CANONICAL_MODELS))
    p.add_argument("--runs-base", default="outputs/runs/session31")
    p.add_argument("--checkpoint", default="checkpoint_iter010000.pt")
    p.add_argument("--cache-dir", default="outputs/session31/q1_latents")
    p.add_argument("--out", default="outputs/session31/q2_temporal.json")
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--horizon-train", type=int, default=8)
    p.add_argument("--predictor-steps", type=int, default=4000)
    p.add_argument("--predictor-batch", type=int, default=64)
    p.add_argument("--predictor-lr", type=float, default=5e-4)
    p.add_argument("--no-transformer", action="store_true")
    p.add_argument("--no-native", action="store_true")
    p.add_argument("--windows-tag", default="anchored_local")
    p.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--decoder-steps", type=int, default=6000)
    p.add_argument("--decoder-batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device is None and os.environ.get("VORTEX_JEPA_ROM_DEVICE"):
        args.device = os.environ["VORTEX_JEPA_ROM_DEVICE"]
    run_q2(
        models=args.models,
        runs_base=args.runs_base,
        checkpoint_name=args.checkpoint,
        cache_dir=args.cache_dir,
        out_json=args.out,
        horizon=args.horizon,
        horizon_train=args.horizon_train,
        predictor_steps=args.predictor_steps,
        predictor_batch=args.predictor_batch,
        predictor_lr=args.predictor_lr,
        do_transformer=not args.no_transformer,
        do_native=not args.no_native,
        windows_tag=args.windows_tag,
        windows_path=args.windows,
        gpu=args.gpu,
        device_override=args.device,
        decoder_steps=args.decoder_steps,
        decoder_batch=args.decoder_batch,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

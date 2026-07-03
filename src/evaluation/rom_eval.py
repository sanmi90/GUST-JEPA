"""Shared ROM-eval harness for Session 31 Track D (Q1/Q2/Q3).

This module rebuilds a frozen canonical encoder from a training checkpoint, then
encodes the encounters of a split into per-(case, encounter, frame) latents plus
the aligned per-frame targets and the pipeline-normalised omega field the decode
floor scores against. The heavy GPU work (the encoder) lives here; the Q1 metric
math and probe fitting live in :mod:`src.evaluation.represent`.

Design notes:
- The encoder is rebuilt in the SAME ``latent_mode`` it was trained in by loading
  the model's kit config from the checkpoint's stored ``args['config']`` and
  wiring a :class:`~src.training.canonical_model.CanonicalModel`, then extracting
  and freezing its ``encoder``. Loading the full ``CanonicalModel`` state dict
  (encoder + predictor/decoder/heads) keeps the BatchNorm running statistics
  exact; only the encoder is kept.
- Targets are model-independent. ``C_L`` / ``C_D`` come straight from the cache;
  ``circulation_pos`` / ``circulation_neg`` / ``wake_enstrophy`` are computed
  on the masked-and-clipped (``preprocess_raw``) omega field with the SAME
  canonical wake-box definitions as ``scripts/session17/exp2_dns_physical_metrics``
  so the numbers match the published closure observables.
- The pipeline-normalised omega field (``preprocess_raw`` then ``normalize``) is
  the decode-floor target; the loss / SSIM convention lives in normalised space
  (CLAUDE.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch import nn

from src.config.kit_config import ResolvedKitConfig, load_model_config
from src.data.omega_pipeline import OmegaPipeline
from src.probes.base import freeze_encoder
from src.training.canonical_model import CanonicalModel

REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical wake-box + integration geometry (verbatim from
# scripts/session17/exp2_dns_physical_metrics.py so the observable targets match
# the published closure numbers). Grid is the (192, 96) cache resolution over the
# (-1.5, 4.5) x (-1.5, 1.5) physical extent.
_DX = 6.0 / 192
_DY = 3.0 / 96
_X_GRID = np.linspace(-1.5, 4.5, 192).astype(np.float32)
_Y_GRID = np.linspace(-1.5, 1.5, 96).astype(np.float32)
_WAKE_X_MIN, _WAKE_X_MAX = 0.5, 4.0
_WAKE_Y_MAX = 1.0
_CIRC_THRESHOLD = 1.0  # |omega| > 1 for the signed circulation, raw scale.

# The five state observables this eval reports, in a fixed column order.
FIELD_OBSERVABLES = ("wake_enstrophy", "circulation_pos", "circulation_neg")
CACHE_OBSERVABLES = ("C_L", "C_D")
ALL_OBSERVABLES = CACHE_OBSERVABLES + FIELD_OBSERVABLES


def _wake_mask_2d() -> np.ndarray:
    xx = _X_GRID[:, None]
    yy = _Y_GRID[None, :]
    return (xx >= _WAKE_X_MIN) & (xx <= _WAKE_X_MAX) & (np.abs(yy) <= _WAKE_Y_MAX)


def physical_observables(omega_clean_raw: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame wake enstrophy + signed circulation on a masked/clipped omega field.

    Args:
        omega_clean_raw: ``(T, 192, 96)`` omega on the RAW (un-normalised) scale
            after ``OmegaPipeline.preprocess_raw`` (spatial mask + per-encounter
            clip). This is the same field the closure observables integrate.

    Returns:
        Dict with ``wake_enstrophy``, ``circulation_pos``, ``circulation_neg``,
        each a ``(T,)`` float64 array (area-weighted wake-box integrals).
    """
    omega = np.asarray(omega_clean_raw, dtype=np.float32)
    if omega.ndim != 3 or omega.shape[1:] != (192, 96):
        raise ValueError(f"expected (T, 192, 96) omega; got {omega.shape}")
    wake = _wake_mask_2d()
    omega_wake = omega * wake[None, :, :]
    wake_enstrophy = (omega_wake**2).sum(axis=(1, 2)) * _DX * _DY
    pos_mask = omega_wake > _CIRC_THRESHOLD
    neg_mask = omega_wake < -_CIRC_THRESHOLD
    circulation_pos = (omega_wake * pos_mask).sum(axis=(1, 2)) * _DX * _DY
    circulation_neg = (omega_wake * neg_mask).sum(axis=(1, 2)) * _DX * _DY
    return {
        "wake_enstrophy": wake_enstrophy.astype(np.float64),
        "circulation_pos": circulation_pos.astype(np.float64),
        "circulation_neg": circulation_neg.astype(np.float64),
    }


# --------------------------------------------------------------------------- model load
@dataclass
class FrozenModel:
    """A frozen canonical encoder plus the resolved config it was built from."""

    encoder: nn.Module
    cfg: ResolvedKitConfig
    latent_dim: int
    latent_grid: tuple[int, int]
    name: str
    checkpoint: Path


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_frozen_model(
    run_dir: str | Path,
    checkpoint_name: str = "checkpoint_iter010000.pt",
    device: str | torch.device = "cpu",
) -> FrozenModel:
    """Rebuild the encoder from a canonical checkpoint and freeze it.

    Args:
        run_dir: A run directory (``outputs/runs/session31/<model>``) or a direct
            path to a checkpoint ``.pt`` file.
        checkpoint_name: Checkpoint filename when ``run_dir`` is a directory.
        device: Device to move the frozen encoder onto.

    Returns:
        A :class:`FrozenModel` with the encoder in ``eval`` mode and every
        encoder parameter ``requires_grad=False``.
    """
    run_path = Path(run_dir)
    ckpt_path = run_path if run_path.is_file() else run_path / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob.get("args", {})
    run_config = blob.get("run_config", {})

    config_path = args.get("config") or run_config.get("config_path")
    if config_path is None:
        raise KeyError(f"checkpoint {ckpt_path} has no config path in args/run_config")
    cfg = load_model_config(_resolve(config_path))

    latent_dim = int(args.get("d", 32))
    projection_norm = args.get("projection_norm", "batchnorm")
    predictor_class = args.get("predictor_class", "resunet")
    model = CanonicalModel(
        cfg,
        latent_dim=latent_dim,
        projection_norm=projection_norm,
        predictor_class=predictor_class,
    )
    model.load_state_dict(blob["model_state_dict"], strict=True)

    encoder = model.encoder
    freeze_encoder(encoder)
    encoder.to(device)
    return FrozenModel(
        encoder=encoder,
        cfg=cfg,
        latent_dim=latent_dim,
        latent_grid=tuple(int(x) for x in encoder.latent_grid),
        name=cfg.name,
        checkpoint=ckpt_path,
    )


# --------------------------------------------------------------------------- references
# Published-baseline references. They use their OWN architectures (fukami: native
# lift-augmented CNN AE; pod: linear basis), so they cannot be rebuilt as a
# CanonicalModel. Each produces a POOLED (B, T, d) latent; encode_split then lifts
# it to the (B, T, d, h, w) spatial grid with the SAME parameter-free broadcast the
# jepa_pool ablation uses, so every downstream Q1/Q2/Q3 helper is byte-identical.
REFERENCE_MODELS = ("fukami", "fukami_wake", "pod", "bvae")

# The spatial grid the pooled reference latent is broadcast to, matched to the
# shared backbone (HybridCNNViTEncoder default (24, 12)) so the decode-floor
# decoder capacity is identical across references and the spine.
REFERENCE_LATENT_GRID = (24, 12)


class _FukamiRefEncoder(nn.Module):
    """Frozen wrapper exposing a trained Fukami CNN encoder as a pooled encoder.

    ``encode_split`` feeds pipeline-normalised omega ``(B, T, 1, H, W)`` (the same
    space the Fukami AE was trained in, ``omega_scale=1.0``), so the reference
    latent is simply the native CNN encoder output ``(B, T, d)``.
    """

    latent_mode = "pooled"

    def __init__(self, cnn_encoder: nn.Module) -> None:
        super().__init__()
        self.cnn_encoder = cnn_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn_encoder(x)


class _PODEncoder(nn.Module):
    """Frozen linear POD projector as a pooled encoder.

    Projects each pipeline-normalised frame onto the top-``d`` POD modes:
    ``z = (flatten(omega_norm) - mean) @ components``. Computed in float32 (autocast
    disabled) so the projection of the ~18k-dim field is not degraded to bf16.
    """

    latent_mode = "pooled"

    def __init__(self, basis) -> None:
        super().__init__()
        self.register_buffer(
            "mean", torch.from_numpy(np.asarray(basis.mean, dtype=np.float32)), persistent=False
        )
        self.register_buffer(
            "components",
            torch.from_numpy(np.asarray(basis.components, dtype=np.float32)),
            persistent=False,
        )
        self._d = int(basis.components.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, H, W) or (B, T, H, W). Flatten spatial dims to P.
        if x.dim() == 5:
            b, t = x.shape[0], x.shape[1]
        elif x.dim() == 4:
            b, t = x.shape[0], x.shape[1]
        else:
            raise ValueError(f"expected (B, T, 1, H, W) omega; got {tuple(x.shape)}")
        flat = x.reshape(b, t, -1).float()
        dev_type = "cuda" if flat.is_cuda else "cpu"
        with torch.autocast(device_type=dev_type, enabled=False):
            z = (flat - self.mean) @ self.components  # (B, T, d)
        return z


def load_reference_model(
    run_dir: str | Path,
    checkpoint_name: str = "checkpoint_iter010000.pt",
    device: str | torch.device = "cpu",
) -> FrozenModel:
    """Load a published-baseline reference (fukami / fukami_wake / pod) as a FrozenModel.

    POD is detected by a ``pod_basis.npz`` in ``run_dir`` (no checkpoint). Fukami
    reads its training checkpoint, rebuilds the native
    :class:`~src.baselines.fukami_ae.FukamiAEWrapper` and keeps only the frozen CNN
    encoder. Both expose a pooled ``(B, T, d)`` latent through the shared harness.
    """
    from src.baselines.pod import load_pod_basis

    run_path = Path(run_dir)

    # ---- POD: linear basis, no neural checkpoint ----
    pod_npz = run_path / "pod_basis.npz"
    if pod_npz.exists():
        basis = load_pod_basis(pod_npz)
        encoder = _PODEncoder(basis)
        freeze_encoder(encoder)
        encoder.to(device)
        cfg = load_model_config(_resolve("configs/reference/pod.yaml"))
        return FrozenModel(
            encoder=encoder,
            cfg=cfg,
            latent_dim=int(basis.d),
            latent_grid=REFERENCE_LATENT_GRID,
            name="pod",
            checkpoint=pod_npz,
        )

    # ---- Fukami: native lift-augmented CNN AE ----
    from src.baselines.fukami_ae import FukamiAEWrapper
    from src.data.wake_observables import mode_output_dim
    from src.models.observable_head import WakeObservableHead

    ckpt_path = run_path if run_path.is_file() else run_path / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"reference checkpoint not found: {ckpt_path}")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob.get("args", {})
    run_config = blob.get("run_config", {})

    # ---- bvae: Solera-Rico beta-VAE (session9 native trainer, wrapper_state_dict) ----
    # Session 32 Track P: native beta-VAE (BetaVAEWrapper = FukamiAEWrapper + variational
    # head + KL). encode returns the deterministic latent mu; the shared harness broadcasts
    # the pooled (B, T, d) mu to the spatial grid, byte-identical to the fukami reference.
    if "wrapper_state_dict" in blob and bool(args.get("vae")):
        from src.baselines.solera_rico import BetaVAEWrapper

        # bvae.yaml carries a provenance-only `bvae:` block the kit whitelist rejects;
        # fukami_wake.yaml has the identical active-term set (recon+lift+wake, objective
        # recon), so it supplies accurate metadata while name is forced to "bvae".
        cfg = load_model_config(_resolve("configs/reference/fukami_wake.yaml"))
        latent_dim = int(args.get("d", 32))
        wake_on = "wake" in cfg.active_terms()
        wake_head_b = (
            WakeObservableHead(
                latent_dim=latent_dim, out_dim=mode_output_dim("patch_signed_spectrum")
            )
            if wake_on
            else None
        )
        wrapper = BetaVAEWrapper(
            latent_dim=latent_dim,
            n_deltas=1,
            lambda_recon=1.0,
            lambda_lift=1.0,
            omega_scale=1.0,
            recon_loss_type="mse",
            encoder_kind="cnn",
            wake_observable_head=wake_head_b,
            wake_observable_weight=1.0 if wake_on else 0.0,
            beta=float(args.get("beta", 2.5e-3)),
            beta_warmup_steps=int(
                float(args.get("beta_warmup_frac", 0.02)) * int(args.get("max_iters", 10000))
            ),
        )
        wrapper.load_state_dict(blob["wrapper_state_dict"], strict=True)
        encoder = _FukamiRefEncoder(wrapper.encoder)
        freeze_encoder(encoder)
        encoder.to(device)
        return FrozenModel(
            encoder=encoder,
            cfg=cfg,
            latent_dim=latent_dim,
            latent_grid=REFERENCE_LATENT_GRID,
            name="bvae",
            checkpoint=ckpt_path,
        )

    config_path = args.get("config") or run_config.get("config_path")
    if config_path is None:
        raise KeyError(f"reference checkpoint {ckpt_path} has no config path")
    cfg = load_model_config(_resolve(config_path))
    latent_dim = int(args.get("d", 32))

    wake_on = "wake" in cfg.active_terms()
    wake_head = (
        WakeObservableHead(latent_dim=latent_dim, out_dim=mode_output_dim("patch_signed_spectrum"))
        if wake_on
        else None
    )
    wrapper = FukamiAEWrapper(
        latent_dim=latent_dim,
        n_deltas=1,
        lambda_recon=1.0,
        lambda_lift=1.0,
        omega_scale=1.0,
        recon_loss_type="mse",
        encoder_kind="cnn",
        wake_observable_head=wake_head,
        wake_observable_weight=1.0 if wake_on else 0.0,
    )
    wrapper.load_state_dict(blob["model_state_dict"], strict=True)
    encoder = _FukamiRefEncoder(wrapper.encoder)
    freeze_encoder(encoder)
    encoder.to(device)
    return FrozenModel(
        encoder=encoder,
        cfg=cfg,
        latent_dim=latent_dim,
        latent_grid=REFERENCE_LATENT_GRID,
        name=cfg.name,
        checkpoint=ckpt_path,
    )


# --------------------------------------------------------------------------- split enum
def enumerate_encounters(split: str, split_manifest: dict) -> list[tuple[str, int]]:
    """List ``(case_id, encounter_index)`` for a split, matching EpisodeDataset.

    ``train`` uses each train case's ``train_encounter_indices``; ``test_a`` uses
    the within-train ``val_encounter_indices``; ``test_b`` / ``test_c`` enumerate
    all encounters of the matching test cases.
    """
    out: list[tuple[str, int]] = []
    for cid, case in split_manifest["cases"].items():
        if split == "train" and case["split"] == "train":
            out += [(cid, int(k)) for k in case["train_encounter_indices"]]
        elif split == "test_a" and case["split"] == "train":
            ks = case.get("val_encounter_indices") or case.get("test_a_encounter_indices")
            out += [(cid, int(k)) for k in ks]
        elif split in ("test_b", "test_c") and case["split"] == split:
            out += [(cid, k) for k in range(int(case["n_encounters_full"]))]
    if not out:
        raise RuntimeError(f"no encounters for split={split!r}")
    return out


# --------------------------------------------------------------------------- encode
@dataclass
class EncodedSplit:
    """Per-frame latents, targets and normalised field for one split, one model.

    All row-indexed arrays share the same first dimension ``N`` (total frames);
    ``case_id`` / ``encounter_index`` / ``frame`` give the alignment key and
    ``window_mask`` marks impact + relaxation frames.
    """

    z_spatial: np.ndarray  # (N, d, h, w) float32
    z_gap: np.ndarray  # (N, d) float32
    omega_norm: np.ndarray  # (N, 192, 96) float32 (pipeline-normalised)
    targets: dict[str, np.ndarray]  # name -> (N,) float64
    case_id: np.ndarray  # (N,) str
    encounter_index: np.ndarray  # (N,) int
    frame: np.ndarray  # (N,) int
    window_mask: np.ndarray  # (N,) bool
    latent_grid: tuple[int, int]
    split: str
    model_name: str
    extras: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.z_gap.shape[0])


def _window_entry(windows: dict | None, case_id: str, k: int) -> dict | None:
    if windows is None:
        return None
    return windows.get(f"{case_id}/{int(k):02d}")


def frame_window_mask_for_entry(entry: dict | None, n_frames: int) -> np.ndarray:
    """Impact + relaxation per-frame boolean mask from a windows.json entry."""
    from src.evaluation.represent import frame_window_mask

    return frame_window_mask(entry, n_frames)


@torch.no_grad()
def _encode_frames(
    encoder: nn.Module,
    omega_norm: np.ndarray,
    device: torch.device,
    frame_chunk: int,
    use_bf16: bool,
) -> np.ndarray:
    """Encode ``(T, 192, 96)`` normalised omega -> ``(T, d, h, w)`` float32 latents."""
    T = omega_norm.shape[0]
    x = torch.from_numpy(omega_norm[None, :, None, :, :]).to(device)  # (1, T, 1, H, W)
    chunks: list[torch.Tensor] = []
    for s in range(0, T, frame_chunk):
        e = min(s + frame_chunk, T)
        sub = x[:, s:e]
        if use_bf16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = encoder(sub)
        else:
            z = encoder(sub)
        chunks.append(z.squeeze(0).float().cpu())
    return torch.cat(chunks, dim=0).numpy()


def encode_split(
    frozen: FrozenModel,
    split: str,
    *,
    partition: str = "v2p2",
    split_manifest_path: str | Path = "configs/splits/split_v2p2.json",
    pipeline_manifest: str | Path = "outputs/data_pipeline/v2p2/manifest.json",
    cache_root: str | Path | None = None,
    windows: dict | None = None,
    device: str | torch.device = "cpu",
    frame_chunk: int = 40,
    limit: int | None = None,
    use_bf16: bool = True,
    verbose: bool = True,
) -> EncodedSplit:
    """Encode every encounter of a split into aligned per-frame latents + targets.

    Returns an :class:`EncodedSplit`. The frozen encoder runs on ``device`` (the
    only GPU work); targets and the normalised field are model-independent and
    computed from the cache.
    """
    device = torch.device(device)
    split_path = _resolve(split_manifest_path)
    with open(split_path) as fh:
        manifest = json.load(fh)
    pipe = OmegaPipeline.from_manifest(_resolve(pipeline_manifest))

    if cache_root is None:
        import os

        env = os.environ.get("VORTEX_JEPA_CACHE")
        if env:
            cache_root = Path(env)
        else:
            prevent = os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT"))
            cache_root = Path(prevent) / "data" / "processed" / "vortex-jepa"
    cache_dir = Path(cache_root) / partition

    encounters = enumerate_encounters(split, manifest)
    if limit is not None:
        encounters = encounters[:limit]

    z_sp_list: list[np.ndarray] = []
    z_gap_list: list[np.ndarray] = []
    field_list: list[np.ndarray] = []
    tgt_lists: dict[str, list[np.ndarray]] = {name: [] for name in ALL_OBSERVABLES}
    cid_list: list[np.ndarray] = []
    enc_list: list[np.ndarray] = []
    frame_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []

    for i, (cid, k) in enumerate(encounters):
        path = cache_dir / cid / f"encounter_{int(k):02d}.h5"
        with h5py.File(path, "r") as g:
            omega_raw = np.asarray(g["omega_z"], dtype=np.float32)
            cl = np.asarray(g["C_L"], dtype=np.float64)
            cd = np.asarray(g["C_D"], dtype=np.float64)
        n_frames = omega_raw.shape[0]
        omega_clean = pipe.preprocess_raw(omega_raw, cid, int(k))
        omega_norm = np.asarray(pipe.normalize(omega_clean), dtype=np.float32)

        z_sp = _encode_frames(frozen.encoder, omega_norm, device, frame_chunk, use_bf16)
        if getattr(frozen.encoder, "latent_mode", "spatial") == "pooled":
            # Pooled encoder (jepa_pool ablation) emits (T, d); lift to the
            # spatial grid by the SAME parameter-free broadcast used at train time
            # (src.probes.PooledToSpatialAdapter) so every downstream Q1/Q2 helper
            # sees a (T, d, h, w) latent. GAP of the constant map recovers pooled.
            fh, fw = frozen.latent_grid
            z_sp = np.ascontiguousarray(
                np.broadcast_to(z_sp[:, :, None, None], z_sp.shape + (fh, fw)),
                dtype=np.float32,
            )
        z_gap = z_sp.mean(axis=(2, 3))

        phys = physical_observables(omega_clean)
        entry = _window_entry(windows, cid, int(k))
        mask = frame_window_mask_for_entry(entry, n_frames)

        z_sp_list.append(z_sp.astype(np.float32))
        z_gap_list.append(z_gap.astype(np.float32))
        field_list.append(omega_norm)
        tgt_lists["C_L"].append(cl)
        tgt_lists["C_D"].append(cd)
        for name in FIELD_OBSERVABLES:
            tgt_lists[name].append(phys[name])
        cid_list.append(np.array([cid] * n_frames, dtype=object))
        enc_list.append(np.full(n_frames, int(k), dtype=np.int64))
        frame_list.append(np.arange(n_frames, dtype=np.int64))
        mask_list.append(mask)

        if verbose and ((i + 1) % 25 == 0 or i == len(encounters) - 1):
            print(
                f"[encode {frozen.name}/{split}] {i + 1}/{len(encounters)} encounters",
                flush=True,
            )

    targets = {name: np.concatenate(tgt_lists[name]) for name in ALL_OBSERVABLES}
    return EncodedSplit(
        z_spatial=np.concatenate(z_sp_list, axis=0),
        z_gap=np.concatenate(z_gap_list, axis=0),
        omega_norm=np.concatenate(field_list, axis=0),
        targets=targets,
        case_id=np.concatenate(cid_list).astype(str),
        encounter_index=np.concatenate(enc_list),
        frame=np.concatenate(frame_list),
        window_mask=np.concatenate(mask_list),
        latent_grid=frozen.latent_grid,
        split=split,
        model_name=frozen.name,
    )


# --------------------------------------------------------------------------- npz cache
def save_latents(path: str | Path, enc: EncodedSplit, *, save_field: bool = False) -> Path:
    """Persist the per-model latents + alignment (and optionally the field) to npz.

    The spatial + GAP latents are model-specific and are what Q2/Q3 reuse; the
    field is model-independent so it is written once elsewhere and omitted here by
    default (``save_field=True`` stores it as float16 for a self-contained cache).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "z_spatial": enc.z_spatial.astype(np.float32),
        "z_gap": enc.z_gap.astype(np.float32),
        "case_id": enc.case_id.astype(str),
        "encounter_index": enc.encounter_index,
        "frame": enc.frame,
        "window_mask": enc.window_mask,
        "latent_grid": np.asarray(enc.latent_grid, dtype=np.int64),
        "split": np.asarray(enc.split),
        "model_name": np.asarray(enc.model_name),
    }
    for name, arr in enc.targets.items():
        payload[f"target_{name}"] = arr
    if save_field:
        payload["omega_norm"] = enc.omega_norm.astype(np.float16)
    np.savez(out, **payload)
    return out


def save_field_cache(path: str | Path, enc: EncodedSplit) -> Path:
    """Persist the model-independent normalised field + targets + alignment once."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "omega_norm": enc.omega_norm.astype(np.float16),
        "case_id": enc.case_id.astype(str),
        "encounter_index": enc.encounter_index,
        "frame": enc.frame,
        "window_mask": enc.window_mask,
    }
    for name, arr in enc.targets.items():
        payload[f"target_{name}"] = arr
    np.savez(out, **payload)
    return out


def load_windows(path: str | Path) -> dict:
    """Load a Track 0.B windows json and return its ``windows`` mapping."""
    with open(_resolve(path)) as fh:
        payload = json.load(fh)
    return payload.get("windows", payload)


def iter_models(base: str | Path, names: Iterable[str]) -> list[tuple[str, Path]]:
    """Resolve ``(name, run_dir)`` pairs under a session-31 runs base directory."""
    base_path = _resolve(base)
    return [(name, base_path / name) for name in names]

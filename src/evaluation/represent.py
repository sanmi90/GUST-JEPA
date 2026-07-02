"""Session 31 Track D Q1: representation eval over the six canonical models.

Two families of numbers per model, both on the frozen encoder:

DECODE FLOOR (the central D-N3 spatial-latent verdict)
    Fit the identical-capacity :class:`~src.models.decoder.SpatialLatentFieldDecoder`
    (the module inside :class:`src.probes.DecoderProbe`) on TRAIN latents ->
    pipeline-normalised omega, then score Test B. Report field VRMSE (aggregated
    numerator/denominator over the whole eval set BEFORE dividing) and per-frame
    SSIM (Wang K1=0.01, K2=0.03, ``L = ssim_data_range('v2p2') = 8.487``, the
    project convention used by the Session 27 decode work).

REPRESENTATION PROBES
    For each sourced state observable (C_L, C_D, circ_pos, circ_neg,
    wake_enstrophy) fit a LINEAR (RidgeCV) and a small MLP probe on TRAIN
    per-frame GAP latents -> target and report the Test B R^2. Per-frame z is the
    correct regime for state-descriptor targets (CLAUDE.md probe methodology).

Both the decode-floor decoder and the probes are fit on ALL train frames; the
``windowed`` scope restricts only the Test B evaluation to the Track 0.B
impact + relaxation frames, so it is directly comparable to Q2/Q3 scope while the
learned capacity stays identical across scopes.

The pure metric helpers (:func:`aggregated_vrmse`, :func:`r2_score_np`,
:func:`frame_window_mask`, :func:`select_window`) are unit-tested in
``tests/test_rom_eval.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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

# Session 31 Track E one-axis ablations (evaluated with the same frozen-probe
# harness). st_d64 is a d=64 spatio-temporal encoder; jepa_pool is a pooled
# latent. Both flow through the shared Q1/Q2 helpers unchanged (the pooled latent
# is broadcast to the spatial grid in encode_split; d flows from the checkpoint).
ABLATION_MODELS = (
    "jepa_cnn",
    "ae_cnn",
    "st_d64",
    "jepa_pool",
    "jepa_vicreg",
)


# --------------------------------------------------------------------------- pure math
def aggregated_vrmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Variance-normalised RMSE, aggregated over the whole eval set before dividing.

    ``sqrt( sum(pred - true)^2 / sum(true - mean_true)^2 )`` where ``mean_true`` is
    the single global mean of ``true`` over all rows and cells. Aggregating the
    numerator and denominator over the whole set (rather than averaging per-sample
    ratios) keeps the metric bounded on near-uniform frames.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs true {true.shape}")
    mean_true = float(true.mean())
    num = float(((pred - true) ** 2).sum())
    den = float(((true - mean_true) ** 2).sum())
    if den <= 0.0:
        return float("nan")
    return float(np.sqrt(num / den))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Held-out R^2 = 1 - SSE/SST with SST about the eval set's own mean."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    sse = float(((y_true - y_pred) ** 2).sum())
    if sst <= 0.0:
        return float("nan")
    return float(1.0 - sse / sst)


def frame_window_mask(
    entry: dict | None, n_frames: int, parts: Sequence[str] = ("impact", "relaxation")
) -> np.ndarray:
    """Per-frame boolean mask that is True inside any of ``parts`` half-open ranges.

    Args:
        entry: A Track 0.B windows entry with ``impact`` / ``relaxation`` = ``[lo, hi)``
            frame bounds, or ``None`` (returns an all-False mask).
        n_frames: Number of frames in the encounter.
        parts: Window names to union (default impact + relaxation).
    """
    mask = np.zeros(int(n_frames), dtype=bool)
    if entry is None:
        return mask
    for part in parts:
        rng = entry.get(part)
        if rng is None:
            continue
        lo = max(0, int(rng[0]))
        hi = min(int(n_frames), int(rng[1]))
        if hi > lo:
            mask[lo:hi] = True
    return mask


def select_window(z: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-select ``z`` and ``y`` by a per-row boolean mask, keeping alignment."""
    z = np.asarray(z)
    y = np.asarray(y)
    mask = np.asarray(mask, dtype=bool)
    if not (z.shape[0] == y.shape[0] == mask.shape[0]):
        raise ValueError(f"row mismatch: z {z.shape[0]}, y {y.shape[0]}, mask {mask.shape[0]}")
    return z[mask], y[mask]


def frame_ssim(pred: np.ndarray, true: np.ndarray, data_range: float) -> float:
    """Mean per-frame SSIM (skimage default Wang K1=0.01/K2=0.03) over ``[N, H, W]``.

    Matches the Session 27 decode convention: 2D SSIM per normalised-omega frame
    with an explicit ``data_range = L`` (see ``ssim_data_range``).
    """
    from skimage.metrics import structural_similarity

    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    vals = [
        float(structural_similarity(true[i], pred[i], data_range=float(data_range)))
        for i in range(pred.shape[0])
    ]
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- probes
def fit_linear_probe(x_tr: np.ndarray, y_tr: np.ndarray):
    """StandardScaler -> RidgeCV pipeline (alpha CV-selected), target-scaled."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    alphas = np.logspace(-3, 4, 24)
    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    model = TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())
    model.fit(x_tr, y_tr)
    return model


def fit_mlp_probe(x_tr: np.ndarray, y_tr: np.ndarray, seed: int = 0):
    """StandardScaler -> small MLPRegressor pipeline, target-scaled (nonlinear probe)."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        alpha=1e-3,
        max_iter=800,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=seed,
    )
    reg = make_pipeline(StandardScaler(), mlp)
    model = TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())
    model.fit(x_tr, y_tr)
    return model


def probe_r2(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ev: np.ndarray,
    y_ev: np.ndarray,
    ev_mask: np.ndarray | None,
    seed: int = 0,
) -> dict[str, float]:
    """Fit linear + MLP probes on ALL train rows; eval R^2 on Test B (optionally windowed)."""
    lin = fit_linear_probe(x_tr, y_tr)
    mlp = fit_mlp_probe(x_tr, y_tr, seed=seed)
    xe, ye = (x_ev, y_ev) if ev_mask is None else select_window(x_ev, y_ev, ev_mask)
    return {
        "linear_r2": r2_score_np(ye, lin.predict(xe)),
        "mlp_r2": r2_score_np(ye, mlp.predict(xe)),
        "n_eval": int(ye.shape[0]),
    }


# --------------------------------------------------------------------------- decode floor
@dataclass
class DecodeFloorResult:
    vrmse: float
    ssim: float
    n: int


def fit_decode_floor_decoder(
    z_train: np.ndarray,
    field_train: np.ndarray,
    latent_grid: tuple[int, int],
    *,
    device,
    steps: int = 6000,
    batch: int = 64,
    lr: float = 5e-4,
    seed: int = 0,
    log_every: int = 1000,
    verbose: bool = True,
):
    """Train the shared SpatialLatentFieldDecoder on TRAIN latents -> normalised field.

    This is the identical-capacity field readout inside
    :class:`src.probes.DecoderProbe`; we train it directly on cached latents (the
    encoder is frozen and already applied) so no encoder forward runs per step.
    """
    import torch
    from torch.optim import AdamW

    from src.models.decoder import SpatialLatentFieldDecoder

    torch.manual_seed(seed)
    fh, fw = int(latent_grid[0]), int(latent_grid[1])
    decoder = SpatialLatentFieldDecoder(
        latent_dim=int(z_train.shape[1]), feature_h=fh, feature_w=fw
    ).to(device)
    decoder.train()
    opt = AdamW(decoder.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    z_t = torch.from_numpy(np.ascontiguousarray(z_train, dtype=np.float32))
    f_t = torch.from_numpy(np.ascontiguousarray(field_train, dtype=np.float32))
    n = z_t.shape[0]
    g = torch.Generator().manual_seed(seed)
    use_cuda = getattr(device, "type", str(device)) == "cuda"

    for step in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        zb = z_t[idx].to(device)  # (B, d, h, w)
        fb = f_t[idx].to(device).unsqueeze(1)  # (B, 1, H, W)
        if use_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = decoder(zb)  # (B, 1, H, W)
                loss = torch.nn.functional.mse_loss(pred.float(), fb)
        else:
            pred = decoder(zb)
            loss = torch.nn.functional.mse_loss(pred, fb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()
        if verbose and (step % log_every == 0 or step == steps - 1):
            print(f"    [decode-floor] step {step}/{steps} mse={loss.item():.5f}", flush=True)

    decoder.eval()
    return decoder


def decode_fields(decoder, z_eval: np.ndarray, device, chunk: int = 256) -> np.ndarray:
    """Decode ``(N, d, h, w)`` latents -> ``(N, 192, 96)`` normalised fields (float32)."""
    import torch

    use_cuda = getattr(device, "type", str(device)) == "cuda"
    out: list[np.ndarray] = []
    z_t = torch.from_numpy(np.ascontiguousarray(z_eval, dtype=np.float32))
    with torch.no_grad():
        for s in range(0, z_t.shape[0], chunk):
            end = s + chunk
            zb = z_t[s:end].to(device)  # (B, d, h, w)
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = decoder(zb)
            else:
                pred = decoder(zb)
            out.append(pred.float().squeeze(1).cpu().numpy())  # (B, H, W)
    return np.concatenate(out, axis=0)


def decode_floor(
    z_train: np.ndarray,
    field_train: np.ndarray,
    z_eval: np.ndarray,
    field_eval: np.ndarray,
    latent_grid: tuple[int, int],
    ssim_L: float,
    *,
    device,
    eval_mask: np.ndarray | None = None,
    steps: int = 6000,
    batch: int = 64,
    seed: int = 0,
    verbose: bool = True,
    decoder=None,
) -> tuple[DecodeFloorResult, object]:
    """Fit (or reuse) the decode-floor decoder and score VRMSE + SSIM on Test B.

    Returns ``(result, decoder)`` so the caller can reuse the trained decoder for a
    second (windowed) eval scope without retraining.
    """
    if decoder is None:
        decoder = fit_decode_floor_decoder(
            z_train,
            field_train,
            latent_grid,
            device=device,
            steps=steps,
            batch=batch,
            seed=seed,
            verbose=verbose,
        )
    pred = decode_fields(decoder, z_eval, device)  # (N, 192, 96)
    true = np.asarray(field_eval, dtype=np.float32)
    if eval_mask is not None:
        pred = pred[eval_mask]
        true = true[eval_mask]
    result = DecodeFloorResult(
        vrmse=aggregated_vrmse(pred, true),
        ssim=frame_ssim(pred, true, ssim_L),
        n=int(true.shape[0]),
    )
    return result, decoder


# --------------------------------------------------------------------------- runner
def run_q1(
    models: Sequence[str] = CANONICAL_MODELS,
    *,
    runs_base: str | Path = "outputs/runs/session31",
    checkpoint_name: str = "checkpoint_iter010000.pt",
    partition: str = "v2p2",
    split_manifest: str | Path = "configs/splits/split_v2p2.json",
    pipeline_manifest: str | Path = "outputs/data_pipeline/v2p2/manifest.json",
    windows_path: str | Path = "outputs/session31/windows_v2p2.json",
    out_json: str | Path = "outputs/session31/q1_representation.json",
    cache_dir: str | Path = "outputs/session31/q1_latents",
    gpu: int = 0,
    device_override: str | None = None,
    decoder_steps: int = 6000,
    decoder_batch: int = 64,
    limit: int | None = None,
    targets: Sequence[str] | None = None,
    frame_chunk: int = 40,
) -> dict:
    """Encode all six models, fit the decode floor + probes, write q1_representation.json."""
    import torch

    from src.data.omega_pipeline import ssim_data_range
    from src.evaluation import rom_eval as re

    if device_override is not None:
        device = torch.device(device_override)
        gpu_name = device_override
    else:
        from src.utils.device import require_rtx6000

        device = require_rtx6000(gpu_index=gpu)
        gpu_name = torch.cuda.get_device_name(device.index)
        if "RTX" not in gpu_name or "6000" not in gpu_name:
            raise RuntimeError(f"hardware policy: gpu_name={gpu_name!r} is not an RTX 6000")

    if targets is None:
        targets = re.ALL_OBSERVABLES
    ssim_L = float(ssim_data_range(str(re._resolve(pipeline_manifest))))
    windows = re.load_windows(windows_path)
    cache_dir = re._resolve(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[q1] device={device} ({gpu_name}) ssim_L={ssim_L:.3f} "
        f"targets={list(targets)} models={list(models)}",
        flush=True,
    )

    results: dict[str, dict] = {}
    fields_saved = False

    for name in models:
        run_dir = re._resolve(runs_base) / name
        print(f"\n[q1] === {name} ===", flush=True)
        # References (fukami / fukami_wake / pod) use their OWN architectures and
        # load through a dedicated path that yields a pooled-latent FrozenModel;
        # the shared harness is otherwise byte-identical (encode_split broadcasts
        # the pooled latent to the spatial grid, as for the jepa_pool ablation).
        # Suffixed variants (e.g. fukami_wake_d16, Session 33 min-d panel) route
        # through the same reference loader via the prefix match.
        if any(name == m or name.startswith(m + "_") for m in re.REFERENCE_MODELS):
            frozen = re.load_reference_model(run_dir, checkpoint_name, device=device)
        else:
            frozen = re.load_frozen_model(run_dir, checkpoint_name, device=device)

        enc_tr = re.encode_split(
            frozen,
            "train",
            partition=partition,
            split_manifest_path=split_manifest,
            pipeline_manifest=pipeline_manifest,
            windows=windows,
            device=device,
            limit=limit,
            frame_chunk=frame_chunk,
        )
        enc_tb = re.encode_split(
            frozen,
            "test_b",
            partition=partition,
            split_manifest_path=split_manifest,
            pipeline_manifest=pipeline_manifest,
            windows=windows,
            device=device,
            limit=limit,
            frame_chunk=frame_chunk,
        )
        re.save_latents(cache_dir / f"latents_{name}_train.npz", enc_tr)
        re.save_latents(cache_dir / f"latents_{name}_test_b.npz", enc_tb)
        # The normalised-field cache is model-independent (same encounter
        # enumeration for every model), so write it once and never overwrite an
        # existing one. This lets a reference-only Q1 run reuse the canonical
        # fields without touching them, keeping Q2/Q3 row-alignment intact.
        if not fields_saved:
            if not (cache_dir / "fields_train.npz").exists():
                re.save_field_cache(cache_dir / "fields_train.npz", enc_tr)
            if not (cache_dir / "fields_test_b.npz").exists():
                re.save_field_cache(cache_dir / "fields_test_b.npz", enc_tb)
            fields_saved = True

        # --- decode floor (fit once on all train frames; eval all + windowed) ----
        floor_all, dec = decode_floor(
            enc_tr.z_spatial,
            enc_tr.omega_norm,
            enc_tb.z_spatial,
            enc_tb.omega_norm,
            enc_tr.latent_grid,
            ssim_L,
            device=device,
            eval_mask=None,
            steps=decoder_steps,
            batch=decoder_batch,
            verbose=True,
        )
        floor_win, _ = decode_floor(
            enc_tr.z_spatial,
            enc_tr.omega_norm,
            enc_tb.z_spatial,
            enc_tb.omega_norm,
            enc_tr.latent_grid,
            ssim_L,
            device=device,
            eval_mask=enc_tb.window_mask,
            decoder=dec,
            verbose=False,
        )
        del dec
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # --- representation probes (per-frame GAP latents) -----------------------
        probes_all: dict[str, dict] = {}
        probes_win: dict[str, dict] = {}
        for tname in targets:
            y_tr = enc_tr.targets[tname]
            y_tb = enc_tb.targets[tname]
            probes_all[tname] = probe_r2(enc_tr.z_gap, y_tr, enc_tb.z_gap, y_tb, None)
            probes_win[tname] = probe_r2(enc_tr.z_gap, y_tr, enc_tb.z_gap, y_tb, enc_tb.window_mask)

        results[name] = {
            "objective": frozen.cfg.objective,
            "active_terms": sorted(frozen.cfg.active_terms()),
            "decode_floor": {
                "all_frames": {"vrmse": floor_all.vrmse, "ssim": floor_all.ssim, "n": floor_all.n},
                "windowed": {"vrmse": floor_win.vrmse, "ssim": floor_win.ssim, "n": floor_win.n},
            },
            "probes": {
                "all_frames": probes_all,
                "windowed": probes_win,
            },
        }
        _print_model_row(name, results[name])
        del frozen, enc_tr, enc_tb
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "task": "SESSION 31 Track D Q1 -- ROM representation eval",
        "params": {
            "partition": partition,
            "split_manifest": str(split_manifest),
            "pipeline_manifest": str(pipeline_manifest),
            "checkpoint": checkpoint_name,
            "gpu_name": gpu_name,
            "ssim_data_range_L": ssim_L,
            "ssim_convention": (
                "Wang K1=0.01 K2=0.03 on pipeline-normalised omega (skimage 2D SSIM)"
            ),
            "vrmse_definition": (
                "sqrt(sum(pred-true)^2 / sum(true-mean_true)^2), aggregated over "
                "eval set before dividing"
            ),
            "window_definition": (
                "impact union relaxation frames from Track 0.B windows_v2p2.json; "
                "windowed scope restricts Test B evaluation only"
            ),
            "decode_floor_decoder": (
                "SpatialLatentFieldDecoder (identical capacity, the DecoderProbe "
                "module), fit on all TRAIN frames"
            ),
            "decoder_steps": decoder_steps,
            "decoder_batch": decoder_batch,
            "probe_regime": (
                "per-frame GAP latent -> observable (state-descriptor regime, CLAUDE.md)"
            ),
            "probe_linear": "StandardScaler -> RidgeCV (alpha CV) -> target-scaled",
            "probe_mlp": "StandardScaler -> MLPRegressor(64,64) -> target-scaled",
            "targets_used": list(targets),
            "target_sources": {
                "C_L": "cache",
                "C_D": "cache",
                "wake_enstrophy": "computed on preprocess_raw omega (session17 wake-box)",
                "circulation_pos": (
                    "computed on preprocess_raw omega (session17 wake-box, |omega|>1)"
                ),
                "circulation_neg": (
                    "computed on preprocess_raw omega (session17 wake-box, |omega|>1)"
                ),
            },
        },
        "models": results,
    }
    out_path = re._resolve(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[q1] wrote {out_path}", flush=True)
    _print_summary(payload)
    return payload


def _print_model_row(name: str, r: dict) -> None:
    fa = r["decode_floor"]["all_frames"]
    print(
        f"[q1] {name}: decode-floor(all) VRMSE={fa['vrmse']:.3f} SSIM={fa['ssim']:.3f} "
        f"| C_L lin/mlp={r['probes']['all_frames']['C_L']['linear_r2']:.3f}/"
        f"{r['probes']['all_frames']['C_L']['mlp_r2']:.3f}",
        flush=True,
    )


def _print_summary(payload: dict) -> None:
    tgts = payload["params"]["targets_used"]
    abbr = {
        "C_L": "C_L",
        "C_D": "C_D",
        "wake_enstrophy": "wakeEns",
        "circulation_pos": "circPos",
        "circulation_neg": "circNeg",
    }
    print(
        "\n[q1] ============ Q1 SUMMARY: decode floor + probe R^2 (all-frames Test B) ============"
    )
    print("[q1] probe cells are linear_R2 / mlp_R2")
    header = f"{'model':<16}{'VRMSE':>8}{'SSIM':>7}  " + "".join(
        f"{abbr.get(t, t) + '(L/M)':>18}" for t in tgts
    )
    print(header)
    for name, r in payload["models"].items():
        fa = r["decode_floor"]["all_frames"]
        cells = ""
        for t in tgts:
            p = r["probes"]["all_frames"][t]
            cells += f"{p['linear_r2']:>8.2f}/{p['mlp_r2']:<8.2f}"
        print(f"{name:<16}{fa['vrmse']:>8.3f}{fa['ssim']:>7.3f}  {cells}")
    print("[q1] ================================================================================\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 31 Track D Q1 representation eval")
    p.add_argument("--models", nargs="+", default=list(CANONICAL_MODELS))
    p.add_argument("--runs-base", default="outputs/runs/session31")
    p.add_argument("--checkpoint", default="checkpoint_iter010000.pt")
    p.add_argument("--partition", default="v2p2")
    p.add_argument("--split", default="configs/splits/split_v2p2.json")
    p.add_argument("--pipeline-manifest", default="outputs/data_pipeline/v2p2/manifest.json")
    p.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    p.add_argument("--out", default="outputs/session31/q1_representation.json")
    p.add_argument("--cache-dir", default="outputs/session31/q1_latents")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None, help="Explicit device (bypasses require_rtx6000).")
    p.add_argument("--decoder-steps", type=int, default=6000)
    p.add_argument("--decoder-batch", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="Debug: cap encounters per split.")
    p.add_argument("--targets", nargs="+", default=None)
    p.add_argument(
        "--frame-chunk",
        type=int,
        default=40,
        help=(
            "Frames encoded per forward. Irrelevant for per-frame encoders; set "
            ">= n_frames (e.g. 200) for the cnn_vit_temporal ablation so its causal "
            "window is never truncated at a chunk boundary."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device is None and os.environ.get("VORTEX_JEPA_ROM_DEVICE"):
        args.device = os.environ["VORTEX_JEPA_ROM_DEVICE"]
    run_q1(
        models=args.models,
        runs_base=args.runs_base,
        checkpoint_name=args.checkpoint,
        partition=args.partition,
        split_manifest=args.split,
        pipeline_manifest=args.pipeline_manifest,
        windows_path=args.windows,
        out_json=args.out,
        cache_dir=args.cache_dir,
        gpu=args.gpu,
        device_override=args.device,
        decoder_steps=args.decoder_steps,
        decoder_batch=args.decoder_batch,
        limit=args.limit,
        targets=args.targets,
        frame_chunk=args.frame_chunk,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Session 16, Experiment 3: pixel-level gradient-SHAP attribution.

For each (target, encounter) pair we compute integrated gradients of the
probe-predicted target with respect to the input omega field at the impact
frame. Integration is over 32 straight-line steps from a phase-matched
baseline (mean of Baseline.h5 encounters 0-3 at the same frame) to the
actual omega field.

Pixel attribution map shape: (192, 96), one per (target, encounter).

Targets chosen from Exp 2's test_b R^2 ranking:
    centroid_x      (R^2 = 0.92)  -- wake position
    circulation_pos (R^2 = 0.91)  -- total positive vorticity
    peak_neg_omega  (R^2 = 0.87)  -- localised negative-vortex strength

Probe definition matches scripts/session16/exp2_probe_sweep.py exactly.
We retrain the 3 probes here (small cost ~15s) rather than carrying weight
checkpoints across scripts.

Output:
    outputs/session16/exp3/shap_attribution.npz
        per-target, per-encounter attribution maps + impact frame + (G,D,Y)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor, nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


ENCODER_CKPT = REPO / "outputs" / "runs" / "session12" / "S12_E_d64" / "encoder" / "checkpoint_iter020000.pt"
OMEGA_MANIFEST = REPO / "outputs" / "data_pipeline" / "v1" / "manifest.json"
SPLIT_MANIFEST = REPO / "configs" / "splits" / "split_v1.json"
TARGETS_DIR = REPO / "outputs" / "session16" / "exp2" / "per_frame_targets"
OUT = REPO / "outputs" / "session16" / "exp3"
OUT.mkdir(parents=True, exist_ok=True)
PARTITION = "v1"
DEFAULT_IMPACT_FRAME = 40
CACHE_ROOT = Path(
    os.environ.get(
        "VORTEX_JEPA_CACHE",
        str(Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
            / "data" / "processed" / "vortex-jepa"),
    )
)

PROBE_TARGETS = ("centroid_x", "circulation_pos", "peak_neg_omega")
N_INTEGRATION_STEPS = 32


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def load_encoder(device: torch.device) -> HybridCNNViTEncoder:
    blob = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=False)
    args = blob["args"]
    enc = HybridCNNViTEncoder(
        latent_dim=int(args["d"]),
        projection_norm=args.get("projection_norm", "batchnorm"),
    )
    state = {
        k.removeprefix("encoder."): v
        for k, v in blob["jepa_state_dict"].items() if k.startswith("encoder.")
    }
    enc.load_state_dict(state, strict=False)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def train_probes(
    target_names: tuple[str, ...],
    train_data: dict,
    device: torch.device,
    n_iters: int = 2000,
    batch_size: int = 64,
    lr: float = 3e-4,
    seed: int = 0,
) -> dict[str, tuple[MLPProbe, float, float]]:
    """Match the Exp 2 probe training procedure."""
    probes: dict[str, tuple[MLPProbe, float, float]] = {}
    z_full = train_data["z_full"]
    n_enc, T_max, d = z_full.shape
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    for target_name in target_names:
        y_train_all = train_data[target_name]
        finite_mask = np.isfinite(y_train_all)
        y_finite = y_train_all[finite_mask]
        mean = float(y_finite.mean())
        std = float(y_finite.std()) or 1.0
        y_norm = (y_train_all - mean) / std

        probe = MLPProbe(in_dim=d).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
        candidate_encs = np.where(finite_mask.any(axis=1))[0]
        probe.train()
        for _ in range(n_iters):
            chosen = rng.choice(candidate_encs, size=min(batch_size, candidate_encs.size), replace=False)
            z_batch = np.empty((chosen.size, d), dtype=np.float32)
            y_batch = np.empty(chosen.size, dtype=np.float32)
            for j, enc_idx in enumerate(chosen):
                ok = np.where(finite_mask[enc_idx])[0]
                t = int(rng.choice(ok))
                z_batch[j] = z_full[enc_idx, t]
                y_batch[j] = y_norm[enc_idx, t]
            z_t = torch.from_numpy(z_batch).to(device)
            y_t = torch.from_numpy(y_batch).to(device)
            opt.zero_grad()
            loss = ((probe(z_t) - y_t) ** 2).mean()
            loss.backward()
            opt.step()
        probe.eval()
        for p in probe.parameters():
            p.requires_grad_(False)
        probes[target_name] = (probe, mean, std)
        print(f"[exp3] probe '{target_name}' trained: target_mean={mean:.3f} target_std={std:.3f}")
    return probes


def load_split_data(name: str) -> dict:
    d = np.load(TARGETS_DIR / f"{name}.npz", allow_pickle=True)
    return {
        "z_full": d["z_full"].astype(np.float32),
        "G": d["G"].astype(np.float32),
        "D": d["D"].astype(np.float32),
        "Y": d["Y"].astype(np.float32),
        "case_id": d["case_id"],
        "encounter_index": d["encounter_index"].astype(np.int32),
        "impact_frame": d["impact_frame"].astype(np.int32),
        **{k: d[k].astype(np.float32) for k in (
            "centroid_x", "circulation_pos", "peak_neg_omega",
            "C_L", "C_D", "wake_enstrophy",
        )},
    }


def gather_split_encounters(split: str) -> list[dict]:
    with open(SPLIT_MANIFEST) as f:
        manifest = json.load(f)
    out: list[dict] = []
    for cid, case in manifest["cases"].items():
        if split == "test_a" and case["split"] == "train":
            ks = case["test_a_encounter_indices"]
        elif split == "test_b" and case["split"] == "test_b":
            ks = list(range(int(case["n_encounters_full"])))
        elif split == "test_c" and case["split"] == "test_c":
            ks = list(range(int(case["n_encounters_full"])))
        elif split == "baseline_pool":
            if cid != "Baseline":
                continue
            ks = list(range(4))  # train encounters 0..3 per spec
        else:
            continue
        for k in ks:
            path = CACHE_ROOT / PARTITION / cid / f"encounter_{int(k):02d}.h5"
            if not path.exists():
                continue
            out.append({
                "case_id": cid, "k": int(k), "path": path,
                "G": float(case.get("G", 0.0)),
                "D": float(case.get("D", 0.0)),
                "Y": float(case.get("Y", 0.0)),
            })
    return out


def load_omega_normalized(
    encounter: dict, pipeline: OmegaPipeline
) -> tuple[np.ndarray, int]:
    with h5py.File(encounter["path"], "r") as f:
        omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
        impact = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
    omega_clean = pipeline.preprocess_raw(omega_raw, encounter["case_id"], int(encounter["k"]))
    omega_norm = pipeline.normalize(omega_clean).astype(np.float32)
    return omega_norm, impact


def integrated_gradients(
    enc: HybridCNNViTEncoder,
    probe: MLPProbe,
    omega_baseline: Tensor,
    omega_input: Tensor,
    n_steps: int,
    device: torch.device,
) -> Tensor:
    """Pixel-level integrated gradients of probe(enc(omega)) wrt omega.

    omega_baseline and omega_input: (192, 96) tensors in normalised space.
    Returns (192, 96) attribution.
    """
    alphas = torch.linspace(0.0, 1.0, n_steps, device=device)
    diff = (omega_input - omega_baseline).to(device)
    grads_accum = torch.zeros_like(diff)
    for alpha in alphas:
        x = omega_baseline + alpha * diff
        x = x.detach().requires_grad_(True)
        x_in = x.view(1, 1, 1, x.shape[0], x.shape[1])  # (B=1, T=1, C=1, H, W)
        z = enc(x_in)  # (1, 1, d) bf16 or fp32 depending on autocast off
        z_imp = z[:, 0, :]
        y = probe(z_imp)
        y.sum().backward()
        grads_accum = grads_accum + x.grad.detach()
    avg_grad = grads_accum / n_steps
    return (avg_grad * diff).detach()


@torch.no_grad()
def compute_phase_matched_baseline(
    baseline_encs: list[dict], pipeline: OmegaPipeline, device: torch.device
) -> dict:
    """Mean omega across baseline encounters, per frame."""
    arrs = []
    impact_frames = []
    for e in baseline_encs:
        o, imp = load_omega_normalized(e, pipeline)
        arrs.append(o)
        impact_frames.append(imp)
    arrs = np.stack(arrs, axis=0)  # (n_baseline, T, H, W)
    return {
        "mean_per_frame": arrs.mean(axis=0),
        "individual": arrs,
        "impact_frames": np.array(impact_frames),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--splits", nargs="+", default=["test_b", "test_c"])
    p.add_argument("--n-steps", type=int, default=N_INTEGRATION_STEPS)
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[exp3] device={device}")

    enc = load_encoder(device)
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)

    train_data = load_split_data("train")
    probes = train_probes(PROBE_TARGETS, train_data, device)

    baseline_encs = gather_split_encounters("baseline_pool")
    print(f"[exp3] baseline pool size = {len(baseline_encs)}")
    baseline = compute_phase_matched_baseline(baseline_encs, pipeline, device)
    print(f"[exp3] phase-matched baseline shape = {baseline['mean_per_frame'].shape}")

    out_artefacts: dict = {
        "baseline_mean_per_frame": baseline["mean_per_frame"],
        "baseline_individual": baseline["individual"],
        "baseline_impact_frames": baseline["impact_frames"],
        "probe_targets": np.array(list(PROBE_TARGETS), dtype=object),
        "n_integration_steps": np.array(args.n_steps),
    }
    summary_records: list[dict] = []

    t0 = time.time()
    for split in args.splits:
        encs = gather_split_encounters(split)
        print(f"\n[exp3] split={split}: {len(encs)} encounters")
        attribution_maps = {tg: np.zeros((len(encs), 192, 96), dtype=np.float32) for tg in PROBE_TARGETS}
        target_pred = {tg: np.zeros(len(encs), dtype=np.float32) for tg in PROBE_TARGETS}
        target_baseline_pred = {tg: np.zeros(len(encs), dtype=np.float32) for tg in PROBE_TARGETS}
        case_ids_split = []
        encounter_indices_split = []
        Gs, Ds, Ys, impact_frames_split = [], [], [], []
        for i, e in enumerate(encs):
            omega_norm, impact = load_omega_normalized(e, pipeline)
            T_actual = omega_norm.shape[0]
            if impact >= T_actual:
                impact = T_actual - 1
            omega_input = torch.from_numpy(omega_norm[impact]).to(device)
            baseline_frame_idx = min(impact, baseline["mean_per_frame"].shape[0] - 1)
            omega_baseline = torch.from_numpy(baseline["mean_per_frame"][baseline_frame_idx]).to(device)

            for target_name in PROBE_TARGETS:
                probe, t_mean, t_std = probes[target_name]
                attr = integrated_gradients(
                    enc, probe, omega_baseline, omega_input,
                    n_steps=args.n_steps, device=device,
                )
                attribution_maps[target_name][i] = attr.cpu().numpy()

                with torch.no_grad():
                    x_in = omega_input.view(1, 1, 1, 192, 96)
                    z = enc(x_in)
                    pred_norm = probe(z[:, 0, :])
                    pred_real = float(pred_norm.cpu().item()) * t_std + t_mean
                    target_pred[target_name][i] = pred_real

                    x_b = omega_baseline.view(1, 1, 1, 192, 96)
                    z_b = enc(x_b)
                    pred_b_norm = probe(z_b[:, 0, :])
                    pred_b_real = float(pred_b_norm.cpu().item()) * t_std + t_mean
                    target_baseline_pred[target_name][i] = pred_b_real

            case_ids_split.append(e["case_id"])
            encounter_indices_split.append(e["k"])
            Gs.append(e["G"])
            Ds.append(e["D"])
            Ys.append(e["Y"])
            impact_frames_split.append(int(impact))
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(encs)} encounters ({time.time() - t0:.1f}s)")

        for target_name in PROBE_TARGETS:
            out_artefacts[f"{split}_{target_name}_attr"] = attribution_maps[target_name]
            out_artefacts[f"{split}_{target_name}_pred"] = target_pred[target_name]
            out_artefacts[f"{split}_{target_name}_pred_baseline"] = target_baseline_pred[target_name]
        out_artefacts[f"{split}_case_id"] = np.array(case_ids_split, dtype=object)
        out_artefacts[f"{split}_encounter_index"] = np.array(encounter_indices_split, dtype=np.int32)
        out_artefacts[f"{split}_G"] = np.array(Gs, dtype=np.float32)
        out_artefacts[f"{split}_D"] = np.array(Ds, dtype=np.float32)
        out_artefacts[f"{split}_Y"] = np.array(Ys, dtype=np.float32)
        out_artefacts[f"{split}_impact_frame"] = np.array(impact_frames_split, dtype=np.int32)
        summary_records.append({
            "split": split, "n_encounters": len(encs),
            "targets": list(PROBE_TARGETS),
        })

    save_npz = OUT / "shap_attribution.npz"
    np.savez_compressed(save_npz, **out_artefacts)
    print(f"[exp3] wrote {save_npz.relative_to(REPO)} ({save_npz.stat().st_size / 1e6:.1f} MB)")
    save_js = OUT / "shap_summary.json"
    save_js.write_text(json.dumps({
        "encoder_ckpt": str(ENCODER_CKPT.relative_to(REPO)),
        "baseline_source": "Baseline.h5 encounters 0..3, phase-matched mean per frame",
        "n_integration_steps": args.n_steps,
        "targets": list(PROBE_TARGETS),
        "splits": summary_records,
    }, indent=2))
    print(f"[exp3] wrote {save_js.relative_to(REPO)}")
    print(f"[exp3] total wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

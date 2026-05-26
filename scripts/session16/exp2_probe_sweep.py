"""Session 16, Experiment 2: probe sweep with prequential coding.

For each per-frame target (G, D, Y, C_L, C_D, plus 9 flow descriptors), train
a small MLP probe (3 hidden layers, width 256, ReLU) from the full 64-D
encoder latent z (no per-dimension splits, per the Day-1 finding that the
linear basis is seed-arbitrary). Log per-iteration loss to drive the
prequential coding estimator from src/evaluation/epiplexity.py.

IID SAMPLING (documented in the loop):
    Every iteration samples ``batch_size`` distinct encounter indices uniformly
    without replacement from the 180 train encounters, then picks ONE random
    frame in [0, 119] per encounter. This enforces the spec rule "at most one
    frame per (case, encounter) per epoch" exactly: any single iteration's
    batch contains 0-or-1 frames from each encounter, never two from the
    same one.

Evaluation: test_b and test_c with ALL 120 frames per encounter (held-out
cases, so no IID constraint needed for eval). R^2 is the variance-weighted
multi-output R^2 from sklearn.metrics.r2_score with multioutput='variance_weighted'
(scalar targets become a regular R^2).

Outputs:
    outputs/session16/exp2/probe_sweep.json
        per-target results: R^2 on every split + prequential coding +
        loss curve summary statistics
    outputs/session16/exp2/probe_loss_curves/{target}.npy
        full per-iteration train loss for posterity
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from sklearn.metrics import r2_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.evaluation.epiplexity import prequential_coding  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


TARGETS_DIR = REPO / "outputs" / "session16" / "exp2" / "per_frame_targets"
OUT = REPO / "outputs" / "session16" / "exp2"
LOSS_DIR = OUT / "probe_loss_curves"
LOSS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAMES = (
    "G", "D", "Y",
    "C_L", "C_D",
    "peak_pos_omega", "peak_neg_omega",
    "centroid_x", "centroid_y",
    "circulation_pos", "circulation_neg",
    "wake_length", "wake_thickness", "wake_enstrophy",
)


class MLPProbe(nn.Module):
    """3 hidden layers, width 256, ReLU. Maps z (64,) -> scalar target."""

    def __init__(self, in_dim: int = 64, hidden: int = 256, out_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def load_split(name: str) -> dict:
    d = np.load(TARGETS_DIR / f"{name}.npz", allow_pickle=True)
    return {
        "z_full": d["z_full"].astype(np.float32),
        **{k: d[k].astype(np.float32) for k in TARGET_NAMES},
    }


def evaluate_full(
    probe: MLPProbe,
    split_data: dict,
    target: str,
    mean: float,
    std: float,
    device: torch.device,
) -> dict:
    """Evaluate probe on every (encounter, frame) pair in split_data.

    Returns r2 and rmse in ORIGINAL target units (un-normalised).
    """
    z = split_data["z_full"]  # (n, T, d)
    y_true_raw = split_data[target]  # (n, T)
    n, T, d = z.shape
    z_flat = z.reshape(n * T, d)
    y_true_flat = y_true_raw.reshape(n * T)
    mask = np.isfinite(y_true_flat)
    if not mask.all():
        z_flat = z_flat[mask]
        y_true_flat = y_true_flat[mask]
    with torch.no_grad():
        z_t = torch.from_numpy(z_flat).to(device)
        y_pred_norm = probe(z_t).cpu().numpy()
    y_pred = y_pred_norm * std + mean
    r2 = float(r2_score(y_true_flat, y_pred))
    rmse = float(np.sqrt(np.mean((y_true_flat - y_pred) ** 2)))
    return {"r2": r2, "rmse": rmse, "n_samples": int(y_true_flat.size)}


def train_one_probe(
    train: dict,
    test_a: dict,
    test_b: dict,
    test_c: dict,
    target: str,
    n_iters: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    n_enc, T_max, d = train["z_full"].shape
    y_train_all = train[target]  # (n_enc, T_max)

    finite_mask = np.isfinite(y_train_all)
    y_train_finite = y_train_all[finite_mask]
    mean = float(y_train_finite.mean())
    std = float(y_train_finite.std())
    if std < 1e-12:
        std = 1.0
    y_norm = (y_train_all - mean) / std

    probe = MLPProbe(in_dim=d).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)

    losses = np.zeros(n_iters, dtype=np.float32)
    probe.train()

    has_finite_per_enc = finite_mask.any(axis=1)  # (n_enc,) True if any frame in encounter is usable
    candidate_encs = np.where(has_finite_per_enc)[0]

    for it in range(n_iters):
        # IID-1-frame-per-encounter sampling:
        if batch_size >= candidate_encs.size:
            chosen_encs = candidate_encs.copy()
            rng.shuffle(chosen_encs)
        else:
            chosen_encs = rng.choice(candidate_encs, size=batch_size, replace=False)
        z_batch = np.empty((chosen_encs.size, d), dtype=np.float32)
        y_batch = np.empty(chosen_encs.size, dtype=np.float32)
        for j, enc_idx in enumerate(chosen_encs):
            ok_frames = np.where(finite_mask[enc_idx])[0]
            t = int(rng.choice(ok_frames))
            z_batch[j] = train["z_full"][enc_idx, t]
            y_batch[j] = y_norm[enc_idx, t]
        z_t = torch.from_numpy(z_batch).to(device)
        y_t = torch.from_numpy(y_batch).to(device)
        opt.zero_grad()
        pred_norm = probe(z_t)
        loss = ((pred_norm - y_t) ** 2).mean()
        loss.backward()
        opt.step()
        losses[it] = float(loss.item())

    probe.eval()
    eval_results = {
        sp: evaluate_full(probe, data, target, mean, std, device)
        for sp, data in (("train", train), ("test_a", test_a), ("test_b", test_b), ("test_c", test_c))
    }
    p_preq = prequential_coding(losses)
    return {
        "target": target,
        "target_mean": mean,
        "target_std": std,
        "n_iters": n_iters,
        "batch_size": batch_size,
        "lr": lr,
        "final_train_loss_normed": float(np.median(losses[-int(0.1 * n_iters):])),
        "p_preq_normed": float(p_preq),
        "loss_curve_path": str((LOSS_DIR / f"{target}.npy").relative_to(REPO)),
        "splits": eval_results,
    }, losses


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[exp2] device={device}")

    train = load_split("train")
    test_a = load_split("test_a")
    test_b = load_split("test_b")
    test_c = load_split("test_c")
    print(
        f"[exp2] z_full shapes: train={train['z_full'].shape} "
        f"test_a={test_a['z_full'].shape} "
        f"test_b={test_b['z_full'].shape} test_c={test_c['z_full'].shape}"
    )

    summary: dict = {
        "encoder_ckpt": "outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt",
        "probe_arch": "MLPProbe: 3 hidden layers, width 256, ReLU",
        "n_iters": args.iters,
        "batch_size": args.batch,
        "lr": args.lr,
        "iid_sampling": "Per iteration: sample batch_size distinct encounters without replacement, then one random frame per encounter. No two samples in a batch share (case, encounter).",
        "evaluation": "R^2 computed on ALL frames of each held-out split (test_b, test_c) and on all train/test_a frames for reference.",
        "results_by_target": {},
    }

    print(
        f"\n[exp2] {'target':<22s} "
        f"{'train R^2':>10s} {'test_a R^2':>11s} {'test_b R^2':>11s} "
        f"{'test_c R^2':>11s} {'P_preq':>10s}"
    )
    print("-" * 90)
    t0 = time.time()
    for target in TARGET_NAMES:
        result, losses = train_one_probe(
            train, test_a, test_b, test_c,
            target=target,
            n_iters=args.iters,
            batch_size=args.batch,
            lr=args.lr,
            device=device,
        )
        np.save(LOSS_DIR / f"{target}.npy", losses)
        summary["results_by_target"][target] = result
        r2s = result["splits"]
        print(
            f"  {target:<22s} "
            f"{r2s['train']['r2']:>+10.3f} {r2s['test_a']['r2']:>+11.3f} "
            f"{r2s['test_b']['r2']:>+11.3f} {r2s['test_c']['r2']:>+11.3f} "
            f"{result['p_preq_normed']:>10.2f}"
        )

    print(f"\n[exp2] total wall: {time.time() - t0:.1f}s")
    save = OUT / "probe_sweep.json"
    save.write_text(json.dumps(summary, indent=2))
    print(f"[exp2] wrote {save.relative_to(REPO)}")


if __name__ == "__main__":
    main()

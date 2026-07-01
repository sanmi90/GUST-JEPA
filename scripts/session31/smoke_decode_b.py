"""Gate B smoke (RTX 6000): spatial encoder -> decoder probe -> field plumbing.

Builds a random-init spatial-latent encoder and a random-init decoder probe,
loads ONE real v2p2 encounter through :class:`EpisodeDataset`, encodes it, and
decodes the frozen spatial latent back to a vorticity field. The gate checks
PLUMBING only: the output field has the right shape and a non-degenerate spatial
variance (> eps). With random init the decode is random, which is fine here --
the real decode-floor verdict is post-Track-C (D-N3), not this smoke.

Hardware rule (CLAUDE.md): selects an RTX 6000 Blackwell via
``require_rtx6000(gpu_index=...)``; never the L40S, never a silent CPU fallback.
This smoke reads a single encounter directly (no DataLoader / worker pool), so
it has no CPU-fan-out concern of its own; still run it confined to a core subset
per the shared-box rule, e.g.::

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \\
        taskset -c 0-15 python scripts/session31/smoke_decode_b.py --gpu 0
"""

from __future__ import annotations

import argparse

import torch

from src.data.episode_dataset import EpisodeDataset
from src.models.encoder import HybridCNNViTEncoder
from src.probes import DecoderProbe
from src.utils.device import NoRTX6000Error, require_rtx6000


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate B decode plumbing smoke (RTX 6000).")
    parser.add_argument("--gpu", type=int, default=0, help="RTX 6000 subset index.")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--subtraj-len", type=int, default=8)
    args = parser.parse_args()

    try:
        device = require_rtx6000(gpu_index=args.gpu)
    except NoRTX6000Error as e:
        print(f"[gate-b] BLOCKED: no RTX 6000 available: {e}", flush=True)
        return 2

    print(f"[gate-b] device={device} ({torch.cuda.get_device_name(device.index)})", flush=True)

    dataset = EpisodeDataset(
        partition="v2p2",
        split="train",
        subtraj_len=args.subtraj_len,
        return_pressure=False,
        return_forces=False,
        split_manifest_path="configs/splits/split_v2p2.json",
        omega_pipeline_manifest="outputs/data_pipeline/v2p2/manifest.json",
        seed=0,
    )
    sample = dataset[0]
    omega = sample["omega_z"]  # (L, 192, 96) in 3-sigma normalized space
    case_id = sample["case_id"]
    x = omega.unsqueeze(0).unsqueeze(2).to(device)  # (1, L, 1, 192, 96)
    print(
        f"[gate-b] loaded case={case_id} enc={sample['encounter_index']} "
        f"omega={tuple(omega.shape)} x={tuple(x.shape)}",
        flush=True,
    )

    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder(latent_mode="spatial", latent_dim=args.latent_dim)
    probe = DecoderProbe(encoder, latent_dim=args.latent_dim).to(device).eval()

    with torch.no_grad():
        z = probe.encode(x)
        field = probe(x)

    print(f"[gate-b] spatial latent shape = {tuple(z.shape)}", flush=True)
    print(f"[gate-b] decoded field shape  = {tuple(field.shape)}", flush=True)

    assert field.shape == (1, args.subtraj_len, 1, 192, 96), f"bad field shape {tuple(field.shape)}"
    assert torch.isfinite(field).all(), "non-finite values in decoded field"

    per_frame_var = field.flatten(-2).var(dim=-1)  # (1, L, 1)
    mean_var = float(per_frame_var.mean())
    min_var = float(per_frame_var.min())
    print(
        f"[gate-b] field spatial variance: mean={mean_var:.6e} min={min_var:.6e}",
        flush=True,
    )

    eps = 1e-8
    assert min_var > eps, f"degenerate (constant) field: min spatial variance {min_var} <= {eps}"

    print(
        f"[gate-b] PASS: shape {tuple(field.shape)}, non-degenerate "
        f"(min spatial var {min_var:.6e} > {eps:g})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

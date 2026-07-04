"""Train + evaluate the SIGReg-JEPA-ROM no-lift arm on v2p2 (Session 34).

The "last no-lift attempt" after the Track C collapse finding (every kit cell
without the scalar lift head collapses to PR ~2): Carlos's uploaded
SIGReg-JEPA-ROM design anchors the latent with a POD-anchored RECONSTRUCTION
plus SIGReg instead of an observable head. This runner adapts it to the data
ONLY (fields, POD basis, splits, hardware rules); the method is exactly the
uploaded file (scripts/session34/sigreg_jepa_rom.py).

Stages run: 1 (encoder warm start on standardized POD coefficients; decoder
residual and predictor head are zero-initialized, so the model starts exactly
at POD + persistence) then 3 (L_rec + L_pred + L_sig, the JEPA core). Stage-1
learning rate is unspecified in the design; a plain AdamW 3e-4 on the encoder
alone is used and logged. Stage 3 uses the file's build_optimizer defaults
(enc 3e-5, dec 1e-4, pred 3e-4, wd 1e-4) and lam = {rec 1, pred 1, sig 0.1}.

Diagnostics every --diag-every steps on a fixed test_b subsample (the kit
diagnostics convention): participation ratio (floor 0.3 r), effective rank,
tau=1 contraction ratio. After training: encodes train + test_b, emits latent
caches in the Track C schema (so trackc_lift_eval can score this arm with the
IDENTICAL frozen probe protocol), fits the RidgeCV C_L probe inline, and
writes a summary JSON.

Run (RTX 6000):
    taskset -c 0-15 python -m scripts.session34.run_rom_nolift --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.session34.rom_pod_basis import H, W, load_split_fields  # noqa: E402
from scripts.session34.sigreg_jepa_rom import (  # noqa: E402
    Encoder,
    HorizonWeights,
    PODAnchoredDecoder,
    Predictor,
    build_optimizer,
    contraction_ratio,
    effective_rank,
    participation_ratio,
    stage3_step,
)

TAUS = (1, 2, 5, 10, 20)
CTX_L = 64


class WindowSampler:
    """Uniform (encounter, start) sampler over preloaded centered fields."""

    def __init__(self, data: dict, mean_field: np.ndarray, ctx_len: int,
                 max_tau: int, seed: int, return_cl: bool = False) -> None:
        self.fields = data["fields"]                    # (M, H, W) float32
        self.cl = data["cl"].astype(np.float32)         # (M,)
        self.mean = mean_field[None]                    # (1, H, W)
        self.ctx_len = ctx_len
        self.max_tau = max_tau
        self.return_cl = bool(return_cl)
        self.rng = np.random.default_rng(seed)
        self.enc = [
            (row0, T) for (row0, T) in data["enc_offsets"].values()
            if T >= ctx_len + max_tau
        ]

    def batch(self, B: int, device) -> dict:
        ctx = np.empty((B, self.ctx_len, 1, H, W), dtype=np.float32)
        tgt = {t: np.empty((B, 1, H, W), dtype=np.float32) for t in TAUS}
        cl_ctx = np.empty((B, self.ctx_len), dtype=np.float32)
        for b in range(B):
            row0, T = self.enc[self.rng.integers(len(self.enc))]
            s = int(self.rng.integers(0, T - self.ctx_len - self.max_tau + 1))
            sl = self.fields[row0 + s : row0 + s + self.ctx_len] - self.mean
            ctx[b, :, 0] = sl
            cl_ctx[b] = self.cl[row0 + s : row0 + s + self.ctx_len]
            last = row0 + s + self.ctx_len - 1
            for t in TAUS:
                tgt[t][b, 0] = self.fields[last + t] - self.mean
        out = {
            "context": torch.from_numpy(ctx).to(device),
            "targets": {t: torch.from_numpy(v).to(device) for t, v in tgt.items()},
        }
        if self.return_cl:
            out["cl_context"] = torch.from_numpy(cl_ctx).to(device)
        return out


@torch.no_grad()
def encode_all(enc: torch.nn.Module, fields: np.ndarray, mean_field: np.ndarray,
               device, batch: int = 256) -> np.ndarray:
    out = []
    enc.eval()
    for i in range(0, fields.shape[0], batch):
        x = torch.from_numpy(fields[i : i + batch] - mean_field[None]) \
            .unsqueeze(1).to(device)
        out.append(enc(x).cpu().numpy())
    enc.train()
    return np.concatenate(out, axis=0)


def save_latent_cache(path: Path, a: np.ndarray, data: dict, split: str,
                      name: str) -> None:
    np.savez_compressed(
        path,
        z_gap=a.astype(np.float32),
        z_spatial=a.astype(np.float32)[:, :, None, None],
        case_id=data["case_id"].astype(str),
        encounter_index=data["encounter_index"],
        frame=data["frame"],
        window_mask=np.zeros(a.shape[0], dtype=bool),
        latent_grid=np.array([1, 1], dtype=np.int64),
        split=np.str_(split),
        model_name=np.str_(name),
        target_C_L=data["cl"].astype(np.float64),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SIGReg-JEPA-ROM no-lift arm")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--warm-iters", type=int, default=2000)
    ap.add_argument("--stage3-iters", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warm-batch", type=int, default=64)
    ap.add_argument("--diag-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--pod", default="outputs/session34/rom_pod_basis.npz")
    ap.add_argument("--out", default="outputs/runs/session34/rom_nolift_s0")
    ap.add_argument("--cache-dir", default="outputs/session34/trackc_latents")
    args = ap.parse_args(argv)

    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    gpu_name = torch.cuda.get_device_name(device.index)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    name = out_dir.name

    pod = np.load(REPO_ROOT / args.pod)
    mean_field = pod["mean_field"].astype(np.float32)
    phi = torch.from_numpy(pod["phi"][:, : args.r]).float()
    lam = torch.from_numpy(pod["lam"][: args.r]).float()
    print(f"[rom] device={device} ({gpu_name}) r={args.r} "
          f"POD energy={float(pod['energy_fraction'][:args.r].sum()):.3f}", flush=True)

    print("[rom] loading train fields", flush=True)
    tr = load_split_fields("train")
    sampler = WindowSampler(tr, mean_field, CTX_L, max(TAUS), args.seed)

    models = {
        "enc": Encoder(c_in=1, r=args.r, width=32, n_down=4, hw=(H, W)).to(device),
        "dec": PODAnchoredDecoder(phi, lam, (1, H, W), r=args.r, width=32,
                                  n_up=4).to(device),
        "pred": Predictor(r=args.r, L=CTX_L, d=128, depth=4, heads=4,
                          taus=TAUS).to(device),
    }

    # Fixed diagnostic batch: test_b subsample (kit diagnostics convention).
    print("[rom] loading test_b fields", flush=True)
    tb = load_split_fields("test_b")
    rng = np.random.default_rng(0)
    diag_rows = rng.choice(tb["fields"].shape[0], size=1536, replace=False)
    diag_x = torch.from_numpy(tb["fields"][diag_rows] - mean_field[None]) \
        .unsqueeze(1)

    # ---- Stage 1: encoder warm start on standardized POD coefficients ------
    Xc = tr["fields"].reshape(tr["fields"].shape[0], -1) - mean_field.reshape(-1)[None]
    a_pod_std = (Xc @ pod["phi"][:, : args.r]) / np.sqrt(pod["lam"][: args.r])[None]
    a_pod_std = a_pod_std.astype(np.float32)
    warm_opt = torch.optim.AdamW(models["enc"].parameters(), lr=3e-4,
                                 weight_decay=1e-4)
    t0 = time.time()
    log_path = out_dir / "train.log"
    log = open(log_path, "a")

    def emit(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    emit(f"[rom] stage 1 warm start: {args.warm_iters} iters "
         f"(AdamW 3e-4, encoder only)")
    M = Xc.shape[0]
    for it in range(1, args.warm_iters + 1):
        idx = np.random.randint(0, M, size=args.warm_batch)
        x = torch.from_numpy(tr["fields"][idx] - mean_field[None]) \
            .unsqueeze(1).to(device)
        y = torch.from_numpy(a_pod_std[idx]).to(device)
        loss = torch.nn.functional.mse_loss(models["enc"](x), y)
        warm_opt.zero_grad(set_to_none=True)
        loss.backward()
        warm_opt.step()
        if it % 200 == 0:
            emit(f"[rom] warm {it}/{args.warm_iters} mse={loss.item():.4f}")
    del warm_opt

    # ---- Stage 3: L_rec + L_pred + L_sig ------------------------------------
    opt = build_optimizer(models)
    hw_ = HorizonWeights(TAUS)
    metrics_path = out_dir / "metrics.jsonl"
    emit(f"[rom] stage 3: {args.stage3_iters} iters "
         f"(B={args.batch}, L={CTX_L}, taus={TAUS})")
    for it in range(1, args.stage3_iters + 1):
        batch = sampler.batch(args.batch, device)
        logs = stage3_step(models, batch, hw_, opt)
        if it % args.log_every == 0:
            emit(f"[rom] iter {it}/{args.stage3_iters} "
                 f"rec={logs['rec']:.4f} pred={logs['pred']:.4f} "
                 f"sig={logs['sig']:.5f} ({time.time() - t0:.0f}s)")
        if it % args.diag_every == 0 or it == args.stage3_iters:
            with torch.no_grad():
                a_diag = []
                for i in range(0, diag_x.shape[0], 256):
                    a_diag.append(models["enc"](diag_x[i : i + 256].to(device)))
                a_diag = torch.cat(a_diag, 0)
                pr = participation_ratio(a_diag)
                er = effective_rank(a_diag)
                # tau=1 contraction on one sampled batch
                b2 = sampler.batch(args.batch, device)
                a_ctx = models["enc"](b2["context"].flatten(0, 1)) \
                    .view(args.batch, CTX_L, -1)
                a_hat = models["pred"](a_ctx, 1)
                a_ref = models["enc"](b2["targets"][1])
                cr = contraction_ratio(a_hat, a_ref)
            rec = {"step": it, "diag/pr": pr, "diag/effective_rank": er,
                   "diag/contraction_tau1": cr, **{f"loss/{k}": v
                                                   for k, v in logs.items()}}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            emit(f"[rom] DIAG iter {it}: PR={pr:.1f} (floor {0.3 * args.r:.1f}) "
                 f"effrank={er:.1f} contraction(tau1)={cr:.2f}")

    torch.save({k: m.state_dict() for k, m in models.items()},
               out_dir / f"checkpoint_iter{args.stage3_iters:06d}.pt")

    # ---- Eval: latent caches + probe + summary -------------------------------
    cache_dir = REPO_ROOT / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    a_tr = encode_all(models["enc"], tr["fields"], mean_field, device)
    a_tb = encode_all(models["enc"], tb["fields"], mean_field, device)
    save_latent_cache(cache_dir / f"latents_{name}_train.npz", a_tr, tr, "train", name)
    save_latent_cache(cache_dir / f"latents_{name}_test_b.npz", a_tb, tb, "test_b", name)

    from src.evaluation.represent import fit_linear_probe

    probe = fit_linear_probe(a_tr, tr["cl"])
    from sklearn.metrics import r2_score

    cl_r2 = float(r2_score(tb["cl"], probe.predict(a_tb)))
    summary = {
        "name": name,
        "gpu_name": gpu_name,
        "seed": args.seed,
        "r": args.r,
        "pod_energy_r": float(pod["energy_fraction"][: args.r].sum()),
        "warm_iters": args.warm_iters,
        "stage3_iters": args.stage3_iters,
        "final_pr_test_b": participation_ratio(torch.from_numpy(a_tb)),
        "final_effective_rank_test_b": effective_rank(torch.from_numpy(a_tb)),
        "cl_linear_probe_r2_test_b": cl_r2,
        "pr_floor": 0.3 * args.r,
        "wall_s": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    emit(f"[rom] SUMMARY: PR(test_b)={summary['final_pr_test_b']:.1f} "
         f"effrank={summary['final_effective_rank_test_b']:.1f} "
         f"C_L probe R2={cl_r2:+.3f}")
    emit(f"[rom] done in {time.time() - t0:.0f}s")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

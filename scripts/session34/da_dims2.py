import sys, json, subprocess
sys.path.insert(0, "/home/carlos/GUST-JEPA")
import numpy as np
import torch
from pathlib import Path

REPO = Path("/home/carlos/GUST-JEPA")
CACHE = REPO / "outputs/session34/trackc_latents"

# ---- 0a. POD latent caches at d in {4,8,16,32} ------------------------------
pod = np.load(REPO / "outputs/session34/rom_pod_basis.npz")
mean = pod["mean_field"].reshape(-1).astype(np.float32)
for split in ("train", "test_b"):
    f = np.load(CACHE / f"fields_{split}.npz", allow_pickle=True)
    X = f["omega_norm"].astype(np.float32).reshape(f["omega_norm"].shape[0], -1) - mean
    A32 = (X @ pod["phi"][:, :32]) / np.sqrt(pod["lam"][:32])[None]
    for d in (4, 8, 16, 32):
        out = CACHE / f"latents_pod_d{d}_{split}.npz"
        if out.exists():
            continue
        z = A32[:, :d].astype(np.float32)
        np.savez_compressed(out, z_gap=z, z_spatial=z[:, :, None, None],
                            case_id=f["case_id"].astype(str),
                            encounter_index=f["encounter_index"], frame=f["frame"],
                            window_mask=f["window_mask"],
                            latent_grid=np.array([1, 1]), split=np.str_(split),
                            model_name=np.str_(f"pod_d{d}"),
                            **{k: f[k] for k in f.files if k.startswith("target_")})
print("[dims2] pod caches done", flush=True)

# ---- 0b. fukami d32 encode ----------------------------------------------------
if not (CACHE / "latents_fukami_wake_train.npz").exists():
    subprocess.run([sys.executable, "-m", "scripts.session34.trackc_encode", "--gpu", "1",
                    "--models", "fukami_wake"], cwd=REPO, check=True, capture_output=True)
print("[dims2] fukami d32 encoded", flush=True)

MODELS = [f"pod_d{d}" for d in (4, 8, 16, 32)] + \
         ["fukami_wake_d4", "fukami_wake_d8", "fukami_wake_d16", "fukami_wake"]

# ---- 1. OSP staircases ---------------------------------------------------------
import scripts.session32.track_o1_recovery as o1
from scripts.session32.osp_select import build_osp_taps
from src.evaluation.rom_eval import load_windows
taps_path = REPO / "outputs/session34/osp_taps_dims2.json"
osp = json.loads((REPO / "outputs/session34/osp_taps_trackc.json").read_text())
osp.update(json.loads((REPO / "outputs/session33/osp_taps_vec.json").read_text()))
if taps_path.exists():
    osp.update(json.loads(taps_path.read_text()))
windows = load_windows(REPO / "outputs/session31/windows_v2p2.json")
qdeim = json.loads((REPO / "outputs/session32/qdeim_taps_v2p2.json").read_text())
p_train = o1.load_pressure(CACHE, "train")["p_wall"]
for m in MODELS:
    if m in osp:
        print(f"[dims2] taps exist: {m}", flush=True); continue
    print(f"[dims2] staircase: {m}", flush=True)
    caches = {m: o1.load_cache(CACHE, m, "train")}
    osp[m] = build_osp_taps(caches, windows, p_train, w=30, qdeim_taps=qdeim, seed=0)[m]
    taps_path.write_text(json.dumps(osp, indent=2))
taps_path.write_text(json.dumps(osp, indent=2))

# ---- 2. REX operators -----------------------------------------------------------
for m in MODELS:
    if (REPO / f"outputs/session34/latent_rex_model_{m}.pt").exists():
        continue
    print(f"[dims2] rex: {m}", flush=True)
    subprocess.run([sys.executable, "-m", "scripts.session34.latent_rex", "--gpu", "1",
                    "--run", m, "--out", f"outputs/session34/latent_rex_{m}.json"],
                   cwd=REPO, check=True, capture_output=True)

# ---- 3. decoders ------------------------------------------------------------------
from src.evaluation.represent import fit_decode_floor_decoder
from src.utils.device import require_rtx6000
device = require_rtx6000(gpu_index=1)
ftr = np.load(CACHE / "fields_train.npz")["omega_norm"].astype(np.float32)
tile = lambda z: np.repeat(np.repeat(z[:, :, None, None], 24, 2), 12, 3).astype(np.float32)
for m in MODELS:
    dpath = REPO / f"outputs/session34/trackc_decoders/decoder_{m}.pt"
    if dpath.exists():
        continue
    print(f"[dims2] decoder: {m}", flush=True)
    ztr = np.load(CACHE / f"latents_{m}_train.npz")["z_gap"]
    dec = fit_decode_floor_decoder(tile(ztr), ftr, (24, 12), device=device,
                                   steps=6000, verbose=False)
    torch.save(dec.state_dict(), dpath)
    del dec; torch.cuda.empty_cache()

# ---- 4. DA phase eval ---------------------------------------------------------------
for m in MODELS:
    out = REPO / f"outputs/session34/da_phase_dim_{m}.json"
    if out.exists():
        continue
    print(f"[dims2] da eval: {m}", flush=True)
    subprocess.run([sys.executable, "-m", "scripts.session34.da_phase_eval", "--gpu", "1",
                    "--model", m, "--cache-dir", "outputs/session34/trackc_latents",
                    "--pressure-dir", "outputs/session34/trackc_latents",
                    "--taps", "outputs/session34/osp_taps_dims2.json",
                    "--rex-ckpt", f"outputs/session34/latent_rex_model_{m}.pt",
                    "--out", str(out)], cwd=REPO, check=True, capture_output=True)
print("DA-DIMS2 COMPLETE", flush=True)

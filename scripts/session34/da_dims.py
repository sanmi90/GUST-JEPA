import sys, json, time
sys.path.insert(0, "/home/carlos/GUST-JEPA")
import numpy as np
import torch
from pathlib import Path

REPO = Path("/home/carlos/GUST-JEPA")
CACHE = REPO / "outputs/session34/trackc_latents"
MODELS = ["jepa_pool_vec_d4", "jepa_pool_vec_d8", "jepa_pool_vec_d16",
          "cln_rexpred_d4_s0", "jepa_pool_ln_rexpred_s0"]

# ---- 1. OSP staircases (own taps per model) --------------------------------
import scripts.session32.track_o1_recovery as o1
from scripts.session32.osp_select import build_osp_taps
from src.evaluation.rom_eval import load_windows

taps_path = REPO / "outputs/session34/osp_taps_dims.json"
osp = json.loads((REPO / "outputs/session34/osp_taps_trackc.json").read_text())
osp.update(json.loads((REPO / "outputs/session33/osp_taps_vec.json").read_text()))
windows = load_windows(REPO / "outputs/session31/windows_v2p2.json")
qdeim = json.loads((REPO / "outputs/session32/qdeim_taps_v2p2.json").read_text())
p_train = o1.load_pressure(CACHE, "train")["p_wall"]
for m in MODELS:
    if m in osp:
        print(f"[dims] taps exist: {m}", flush=True); continue
    print(f"[dims] building OSP staircase: {m}", flush=True)
    caches = {m: o1.load_cache(CACHE, m, "train")}
    osp[m] = build_osp_taps(caches, windows, p_train, w=30, qdeim_taps=qdeim, seed=0)[m]
taps_path.write_text(json.dumps(osp, indent=2))
print("[dims] taps done", flush=True)

# ---- 2. REX operators per model ---------------------------------------------
import subprocess
for m in MODELS:
    ck = REPO / f"outputs/session34/latent_rex_model_{m}.pt"
    if ck.exists():
        print(f"[dims] rex exists: {m}", flush=True); continue
    print(f"[dims] training rex: {m}", flush=True)
    subprocess.run([sys.executable, "-m", "scripts.session34.latent_rex", "--gpu", "0",
                    "--run", m, "--out", f"outputs/session34/latent_rex_{m}.json"],
                   cwd=REPO, check=True, capture_output=True)

# ---- 3. decode-floor decoders per model --------------------------------------
from src.evaluation.represent import fit_decode_floor_decoder
from src.utils.device import require_rtx6000
device = require_rtx6000(gpu_index=0)
ftr = np.load(CACHE / "fields_train.npz")["omega_norm"].astype(np.float32)
tile = lambda z: np.repeat(np.repeat(z[:, :, None, None], 24, 2), 12, 3).astype(np.float32)
for m in MODELS:
    dpath = REPO / f"outputs/session34/trackc_decoders/decoder_{m}.pt"
    if dpath.exists():
        print(f"[dims] decoder exists: {m}", flush=True); continue
    print(f"[dims] fitting decoder: {m}", flush=True)
    ztr = np.load(CACHE / f"latents_{m}_train.npz")["z_gap"]
    dec = fit_decode_floor_decoder(tile(ztr), ftr, (24, 12), device=device,
                                   steps=6000, verbose=False)
    torch.save(dec.state_dict(), dpath)
    del dec; torch.cuda.empty_cache()

# ---- 4. phase-resolved DA per model ------------------------------------------
for m in MODELS:
    out = REPO / f"outputs/session34/da_phase_dim_{m}.json"
    if out.exists():
        print(f"[dims] da exists: {m}", flush=True); continue
    print(f"[dims] da eval: {m}", flush=True)
    subprocess.run([sys.executable, "-m", "scripts.session34.da_phase_eval", "--gpu", "0",
                    "--model", m, "--cache-dir", "outputs/session34/trackc_latents",
                    "--pressure-dir", "outputs/session34/trackc_latents",
                    "--taps", "outputs/session34/osp_taps_dims.json",
                    "--rex-ckpt", f"outputs/session34/latent_rex_model_{m}.pt",
                    "--out", str(out)], cwd=REPO, check=True, capture_output=True)
print("DA-DIMS COMPLETE", flush=True)

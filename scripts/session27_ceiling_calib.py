"""Rollout-free DNS ceiling: probe the TRUE encoder latent at impact+16 (z_dns)
for each family with identical probe machinery (cr.fit_probes). Calibrates whether
the PredAE ceiling sits with the reconstructive/linear families or is anomalous.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session20"))
import exp_closure_r2 as cr  # noqa: E402

H = 16
dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)


def r2(yp, yt):
    yp, yt = np.asarray(yp), np.asarray(yt)
    return 1.0 - float(((yp - yt) ** 2).sum()) / max(float(((yt - yt.mean()) ** 2).sum()), 1e-12)


def ceiling(latdir):
    probe = cr.fit_probes(Path(latdir), dns)["wake_enstrophy"]
    out = {}
    for split in ("test_b", "test_c"):
        b = np.load(Path(latdir) / f"{split}.npz", allow_pickle=True)
        zf = b["z_full"].astype(np.float32)
        imp = cr._get(b, "impact_frame").astype(int)
        cid = cr._get(b, "case_id", "case_ids"); ei = cr._get(b, "encounter_index", "encounter_indices")
        midx = cr.match_index(cid, ei, dns[f"{split}_case_id"], dns[f"{split}_encounter_index"])
        yp, yt = [], []
        for j in range(len(imp)):
            if midx[j] < 0:
                continue
            te = int(imp[j]) + H
            yp.append(float(cr.apply_probe(zf[j, te][None], probe)[0]))
            yt.append(float(dns[f"{split}_wake_enstrophy"][midx[j], te]))
        out[split] = r2(yp, yt)
    return out


fams = [("JEPA (pred-latent)", "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"),
        ("Fukami AE (static recon)", "outputs/session18/exp_b1/latents_fukami_d64_noBN"),
        ("POD (linear)", "outputs/session18/exp_b1/latents_pod_d64_noBN")]
print("=== rollout-free DNS ceiling: wake-enstrophy R2 @ impact+16 (true latent) ===")
for fam, latdir in fams:
    c = ceiling(REPO / latdir)
    print(f"  {fam:26s} test_b {c['test_b']:+.3f}   test_c {c['test_c']:+.3f}")
print("  PredAE (pred-pixel)        test_b -0.210   test_c -1.431   (from session27_predae_eval)")

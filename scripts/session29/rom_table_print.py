"""Print the fair ROM field-forecast table from the JSONs in
outputs/session29/rom_field_forecast/. SSIM vs forecast horizon, per split.
Rows are models on the SHARED matched-predictor recipe (only latent + decoder
vary); reference rows (jepa-own co-trained predictor, vjepa_dense) shown where
available. Missing entries print as '-' so the table renders before every model
has landed."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROM = REPO / "outputs" / "session29" / "rom_field_forecast"
HORIZONS = ["0", "1", "2", "4", "8", "12", "16", "24", "32"]

# (label, rollout-tag). The matched-predictor fair set first, then references.
FAIR = [
    ("jepa  (per-frame, paper)", "jepa_matched_d64_s0"),
    ("regae (reconstructive AE)", "regae_d64"),
    ("bvae_match (beta-VAE+heads)", "bvae_match_matched_s0"),
    ("vjepa_best (FULL 2.1+heads)", "vjepa_best_matched_s0"),
]
REF = [
    ("  [ref] jepa-own predictor", "jepa_own_d64_s0"),
    ("  [ref] vjepa_dense matched", "vjepa_dense_matched_s0"),
]


def load(tag: str, split: str) -> dict | None:
    fn = ROM / (f"{tag}.json" if split == "test_b" else f"{tag}.{split}.json")
    if not fn.exists():
        return None
    return json.load(open(fn)).get("per_horizon", {})


def row(label: str, ph: dict | None) -> str:
    cells = []
    for h in HORIZONS:
        if ph and h in ph:
            cells.append(f"{ph[h]['ssim']:6.3f}")
        else:
            cells.append(f"{'-':>6}")
    return f"{label:<30}" + " ".join(cells)


def main() -> None:
    for split in ("test_b", "test_c"):
        print(f"\n=== ROM field-forecast SSIM | {split} "
              f"({'in-envelope' if split == 'test_b' else 'OOD G=+4'}) ===")
        print(f"{'model':<30}" + " ".join(f"{('h='+h):>6}" for h in HORIZONS))
        for label, tag in FAIR:
            print(row(label, load(tag, split)))
        for label, tag in REF:
            ph = load(tag, split)
            if ph:
                print(row(label, ph))
    print("\n(higher SSIM = better field reconstruction at that horizon; "
          "h=0 is the reconstruction anchor.)")


if __name__ == "__main__":
    main()

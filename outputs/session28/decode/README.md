# Session 28 / T9 decoded-field artifacts (Phase C/D)

Decoded omega_z fields for the four T9 visualization decoders, produced by
`scripts/session28/decode_operating_points.py` on the RTX 6000 Blackwell at
`require_rtx6000(gpu_index=1)` (torch `cuda:3` / `nvidia-smi` index 1; the L40S
cards were never used). All fields are float32 raw-scale omega
(pipeline-unnormalized via `outputs/data_pipeline/v2p1/manifest.json`).

## Operating points (frozen rule)

Per `PROTOCOL.md` Section 6 / `eval_protocol_v2p1.yaml` `decoders.operating_point`:
for each family, the saved checkpoint (every 2000 iterations) with maximum
validation (test_a) SSIM_mean; the SAME rule for every family, no reuse of
another family's iteration. SSIM is the trainer's own `evaluate_split` (Fukami
SSIM definition, raw-scale through the v2p1 omega pipeline) on the FULL test_a
(87 encounters), `ssim_data_range` registry value `split_v2p1 = 8.45`. The
per-checkpoint sweep and the SAME-rule statement are in `operating_points.json`.

| family | operating iter | val (test_a) SSIM_mean | SSIM at 30k | still rising at 30k |
|---|---|---|---|---|
| dec_jepa_tf_noc_d64_s42 (production transformer) | 20000 | 0.606 | 0.603 | no |
| dec_jepa_lstm_noc_d64_s42 (production LSTM) | 22000 | 0.591 | 0.591 | no |
| dec_posthoc_fukami_d64 (post-hoc Fukami AE) | 20000 | 0.499 | 0.495 | no |
| dec_posthoc_pod_d64 (post-hoc POD) | 8000 | 0.676 | 0.668 | no |

No family is still improving at 30k, so none is undertrained. POD peaks earliest
(iter 8000): its latents are a fixed linear projection, so the matched decoder
converges fast and then mildly overfits the train set; this is a fast-converging
case under the frozen rule, not an anomaly.

## What was decoded

Two endpoints, on test_b and test_c, at each family's operating point:

- `recon_{test_b,test_c}.npz`: decode of the encoder/latents at the TRUE encoded
  latents (`z_dns`), i.e. the representational decode-ceiling reference.
- `forecast_{test_b,test_c}.npz`: decode of the FULL-CONTEXT autoregressive
  rollout (`z_full`). The Markov single-frame seeding (`z_markov`) is RETIRED and
  was never decoded.

Latent sources: `z_dns` / `z_full` were read from
`outputs/session28/rollouts/{jepa_d64,fukami_d64,pod_d64}/{split}.npz` for the
transformer, Fukami, and POD families. The LSTM family has no pre-built rollout
NPZ, so its full-context rollout was generated on the fly by rolling its own
cond_dim=0 LSTM predictor (`jepa_lstm_noc_d64_s42` checkpoint_iter020000) from the
pre-impact window, matching `scripts/session27_gen_rollouts_uncond.py`. Because
`z_full[:impact+1] == z_dns`, the recon and forecast fields are bit-identical up
to the impact frame and diverge only afterward (verified per encounter).

## NPZ layout

Each file holds (n_enc = 42 for test_b, 24 for test_c):

- `omega_decoded_window` (n_enc, 31, 192, 96): decoded fields over the impact
  window, absolute frames [25, 55] inclusive (`window_frames`).
- `omega_decoded_horizon` (n_enc, 4, 192, 96): decoded fields at impact + H for
  H in `horizon_offsets = {0, 8, 16, 24}`; `horizon_abs_frames` gives the
  absolute frame index per encounter (clamped to frame 119 if impact + H exceeds
  the cache).
- `impact_frame`, `case_ids`, `encounter_indices`: per-encounter metadata.
- `decoder_iter`, `family`, `endpoint`, `split`, `rollout_zkey`, `latent_source`.
- `dns_truth_ref`: the DNS truth reference path template
  (`${VORTEX_JEPA_CACHE}/v2p1/<case_id>/encounter_<k:02d>.h5 :: omega_z`,
  pipeline-preprocessed with the same mask + clip before any comparison).

We save the impact window plus the four horizons rather than all 120 frames to
bound size (~300 MB/family); this covers every frame the forecast figures and the
labelled decode-ceiling comparison need.

## Phase-blindness caveat (carry into the manuscript)

The decoder's spectral-amplitude loss term matches amplitude spectra only and is
blind to phase: decoded fields can carry plausible wake texture displaced from its
true location. These decoded fields therefore support visualization and the
labelled decode-ceiling comparison ONLY. No quantitative localization claim may
rest on them; such claims probe the latents directly.

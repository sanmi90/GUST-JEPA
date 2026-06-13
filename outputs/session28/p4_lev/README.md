# Physics Track P4: LEV circulation budget and gust-sign asymmetry (v2.1)

Read-only CPU analysis. Extends D146 (Session 23 LEV tracking) onto the v2.1 unconditioned rebuild with MATCHED decoders. Master plan Phase C, Physics Track P4.

## LEV definition (REUSED from scripts/session23/exp_lev_tracking.py)

The dominant suction-side leading-edge vortex is the strongest connected NEGATIVE-vorticity component (omega_z < 0, clockwise; convention du/dy - dv/dx) in the leading-edge / wing region x/c < 1 (streamwise pixel index < 80), with the frozen 140-cell airfoil-adjacent mask removed, on the LARGE-SCALE band (Gaussian filter sigma/c = 0.05 = 1.6 px, mode='nearest', matching exp_scale_decomposition). Identification uses an ABSOLUTE iso-level |omega_L| > 0.5 in 3-sigma normalised units (fair across families). Gamma_LEV = sum_lobe omega_z * dx * dy (signed, normalised); the centroid is the |omega|-weighted center_of_mass. The detachment clock is a circulation half-life proxy: the first frame strictly after the peak where the lobe is lost OR |Gamma_LEV| falls below 50% of its peak (the LE-region search confines the centroid to x/c < 1 by construction, so the shedding signal must be read from the circulation budget collapsing, not from a centroid crossing). The exp_lev_tracking lev_from_field and large_scale functions are imported verbatim; only the per-frame time series, the detachment clock, and the DNS-vs-decode budget around them are new.

Scale: the v2.1 DNS cache holds raw (mask + clip) omega_z and the matched decoded fields are raw (pipeline-unnormalised). Both are divided by 3 * train_std = 10.9010 (split_v2p1, outputs/data_pipeline/v2p1/manifest.json) before filtering and thresholding, so the LEV iso-level matches exp_lev_tracking exactly.

## NO impulse claim (D167 stands)

We report ONLY the DIRECT CORRELATION between peak |Delta C_L| and peak |Gamma_LEV|. There is NO impulse-theorem argument anywhere; we do NOT assert that the LEV circulation explains lift.

## (i) GATE GP4 (held-out trend)

Pearson r = 0.530 (95% case-clustered CI [0.268, 0.882], case-permutation two-sided p = 0.0445); Spearman rho = 0.751 (95% CI [0.412, 0.887], perm p = 0.0007999); n = 66 encounters across 14 cases (test_b + test_c). |corr| headline = 0.530 vs GP4 threshold 0.6 -> **APPENDIX**. The case-permutation p is VALID here: this is a correlation / trend test, not a paired location test, so the B6 paired-location degeneracy does not apply.

## (ii) G-sign asymmetry (positive vs negative gusts, matched |G|)

Matched |G| buckets: [0.5, 1.0, 1.5, 2.0, 3.0] (n_pos = 22, n_neg = 20). Peak |Gamma_LEV|: G>0 mean 0.155 vs G<0 mean 0.395 (pos - neg = -0.240, Mann-Whitney two-sided p = 0.06062); matched-bucket paired difference -0.233 (95% CI [-0.513, 0.010], 5 buckets). Detachment time t/c: G>0 mean 0.115 vs G<0 mean 0.350 (MW p = 0.02403). The PRF text describes the split-and-merge of the gust with the pre-existing suction-side vorticity qualitatively; this quantifies the budget asymmetry.

## (iii) Y modulation

By |Y|: |Y|=0.00: mean peak |Gamma_LEV|=0.036, corr(dCL,Gamma)=0.99 (n=4); |Y|=0.10: mean peak |Gamma_LEV|=0.491, corr(dCL,Gamma)=0.28 (n=42); |Y|=0.20: mean peak |Gamma_LEV|=0.154, corr(dCL,Gamma)=-0.32 (n=8); |Y|=0.40: mean peak |Gamma_LEV|=0.191, corr(dCL,Gamma)=0.95 (n=12).

By Y sign: Y_pos: mean peak |Gamma_LEV|=0.313, corr(dCL,Gamma)=0.67 (n=30); Y_neg: mean peak |Gamma_LEV|=0.460, corr(dCL,Gamma)=0.43 (n=32); Y_zero: mean peak |Gamma_LEV|=0.036, corr(dCL,Gamma)=0.99 (n=4).

## PART B: matched-decoder LEV tracking (D146 regenerated)

Fraction of encounters whose decoded Gamma_LEV(t) tracks the DNS Gamma_LEV(t) over [25, 55] (Pearson corr > 0.5). RECON decode is the headline (decode of the true encoded latent: does the family latent CARRY the LEV?); forecast is noted separately.

- tf_noc: recon 19/66 (mean corr 0.30); forecast 25/66 (mean corr 0.33).
- fukami: recon 18/66 (mean corr 0.25); forecast 13/66 (mean corr 0.12).
- pod: recon 32/66 (mean corr 0.41); forecast 21/66 (mean corr 0.30).

Paired recon tf_noc vs fukami (sign test, NOT case-permutation): tf_noc higher tracking-corr in 37/66 effective encounters, one-sided sign p = 0.1945; mean delta corr 0.050 (case-clustered 95% CI [-0.05614021719874016, 0.1610303208461143]).


D146 reported predictive 42/42 vs reconstructive 36/42 on the OLD (confounded) decoder. The numbers above are the HONEST regeneration on the matched decoders; they may differ and are reported as measured, weak or strong.

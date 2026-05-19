# Section 4 — Failure modes of pure self-supervised JEPA on low-intrinsic-dim physics

LaTeX-friendly markdown. Approximate target length: 3 pages. Figure 1
of this section is the 2x2 outcome table from Session 6. The text walks
through the four failure regions found across Sessions 5, 5.PLDM, and 6
before introducing Session 7's full-scale evaluation.

## 4.1 The diagnostic axes

A two-term SIGReg JEPA failing on physics data does not look like one
thing. We characterise the failure space along two axes derived from
the held-out Test A audit (Section 3.5).

Axis 1: latent rank. The participation ratio PR(z) collapses the
singular-value spectrum of the latent batch to a scalar. PR ranges
from 1 (the entire batch lives on a single axis in R^d) to d (the
batch is isotropic). For our d=32 latent, PR < 2 indicates
near-complete rank collapse; PR in [2, d/2] indicates partial rank;
PR > d/2 indicates a healthy spread.

Axis 2: case-vs-dynamics encoding. Decompose `z = z_mean + z_dyn`
where `z_mean` is the per-case mean over time and encounters, and
`z_dyn` is the within-case residual. A linear probe r^2(z_dyn -> c)
near zero means the dynamic part of the latent does NOT encode the
case descriptor c=(G, D, Y) — the c information lives in the case
mean. Conversely, r^2(z_dyn -> c) materially above zero indicates
the within-case latent still leaks case identity — what we call the
SPREAD_TRIVIAL signature.

The 2x2 of these two axes generates four named failure regions plus
the healthy quadrant.

## 4.2 The 2x2 outcome table (Figure 1)

|                           | r^2(z_dyn -> c) ~ 0 (case-mean encodes c) | r^2(z_dyn -> c) > 0 (case leaks into dyn) |
|---------------------------|-------------------------------------------|-------------------------------------------|
| PR(z) ~ 1 (rank-1)        | TRIVIAL                                   | (does not occur; trivial latent cannot leak) |
| PR(z) > d/2 (healthy rank)| HEALTHY                                   | SPREAD_TRIVIAL                            |

Session 5's pure SIGReg JEPA (Run A: SIGReg + BN) landed in the
TRIVIAL quadrant (D27): PR=1.02, r^2(z -> c)=0.78, all latent
magnitude in the case mean (norm ratio ||z_dyn||/||z||=0.10). The
encoder reduced its 32-d latent to a single axis along which c is
linearly decodable; the within-encounter dynamics were not encoded
in any meaningful sense.

Session 5.PLDM (D31) landed in the SPREAD_TRIVIAL quadrant: PR=5.97,
r^2(z -> c)=0.97, but r^2(z_dyn -> c)=-0.09 with r^2(z_dyn -> phase)
=0.58 and r^2(z -> CL_future) on Test A = 0.96 — the dynamic part
carries meaningful phase information and CL prediction works, but
the case-mean dominates the latent norm (D39 self-correction of the
D31 reading; the "DATA_SCALE_BOUND" label was too pessimistic for what
the audit metrics actually show).

Session 6 (D38, D39) ran the five factorial single-axis variants
F-L (L=64), F-CD (c-dropout=0.5), F-NC (cond_dim=0), F-S (24 cases),
and F-OBS (observable head, eta=0.01) at the 5-case smoke scale. Four
of the five (F-L, F-CD, F-NC, F-S) remained in the TRIVIAL quadrant.
F-OBS escaped partially: PR climbed from 1.02 to 3.11 at 5k iters and
to 3.83 at 10k iters; r^2(z -> CL_future) on Test A jumped from -0.02
to 0.95, beating the (c, t) baseline of 0.90. The static-vs-dynamic
audit confirmed the climb was real (r^2(z_dyn -> phase) = 0.47, no
c-leak into z_dyn).

The PLDM + observable head extension (Session 6, mid-session) landed
in the HEALTHY quadrant by the smoke metric: PR=6.09, r^2(z_dyn -> c)
=-0.13, r^2(z_dyn -> phase)=0.54, r^2(z -> CL_future)=0.96. But this
healthy reading is on the same 5 cases the smoke F-OBS run trained on,
and the (c, t) baseline at 5 cases is also 0.90 — high enough that
the latent only beats it by 0.06. Whether the same configuration
"holds up at scale" — meaning Test B at 41 cases, where the (c, t)
baseline drops because there are more cases to interpolate between —
is the central question Section 5 (full-scale results) answers.

## 4.3 The observable-augmentation lineage

A small auxiliary supervised head that predicts a future scalar from
the latent is not novel. Fukami and Taira's lift-augmented autoencoder
(\cite{fukami2023}; J. Fluid Mech. 2023) trained an encoder + decoder
+ CL prediction head jointly on transonic airfoil flow and showed that
the auxiliary CL loss reduced the intrinsic dimension of the discovered
manifold from ~10 to 3. Solera-Rico et al. (\cite{solera2024};
Nat. Commun. 2024) used the same gust-airfoil dataset family as this
paper and showed a beta-VAE + transformer ROM that conditioned on
aerodynamic observables produces predictive latent dynamics. Fukami
et al.'s 2025 transonic-buffet extension (\cite{fukami2025};
J. Fluid Mech. 2025) generalised the observable-augmentation idea
across multi-source turbulent flow data.

The contribution this paper makes on top of that lineage is the
regulariser-asymmetry observation. At the 5-case smoke scale the
observable CL head **rescues** a two-term SIGReg JEPA from TRIVIAL
collapse (Section 4.2, F-OBS) but only **marginally improves** a
5-term PLDM JEPA that is already in the HEALTHY quadrant (Section 4.2,
PLDM + OBS) -- the asymmetry that D39 read as "PLDM is the recommended
base." Section 5 (Sessions 7 and 8) **inverts this reading at scale.**
On the full 41-case partition, the simpler SIGReg + OBS generalises to
Test B with delta = +0.14 over the parametric (c, t) baseline, while
the 5-term PLDM + OBS is no better than that baseline on Test B
(delta = -0.01). The smoke-scale PR diagnostic alone is therefore not
a reliable proxy for generalisation; PR = 10+ on the PLDM trajectory
turns out to encode case-specific memorisation when 41 cases are
available to memorise. The Test B delta over the (c, t) baseline is
the diagnostic that survives the smoke-to-full-scale inversion. The
observable head still interacts asymmetrically with the regulariser
choice, but the direction of the asymmetry at scale is opposite the
smoke-scale reading: at scale, SIGReg + OBS wins.

## 4.4 The diagnostic-suite contribution

Independent of the model design, the static-vs-dynamic decomposition
and the (c, t) baseline are useful diagnostic tools for any latent-
dynamics model on low-intrinsic-dim physics data. The decomposition
distinguishes "encoder memorised the 41 case labels" (high
r^2(z -> c), low r^2(z_dyn -> c), low r^2(z -> CL_future)) from
"encoder learned dynamics" (high r^2(z_dyn -> phase), high r^2(z_dyn
-> CL_future), positive delta vs (c, t) baseline). The (c, t)
baseline is the minimum bar any genuine encoder must beat: if a tiny
MLP from (case_descriptor, time_index) predicts CL as well as the
trained encoder, the encoder is at best a glorified lookup.

The Session 7 evaluation suite (notebooks/05_session7_full_evaluation.ipynb)
applies these diagnostics to Test A, Test B, and Test C for each of the
three full-scale runs. Section 5 reports the complete metric table.

## Open writing TODO

- Figure 1 SVG: render the 2x2 table as a labeled axis diagram with the
  three named regions and the four Session 6 (R)un placements as dots.
  Bottom-right "HEALTHY" gets two dots: PLDM-A from Session 5.PLDM
  audit and PLDM + OBS from Session 6 mid-session extension.
- Cite Brain-JEPA / Echo-JEPA where the encoder is fully responsible
  for encoding subject-level information (analogous to F-NC).
- Cite Ho & Salimans \cite{cfg2022} for the F-CD inspiration.
- Decide if D39's "regulariser asymmetry" finding deserves its own
  named contribution (Contribution 3) or is folded into Contribution 2
  ("an observable-augmentation rescue regime"). Likely the former.

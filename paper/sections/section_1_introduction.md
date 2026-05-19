# Section 1: Introduction

LaTeX-friendly markdown. Approximate target length: 2 to 3 pages.
Citations are placeholder `\cite{key}` tokens whose `key` matches
HANDOFF.md "Key references"; the final BibTeX will resolve them.

## 1.1 Unsteady gust-airfoil interactions and the case for reduced-order models

Sharp transverse gusts interacting with lifting surfaces at low chord
Reynolds numbers are a routine operating condition for fixed-wing
micro air vehicles, urban-air-mobility rotorcraft, and small uncrewed
aircraft flying in the atmospheric boundary layer or in the wake of
buildings and vehicles. The unsteady aerodynamic response is dominated
by the timing of the vortex impact relative to the airfoil's own
shedding cycle, by the wall-normal offset of the vortex core relative
to the leading edge, and by the gust strength and core diameter. Even
at the moderate Reynolds number of Re=5000 the post-stall flow at
angle of attack alpha=14 degrees is fully separated, and the gust-
induced disturbance interacts non-linearly with the natural shedding
\cite{fukami2025,solera2024}. Direct numerical simulation (DNS) of the
full parametric envelope is expensive enough that reduced-order
modelling (ROM) of the encounter-to-encounter and case-to-case
variability is an active area of work.

Recent ROM efforts on this regime have converged on observable-
augmented autoencoder architectures. Fukami and Taira's
lift-augmented autoencoder \cite{fukami2023} trained an encoder, a
decoder, and a CL prediction head jointly on transonic airfoil flow
and reported that the auxiliary lift-prediction loss reduced the
intrinsic dimension of the discovered manifold from approximately ten
to three. Solera-Rico et al. \cite{solera2024} (Nat. Commun. 2024)
used a beta-VAE plus transformer ROM conditioned on aerodynamic
observables to forecast latent dynamics on a vortex-gust airfoil
dataset of the same family as the data in this paper. Fukami et al.'s
2025 transonic-buffet extension \cite{fukami2025} (J. Fluid Mech.
2025) generalised the observable-augmentation idea across
multi-source turbulent flow data. The common ingredient is a
reconstruction-trained encoder that also predicts an aerodynamic
scalar (CL, drag, surface pressure) from the bottleneck. The
reconstruction term anchors the latent geometry; the observable
prediction term shapes the manifold along the directions that matter
for downstream forecasting.

What has not been tried on this regime is a Joint-Embedding
Predictive Architecture (JEPA): a class of self-supervised latent
dynamics models that drops the reconstruction term entirely and trains
the encoder and predictor end-to-end only on latent-space forecasting
plus an anti-collapse regulariser. Removing the reconstruction term
frees the encoder from spending capacity on photorealistic detail and
lets it focus on the dynamically relevant degrees of freedom. The
JEPA lineage runs from the I-JEPA images formulation through V-JEPA
on video \cite{vjepa2}, with two recent end-to-end-from-pixels variants
that close the loop on a moving target encoder: LeWM \cite{lewm}
(Maes et al., arXiv:2603.19312, 2026), which trains a single online
encoder against the characteristic-function SIGReg regulariser of
LeJEPA \cite{lejepa} (Balestriero and LeCun, arXiv:2511.08544, 2025),
and PLDM \cite{pldm} (Sobal et al., arXiv:2502.14819, 2025), which
uses a five-term VICReg-derived objective \cite{vicreg}. Both
architectures have so far been benchmarked on gridworld and toy
visual domains; their behaviour on a low-intrinsic-dim physics
dataset, where the data itself sits on a roughly five- to ten-
dimensional manifold, is not characterised.

## 1.2 What this paper contributes

We train an end-to-end JEPA on DNS of a NACA 0012 airfoil at Re=5000,
alpha=14 degrees, perturbed by a parametric Taylor vortex characterised
by gust strength G, vortex core diameter D, and wall-normal offset
Y/c. The data partition v1.2 holds 41 train cases (138 encounters),
6 Test B cases at unseen interior (G, D, Y) values (28 encounters),
and 4 Test C cases at the |G|=4 extrapolation boundary (24
encounters); split sha256
`a721dc92f6e278ee054bb952933c14ba20a58137f79f3a19fc6ad71b70a007dd`,
locked at `configs/splits/split_v1.json`. The encoder is a hybrid
CNN+ViT mapping mid-plane spanwise vorticity (192 x 96) to a
d=32 latent through a one-layer MLP projection with BatchNorm
\cite{lewm}; the predictor is a six-layer autoregressive transformer
with rotary positional embeddings \cite{rope} and AdaLN-Zero
conditioning on c=(G, D, Y). Section 3 details the architecture, the
two-term SIGReg \cite{lejepa} and five-term PLDM \cite{pldm} loss
compositions, and the optional auxiliary CL-prediction head we adopt
from the Fukami / Solera-Rico observable-augmented lineage. The
headline figure of merit on each held-out split is
`delta = r2(z -> CL_future) - r2((c, t) -> CL_future)`, the gain over
a tiny parametric MLP baseline that maps (case_descriptor,
time_index) to CL.

The paper's three contribution claims, with their headline numbers
from Sessions 7 and 8 (see Section 5):

Contribution 1 (controlled-collapse SIGReg + OBS at scale). The
production cell at the (eta x lambda) grid optimum
(eta*=0.01, lambda*=0.01, d=32) reaches Test B delta = +0.159 over
the parametric baseline at 20k training iterations (D53 E4). This is
the simpler of the two anti-collapse bases in the JEPA literature
(two-term SIGReg, not five-term PLDM) augmented by a small
CL-prediction head. The encoder converges into a controlled-collapse
regime: PR(z) stays at approximately 3 (PR_all=2.61 on Test B at E4,
versus d=32) while the latent retains Test A to Test B alignment
strong enough to drive a CL prediction r2 of 0.88 on unseen
parametric cases. The same configuration extrapolates to Test C at
|G|=4 with delta=+0.45 (Section 5.5, Table 3).

Contribution 2 (the observable head is load-bearing for both
anti-collapse bases at full scale). Pure SIGReg + BatchNorm without
the observable head (R0 control, D55) fails catastrophically on Test
B at both lambda=0.1 and lambda=0.01: delta_test_b = -0.74 and -0.75
respectively. Pure five-term PLDM without the observable head (R2,
D47) fails at delta_test_b = -0.85. Adding the observable head at
eta=0.01 contributes +0.90 absolute Test B delta to the SIGReg path
(R0 -0.74 to E4 +0.16) and +0.84 absolute to the PLDM path (R2 -0.85
to R1 -0.003). The two regulariser bases are similar in their
dependence on the observable head; neither produces a generalising
latent without it at this data scale.

Contribution 3 (regulariser-asymmetry inversion at scale, with a
controlled-collapse mechanism). At the 5-case smoke scale the
five-term PLDM + observable head reached the HEALTHY quadrant of the
static-versus-dynamic diagnostic with PR around 10 while two-term
SIGReg + observable head plateaued at PR around 3 (D39, Section 4).
The smoke-scale reading was "PLDM is the recommended base." At the
full 41-case scale the reading inverts. SIGReg + observable head
(R3, the Session 7 anchor) generalises to Test B at delta=+0.14;
PLDM + observable head (R1) does not (delta=-0.003). The asymmetry is
robust to PLDM hyperparameter choice: a PLDM run with the paper-tuned
Two-Rooms weights from \cite{pldm} Appendix J.2
(alpha=4.0, beta=6.9, delta=0.75, omega=0.0) is even worse
(E10, D53b: delta_test_b = -0.095). The participation ratio that
looked healthy on PLDM at the smoke scale, where the 5-term loss has
only 5 case labels to memorise, encodes case-specific memorisation
once 41 cases are available; PR_all rises in lockstep with
cross-split degradation in R2's trajectory audit, while PR_within
shrinks (D50). The mechanism behind the SIGReg + OBS win is a
controlled-collapse equilibrium: at lambda=0.01 the SIGReg loss
provides directional pressure that prevents rank-1 collapse without
forcing the encoder to maintain the high-PR isotropy that pure SIGReg
would impose. The PLDM five-term loss cannot be "bought off" cheaply
because all four collapse-prevention terms pull simultaneously, and
the resulting full-scale latent absorbs case-axis variation that does
not transfer to Test B. The LeWM Two-Room intrinsic-dimension
prediction (smaller d closer to the data's intrinsic dim should win)
is not confirmed on this data either: the d sweep at the production
operating point (d in {8, 16, 32}) finds d*=32 best on Test B by
+0.07 over d=8 (D54), with PR_all flat across d. The extra
dimensions help the downstream cross-split probe, not the encoder.

The Test B delta over the (c, t) parametric baseline is the
diagnostic that survives the smoke-to-full-scale inversion. The
participation ratio alone, the headline number in the SIGReg
\cite{lejepa} and PLDM \cite{pldm} literature, is anti-correlated
with generalisation on R2's trajectory and does not separate "encoder
memorised 41 case labels" from "encoder learned dynamics" in this
regime. The static-versus-dynamic decomposition introduced in
Section 4, combined with the (c, t) baseline, gives a more robust
audit.

## 1.3 Honest scope

The paper reports a single locked partition (v1.2), a single
training duration (20k iterations), and a single-seed grid sweep
with a single-seed-variance datapoint at the R3 anchor (seed=42,
Test B delta = +0.121; pass criterion [+0.05, +0.25], D52). Section
5.9 lists these and the other limitations of the evidence base. The
lambda bisection over the finer interval around lambda*=0.01 is
running concurrently with the writing of this manuscript; the
visualisation decoder on the frozen d=32 SIGReg + OBS encoder is
also a Session 9 deliverable. The numbers in Section 5 are the
Session 7 and 8 production runs only; subsequent revisions will
fold in the bisection result and the decoder reconstruction
metrics.

## 1.4 Roadmap

Section 2 reviews related work on observable-augmented autoencoders
for unsteady aerodynamics, on JEPA architectures, and on anti-collapse
regularisation for self-supervised representation learning. Section 3
specifies the data partition, the encoder and predictor architectures,
the two-term SIGReg and five-term PLDM loss compositions, the
observable head, and the diagnostic suite. Section 4 walks through
the four failure modes that pure two-term SIGReg JEPA hit at the
5-case smoke scale, organised as the TRIVIAL / SPREAD_TRIVIAL /
HEALTHY 2x2 of latent rank versus case-versus-dynamics encoding, and
introduces the observable-augmentation rescue. Section 5 reports the
full-scale results: the Session 7 three-way comparison
(R1 PLDM+OBS, R2 PLDM-only, R3 SIGReg+OBS), the Session 8 validation
diagnostics (trajectory audit, head ablation, seed-variance bound),
the (eta x lambda) grid that locates the production operating point,
the latent-dimension sweep, and the R0 control that confirms the
observable head is load-bearing. Section 6 reports the visualisation
decoder trained on the frozen SIGReg + OBS encoder. Section 7 runs
the architecture-spec ablation suite. Section 8 concludes.

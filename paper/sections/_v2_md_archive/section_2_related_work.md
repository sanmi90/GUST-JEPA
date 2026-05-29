# Section 2: Related work

LaTeX-friendly markdown. Approximate target length: 3 to 4 pages.
Citations are placeholder `\cite{key}` tokens whose `key` matches
HANDOFF.md "Key references"; the final BibTeX will resolve them. This
section positions our contribution against three threads: the JEPA
lineage of joint-embedding predictive architectures from pixels, the
observable-augmented autoencoder lineage for fluid reduced-order
modelling (ROM), and the classical and learned ROM literature for
unsteady aerodynamics. Section 2.4 states the gap that the present work
closes.

## 2.1 Joint-embedding predictive architectures (JEPAs)

The architectural family we work in was outlined by LeCun in
"A Path Towards Autonomous Machine Intelligence" (\cite{lecun2022jepa},
2022) as joint-embedding predictive architectures: an encoder maps an
input x to a representation z, a predictor takes a context z_ctx and
an action or temporal increment a and produces a target representation
z_tgt, and the training objective minimises the distance between the
predicted and the actually observed target embedding. The key contrast
with autoencoders or pixel-space world models is that the predictor
operates in latent space and the loss never reconstructs pixels. Two
mechanical consequences follow that the present paper inherits. First,
the encoder is free to discard information that is not predictive of
future latents; second, the predictor cannot "cheat" via pixel-level
shortcuts because no pixel-level signal is available to it. The cost
is that the encoder can collapse to a constant representation that is
trivially predictable, which is why every JEPA in the literature pairs
the predictive loss with an explicit or implicit anti-collapse
mechanism.

The first scaled instantiation in the image domain is I-JEPA (Assran,
Duval, Misra, Bordes, Vincent, Morcos, Ballas, LeCun, CVPR 2023;
\cite{ijepa2023}), which predicts masked-image patches in latent space
from a context patch using a ViT encoder, predictor, and a target
encoder updated via EMA. I-JEPA established that masked-modelling
without pixel reconstruction is competitive on standard image
classification probes and that the predictor must operate on
positional information about the masked regions so the encoder is not
penalised for losing absolute position. We do not use masking in this
paper because temporal continuity of the vorticity field already
supplies the predictive supervision signal, but the masked-image
precedent motivates our design choice that the encoder be
unconditional and the predictor be the conditioned component.

V-JEPA (Bardes, Garrido, Ponce, Rabbat, LeCun, Bojanowski, Ballas,
arXiv:2404.08471, 2024; \cite{vjepa2024}) extended I-JEPA from images
to short video clips and showed that a single-step latent prediction
objective produces useful representations for action recognition. The
follow-up V-JEPA 2 and V-JEPA 2-AC (Assran et al., arXiv:2506.09985,
2025; \cite{vjepa2}) generalised the recipe to longer rollouts with
scheduled sampling: a multi-step open-loop rollout where the predictor
is conditioned on its own previous prediction with probability p that
ramps up during training. We adopt exactly the V-JEPA 2-AC scheduled-
sampling design (H_roll = 8, p linearly ramped from 0 to 1 over the
first half of training; D21 in HANDOFF.md) because it is the cleanest
published recipe for training a predictor to be stable under its own
errors. The vanilla V-JEPA stop-gradient and EMA target encoder are
intentionally omitted here following the LeWM thesis (Section 2.1
below) that an explicit characteristic-function regulariser obviates
the need for either.

LeWM (Maes, Le Lidec, Scieur, LeCun, Balestriero,
"LeWorldModel: Stable End-to-End Joint-Embedding Predictive
Architecture from Pixels," arXiv:2603.19312, March 2026; \cite{lewm})
is the architectural template the present work follows most closely.
LeWM removes both stop-gradient and EMA by replacing them with two
ingredients: a BatchNorm-projected encoder output and a
characteristic-function regulariser (SIGReg) applied to the joint
batch of all encoded frames in a sub-trajectory. The combination is
shown to train stably from pixels on the Two-Room gridworld and the
DMC Walker / Cheetah / Quadruped suites at sample efficiencies
comparable to model-based RL, without any contrastive pairs and
without an auxiliary reconstruction loss. Two LeWM appendices are
directly load-bearing for our work. Appendix A gives the SIGReg
mechanics (M projections, Epps-Pulley test against a Gaussian target
distribution, BatchNorm at the latent boundary). Appendix G specifies
the lambda bisection used to set the regulariser weight: six to eight
evaluations over a logarithmic interval, with each evaluation a full
training run, picking the lambda that maximises a downstream probe
metric. We use this exact protocol in Session 9 over the lambda
interval centred on the (eta x lambda) grid optimum identified in
Section 5.5. The LeWM Two-Room results also include the observation
that SIGReg's effectiveness depends on the relationship between latent
dimension d and the intrinsic dimension of the underlying data: when
d is close to the intrinsic dim, SIGReg's anti-collapse pressure does
not have to "fight itself" by spreading the latent uniformly across
unused dimensions. Section 5.6 of the present work tests this LeWM
prediction empirically on physics data and finds it does not transfer
to the OBS-augmented regime we operate in (D54).

LeJEPA (Balestriero and LeCun, "LeJEPA: Provable and Scalable Self-
Supervised Learning Without the Heuristics," arXiv:2511.08544,
November 2025; \cite{lejepa}) provides the formal SIGReg analysis on
which the LeWM appendix-A implementation rests. The key results are
two: an isotropic-Gaussian latent distribution is the maximum-entropy
distribution under fixed second moments, so penalising deviations
from Gaussian via a characteristic-function test is the
distribution-free analogue of penalising deviations from white-noise
covariance; and SIGReg is invariant to invertible affine
reparametrisations of the latent, which removes the need for any
covariance-shaping component analogous to VICReg's variance and
covariance terms. The LeJEPA paper also gives the clean rationale for
why BatchNorm rather than LayerNorm must be used at the latent
boundary: SIGReg compares the empirical characteristic function of
projections of z to that of a unit Gaussian, and a LayerNorm-projected
z has per-sample unit norm, which collapses the SIGReg target. The
present work adheres strictly to this BN-vs-LN rule (D17) and uses
SIGReg with M = 256 projections and 17 Epps-Pulley knots over [0.2,
4.0] following the LeJEPA reference implementation. The architecture-
agnostic claim of LeJEPA is that any encoder + predictor pair trained
to minimise rollout MSE plus SIGReg on the latent is a complete JEPA;
the present work tests that claim on physics data and reports both a
regime where it holds (SIGReg + observable head, Section 5.2 R3) and
a regime where it does not (pure SIGReg without observable head,
Section 5.7 R0).

PLDM (Sobal, Zhang, Cho, Balestriero, Rudner, LeCun, "Learning from
Reward-Free Offline Data: A Case for Planning with Latent Dynamics
Models," arXiv:2502.14819, February 2025; \cite{pldm}) is the direct
JEPA-from-pixels precursor to LeWM and the principal baseline of the
present work. PLDM trains an encoder + predictor + inverse dynamics
model jointly on offline trajectories without rewards, using a five-
term VICReg-derived objective: a multi-step rollout MSE term L_sim,
two VICReg anti-collapse terms (variance and covariance) on the
rolled-out latents (\cite{vicreg}; Bardes, Ponce, LeCun, ICLR 2022),
a temporal-similarity prediction term L_time_sim, and an inverse-
dynamics-model loss L_idm. The four collapse-prevention terms each
have a tunable weight (alpha, beta, delta, omega in the paper's
notation) plus the implicit unit weight on L_sim. The contrast with
SIGReg is methodological: SIGReg replaces the four collapse-prevention
terms with a single distribution-matching regulariser whose
hyperparameter space is a single scalar lambda (LeWM Appendix G
calibrates this via bisection). The workshop precursor (Sobal,
Jyothir, Jalagam, Carion, Cho, LeCun, "Joint Embedding Predictive
Architectures Focus on Slow Features," arXiv:2211.10831, NeurIPS SSL
workshop 2022; \cite{pldm_workshop}) introduced the slow-feature
motivation for PLDM; D32 in HANDOFF.md records the correction from
the workshop paper to the journal-length arXiv:2502.14819 as the
canonical PLDM citation. The PLDM stress-test (Sobal, Zhang, Cho,
Balestriero, Rudner, LeCun, "Stress-testing Offline Reward-Free
Reinforcement Learning," Robot Learning Workshop 2025;
\cite{pldm_stress}) examines failure modes across out-of-distribution
goals and out-of-distribution dynamics. We add the present work as a
further stress test of PLDM in a non-RL physics-data regime: PLDM with
default unit weights (Section 5.2 R1, R2), PLDM with paper-tuned Two-
Rooms weights (Section 5.5 E10), and the cross-comparison of both
PLDM variants against the SIGReg + observable-head reference confirm
that the 5-term objective produces high-PR but case-mean-dominated
latents that fail to generalise on Test B at full scale (Section 5.4
D50).

The central methodological contrast the present paper owns is
"SIGReg + 2-term proposed" against "VICReg + 5-term PLDM": one
distribution-matching scalar lambda with a published O(log n)
bisection procedure (LeWM Appendix G) versus four collapse-prevention
weights without a published joint-tuning procedure. The Session 5
smoke at five cases (D31) showed both regularisers can collapse on
this data when the case count is small, so the methodological contrast
is regime-dependent. Section 5 of the present paper reports the
regime-specific outcome at 41 cases: at scale the SIGReg + OBS path
generalises (Test B delta +0.131 +/- 0.032 (1-sigma) across three
seeds against the parametric baseline) while both PLDM + OBS variants
do not (default +0.0 to -0.01; paper-tuned -0.10).

## 2.2 Observable-augmented autoencoders for fluid ROM

A parallel literature on fluid-flow reduced-order modelling has
arrived at a similar end-to-end-trained recipe through a different
route. Fukami and Taira ("Grasping extreme aerodynamics on a low-
dimensional manifold," J. Fluid Mech. 2023; arXiv:2305.18394;
\cite{fukami2023}) trained an autoencoder on transonic-airfoil PIV
fields jointly with a lift-prediction head: a convolutional encoder
maps the velocity field to a 3D latent, a convolutional decoder
reconstructs the velocity field, and a small MLP head reads off the
instantaneous lift coefficient CL from the latent. The reported
finding is that the auxiliary CL loss reduces the intrinsic dimension
of the discovered manifold from approximately 10 to 3 and produces
latents that interpolate smoothly between angle-of-attack regimes.
The interpretation we adopt is that an aerodynamic observable head
acts as a low-cost supervised signal that shapes the encoder toward
flow-state-relevant rather than reconstruction-relevant features.

Fukami, Iwatani, Maejima, Asada, Kawai ("Compact Representation of
Transonic Airfoil Buffet Flows with Observable-Augmented Machine
Learning," J. Fluid Mech. 1021, A39, 2025; arXiv:2509.17306;
\cite{fukami2025}) extended the same observable-augmentation recipe
to transonic buffet across multi-source turbulent flow data and
showed that the technique generalises beyond the single-flow regime
of the 2023 paper. The same group's vortex-gust paper (Fukami, Smith,
Taira, "Extreme Vortex-Gust Airfoil Interactions at Reynolds Number
5000," Phys. Rev. Fluids 10, 084703, 2025; \cite{fukami2025prf})
reports DNS results in the same regime as the present work (NACA 0012
at Re = 5000, alpha = 14 degrees, perturbed by Taylor vortices)
without an autoencoder analysis; the 2025 PRF paper is the closest
prior DNS-side reference for our data domain. The combination of the
two Fukami 2025 references makes the observable-augmentation route the
natural baseline against which to compare a JEPA-from-pixels approach
on the same flow.

Solera-Rico, Sanmiguel Vila, Gomez-Lopez, Wang, Almashjary, Dawson,
Vinuesa ("beta-Variational Autoencoders and Transformers for Reduced-
Order Modelling of Fluid Flows," Nat. Commun. 15, 1361, 2024;
\cite{solera2024}) used a closely related dataset family (the gust-
airfoil family at Re = 5000) and proposed a beta-VAE + transformer
ROM. The beta-VAE encoder is trained on reconstruction MSE with a
KL-divergence penalty against an isotropic Gaussian prior; a separate
transformer is trained to predict latent trajectories from a context
window. The reported metric of merit is forecasting horizon at matched
latent dimension, which is also the metric the present paper centres
on. Two design choices distinguish that line of work from the present
JEPA approach. First, beta-VAE conditions the encoder on a
reconstruction objective; the encoder must preserve enough information
to decode the input field, which constrains the representation in a
direction that is orthogonal to the predictive-only objective of JEPA.
Second, the transformer ROM is trained on the frozen VAE latents in a
two-stage protocol, so the encoder cannot specialise its
representation to the dynamical structure of the flow. The present
work treats both decisions as design alternatives to compare against:
Section 5.7 reports a pure-SIGReg reference without the observable
head, which is the closest JEPA analogue to "encoder trained on
predictive objective only," and Section 3.2 inherits the observable
head from Fukami and Taira (\cite{fukami2023}) without inheriting
their reconstruction loss. We do not retrain a beta-VAE baseline as
part of this paper; the comparison is at the conceptual level, with
the published forecasting-horizon numbers from \cite{solera2024}
serving as the reference target.

The recurring pattern across the Fukami and Solera-Rico lineage is
encoder + decoder + observable head trained jointly on reconstruction
MSE plus observable MSE (Fukami) or reconstruction MSE plus KL plus a
separate predictor on the frozen latent (Solera-Rico). The
methodological gap the present paper occupies is "JEPA + observable
head trained jointly on rollout MSE plus observable MSE, without a
reconstruction loss," which is, to our knowledge, untested on this
flow regime. Section 5 reports the outcome.

## 2.3 Classical and learned ROM for unsteady aerodynamics

Reduced-order modelling for unsteady aerodynamics has a long history
that bookends the autoencoder lineage above. The linear baseline is
proper orthogonal decomposition (POD), formalised in the fluid
mechanics setting by Lumley and developed at length in Holmes, Lumley,
Berkooz ("Turbulence, Coherent Structures, Dynamical Systems and
Symmetry," Cambridge, 1996; \cite{holmes1996}). POD computes the
singular value decomposition of a snapshot matrix and projects the
flow onto the d leading singular vectors; the resulting modal basis
is optimal in the $L^2$ sense at preserving energy under the
proper-orthogonal-decomposition norm but does not by itself produce a
dynamical model. The Galerkin reduction of the Navier-Stokes equations
onto the POD basis (reviewed in Rowley and Dawson, "Model Reduction
for Flow Analysis and Control," Annu. Rev. Fluid Mech. 49, 387, 2017;
\cite{rowley2017}) yields a low-dimensional ODE system that can be
integrated forward in time but typically requires explicit closure
modelling for stability outside the training regime. We include POD
at matched latent dimension d = 32 as the linear floor baseline in
the comparison protocol of Section 7.

Dynamic mode decomposition (Schmid, "Dynamic mode decomposition of
numerical and experimental data," J. Fluid Mech. 656, 5, 2010;
\cite{schmid2010}; Tu, Rowley, Luchtenburg, Brunton, Kutz, "On dynamic
mode decomposition: Theory and applications," J. Comput. Dyn. 1, 391,
2014; \cite{tu2014}) approximates the Koopman operator on a finite-
dimensional snapshot basis and produces a linear dynamical model whose
eigenvalues correspond to oscillation frequencies and growth rates.
DMD is data-driven (no projection of the governing equations is
required) and produces a forward-integrable model out of the box, at
the cost of being purely linear. The neural-network extensions
(Constante-Amores and Graham, "Data-Driven State-Space and Koopman
Operator Models of Coherent State Dynamics on Invariant Manifolds,"
J. Fluid Mech. 984, R9, 2024; arXiv:2312.03875;
\cite{constanteamores2024}) lift the linearity restriction by learning
an embedding that linearises the dynamics on an invariant manifold,
which is methodologically close to the JEPA premise of a latent
predictor but inherits the reconstruction loss of the encoder-decoder
backbone.

Neural-network ROMs that are explicitly nonlinear and trained on
reconstruction MSE include Lee and Carlberg ("Model reduction of
dynamical systems on nonlinear manifolds using deep convolutional
autoencoders," J. Comput. Phys. 404, 108973, 2020;
\cite{lee2020}), which trains a convolutional autoencoder and a
latent-space neural ODE jointly on PDE solution snapshots, and Maulik,
Lusch, Balaprakash ("Reduced-order modeling of advection-dominated
systems with recurrent neural networks and convolutional
autoencoders," Phys. Fluids 33, 037106, 2021; \cite{maulik2021}),
which uses a convolutional autoencoder for the spatial reduction and
an LSTM for the latent forecasting on advection-dominated PDEs. The
common pattern is encoder + decoder + recurrent or ODE-based latent
predictor, all trained on reconstruction MSE. The Hasegawa, Fukami,
Murata, Fukagata line of work (Hasegawa, Fukami, Murata, Fukagata,
"CNN-LSTM based reduced order modeling of two-dimensional unsteady
flows around a circular cylinder at different Reynolds numbers,"
Fluid Dyn. Res. 52, 065501, 2020; \cite{hasegawa2020}) is a closely
related example specifically targeting unsteady cylinder wake flow
and reports forecasting accuracy at Reynolds numbers held out from the
training set, which is the parametric-interpolation question that
maps directly onto our Test B setting.

These neural-network ROMs differ from the present work in the same way
the Fukami and Solera-Rico autoencoder lineage does: the encoder is
trained on a reconstruction signal, which constrains the latent to
preserve pixel-level detail at the cost of latent capacity for
predictive structure. The transformer-based variants (Solera-Rico et
al., \cite{solera2024}; and the AeroJEPA preprint from the Vinuesa
group expected in 2026) replace the LSTM with a transformer predictor
on a frozen latent, but the underlying autoencoder is still trained on
reconstruction. None of the above references trains the encoder on a
predictive-only latent objective without a reconstruction loss.

## 2.4 The gap this paper closes

Two gaps follow from the survey above. First, all prior fluid-ROM work
on the gust-airfoil regime at Re = 5000 is either reconstruction-based
(POD, DMD, beta-VAE + transformer) or observable-augmented
autoencoders (Fukami 2023, Fukami 2025). None of it uses a JEPA-style
predictive-only objective on the same flow. The present work applies
JEPA-from-pixels to gust-airfoil DNS at the same regime and reports
forecasting horizon and probing $r^2$ at matched latent dimension
against the published Fukami and Solera-Rico numbers. Second, the
LeWM controlled-collapse mechanism (SIGReg with small lambda plus an
observable head; \cite{lewm}) has not been tested on low-intrinsic-
dim physics data outside the LeWM Two-Room gridworld. The Two-Room
prediction (\cite{lewm} Section 5) is that SIGReg works best when the
latent dimension d is close to the intrinsic dimension of the data.
Section 5.6 of the present work tests this prediction on a physics
dataset with intrinsic dimension estimated at 5 to 10 (Section 4) and
finds that the prediction does not extend to the SIGReg + observable-
head regime: d = 32 wins the Test B delta against d = 8 and d = 16 by
+0.07 absolute, and the participation ratio is flat in d across the
sweep, indicating that the encoder uses approximately the same
effective dimensionality regardless of the available latent budget
(D54 in HANDOFF.md).

The present paper closes both gaps. We identify the SIGReg +
observable-head + BatchNorm configuration at (eta*, lambda*) =
(0.01, 0.01) and d = 32 as the production operating point, and we
report a mean Test B delta of +0.131 +/- 0.032 (1-sigma across three
seeds) against the parametric (c, t) baseline on 28 held-out
encounters at unseen interior (G, D, Y) values (Section 5.5 and
Section 5.8). The reading of the LeWM intrinsic-dimension mechanism
in this regime is that the observable head, not SIGReg, is the
dominant regulariser at the operating point we identify (Section 5.5);
SIGReg at lambda = 0.01 provides residual directional pressure but
its gradient is small enough that the encoder satisfies it without
maintaining the high-PR isotropy that the LeWM Two-Room regime
exhibits. The methodological contribution that survives the smoke-to-
full-scale inversion (Section 5.2) is therefore not "JEPA with SIGReg
alone works on physics data," which Section 5.7 explicitly refutes,
but "JEPA with SIGReg plus an observable head generalises on
parametric-interpolation Test B at scale while PLDM with the same
observable head does not," together with the diagnostic-suite
contribution (Section 5.4) that the (c, t) parametric baseline plus
Test-B-delta-over-baseline is the diagnostic that survives the
smoke-to-full-scale inversion of the participation-ratio proxy.

## Open writing TODO

- Confirm the AeroJEPA preprint citation once the embargo lifts; the
  Vinuesa-group preprint is referenced in Section 2.3 but no arXiv ID
  is yet public. Hold a placeholder `\cite{aerojepa2026}`.
- Decide whether the workshop precursor \cite{pldm_workshop} merits a
  full sentence in Section 2.1 or only a parenthetical in Section 2.4.
  Current draft folds it into the PLDM paragraph in Section 2.1.
- Confirm the exact J. Fluid Mech. volume/page for Hasegawa et al.
  2020. The 65501 page number is taken from the Fluid Dyn. Res.
  volume; the title and authors are correct but the journal
  cross-reference should be verified during final BibTeX preparation.
- Cross-check that the Constante-Amores and Graham 2024 citation
  matches the J. Fluid Mech. 984, R9 entry in HANDOFF.md "Key
  references"; it does, but the arXiv:2312.03875 ID should appear in
  the BibTeX entry alongside the journal reference.

# Abstract

Approximate target length: 200 words. Three contribution claims with
their headline numbers. The bisection winner from Session 9 will
update the lambda value reported here once Step 1 lands.

---

Predicting the lift response of an airfoil to a transverse vortex gust
across a parametric envelope of gust strength, core diameter, and
wall-normal offset requires reduced-order models that capture both
the case-mean morphology and the encounter-to-encounter dynamics on
unseen parameters. We apply a Joint-Embedding Predictive Architecture
(JEPA) trained end-to-end from mid-plane vorticity fields to a direct
numerical simulation database of a NACA 0012 airfoil at angle of
attack alpha=14 degrees and chord Reynolds number Re=5000, with
Taylor-vortex gusts spanning 41 training cases (138 encounters) and
10 held-out cases (52 encounters across two parametric strata). At
the production operating point of latent dimension d=32, observable-
head weight eta=0.01, and SIGReg weight lambda=0.01, the JEPA latent
generalises to the held-out parametric stratum with a downstream
lift-prediction Test B advantage of +0.16 over the canonical
(condition, time) parametric baseline at matched probe capacity. We
report three findings. First, a controlled-collapse regime in which
SIGReg with small lambda combines with a future-lift auxiliary head
to produce a low participation-ratio latent that nonetheless
generalises across the parametric envelope. Second, the observable
head is load-bearing: removing it collapses the Test B advantage to
-0.74 absolute regardless of the anti-collapse regulariser, a +0.90
swing for SIGReg and +0.84 for the PLDM five-term objective. Third,
the smoke-scale finding that PLDM outperforms SIGReg inverts at full
scale and full data, with SIGReg+OBS leading PLDM+OBS by +0.16
absolute on the Test B parametric stratum; we trace the inversion to
the controlled-collapse mechanism and document that the LeWM Two-Room
intrinsic-dimension prediction does not transfer to this regime.

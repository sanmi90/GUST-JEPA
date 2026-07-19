
### D-W8 (Wu, Balestriero & Levine arXiv:2606.02572 VISReg assessed: optional s5.1 cite, NOT methods) (2026-07-19, Session 39)

Carlos pointed at "VISReg: Variance-Invariance-Sketching Regularization for
JEPA training" (Wu, Balestriero & Levine, cs.CV, 2026-06-01; the LeJEPA
group). VISReg = VICReg's variance term + a Sliced-Wasserstein sketching
objective replacing covariance, pitched against SIGReg ("lacks flexibility,
vanishing gradients under collapse"; claims low-rank-regime resilience).
ASSESSMENT: does not threaten the collapse result. The cube's no-lift cells
collapsed under a fixed, disclosed SIGReg configuration and the
objective-free supervised control isolates SUPERVISION, not the regulariser,
as what supplies readability and the protected subspace; that logic is
regulariser-agnostic (any shape-targeting term, VISReg included, targets a
fixed distribution blind to the observables). Its evidence is
ImageNet-class vision, not physics. VERDICT: (i) do NOT cite in s3.1, the
D-W2 logic (a methods cite invites "why not VISReg"; the anti-collapse
choice is defended as fixed/disclosed/controlled, not optimal); (ii)
OPTIONAL one-clause cite in s5.1 next to broustail2026: the SSL lineage
itself now concedes isotropic-Gaussian sketching alone is insufficiently
flexible, an in-family support for the supervision-supplies-geometry
mechanism; Carlos decides, and a full-text read to verify their SIGReg
characterisation must precede any citation landing; (iii) PARKED for paper
2 / Track W-eq: a low-rank-resilient regulariser addresses exactly the
D29/D31 low-intrinsic-dim failure mode and could retire the VICReg
auto-fallback in the follow-up campaign's anti-collapse story.

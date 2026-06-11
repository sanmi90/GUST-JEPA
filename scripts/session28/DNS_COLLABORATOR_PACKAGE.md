# DNS resolution package request (Session 28, Track A3; HANDOFF D180)

Ready-to-send draft for the simulation collaborators (Miro / Lehmkuhl, SOD2D).
Carlos sends this; log the actual send date in HANDOFF D180. The row list below
is byte-aligned with the manuscript's author-fill checklist
(`paper/sections/section_2_flow_and_data.tex`, table `tab:dns_pending`).

---

Subject: SOD2D resolution details needed for the vortex-gust JFM submission (Table 1)

Dear Oriol, dear Miro,

we are preparing the JFM submission of the vortex-gust latent-model paper built
on your SOD2D simulations of the NACA 0012 at alpha = 14 deg, Re = 5000 (the
Fukami et al. 2025 configuration, gusts parametrised by (G, D, Y)). The solver
and resolution section (Table 1) is yours, and the referees will hold it to the
DNS standard, so we would rather report your numbers verbatim than approximate
anything on our side. Could you provide the following rows?

1. Free-stream Mach number used in the low-Mach formulation, or an explicit
   incompressible-limit confirmation.
2. Computational domain extents and the spanwise extent L_z/c.
3. Element count and solution-point count (polynomial order included), for the
   periodic campaign and, if different, the run3 campaign (the re-simulated
   finer-dt batch you delivered in June).
4. Minimum wall-normal spacing at the wing in wall units (Delta y+ at the first
   solution point), demonstrating the near-wall layer is resolved without a
   wall model.
5. Time step Delta t * u_inf / c and the maximum CFL, for both campaigns (the
   run3 re-simulation changed dt; both values should appear).
6. Gust-release station x_0/c. Fukami et al. release at x_0/c = -2; our
   manuscript must state our own station explicitly rather than imply it by
   citation, and our wall-normal envelope is |Y| <= 0.4 versus their 0.3, so
   the exact release geometry matters.
7. Grid AND time-step sensitivity evidence: whatever you have (a coarser/finer
   pair on forces, a dt halving check, or equivalent). One paragraph and one
   small table suffice, but the referees will ask for it.

As the exemplar of the level of detail the lineage reports, the attached
Fukami, Smith and Taira (PRF 10, 084703, 2025) Section II runs LES on three
grids (8M / 20M / 40M cells) and tabulates mean and rms C_L and C_D per grid
against Rolandi et al. (2025) and Gupta et al. (2023) (their Fig. 2(b)). One
important asymmetry: they report LES with a Vreman subgrid model, while we
claim DNS (no subgrid model), so the resolution-evidence bar for us is HIGHER,
not lower; the sensitivity row is the one a referee can hold the submission on.

For your reference, our own undisturbed-flow statistics from the delivered
Baseline case (computed independently on the shared data) are: mean C_L, rms
C_L, mean C_D and the lift-spectrum Strouhal peak; we will print them next to
the PRF 2025 fine-grid values (C_L = 0.737, C_L,rms = 0.116, C_D = 0.249) so
your Table 1 rows and our validation block appear together. [The exact numbers
are inserted from outputs/session28/undisturbed_stats.json once computed.]

We will not paraphrase: the rows go into the paper as you write them, with you
as the corresponding authors for the solver section.

Thank you, and best regards,
Carlos

---

Checklist (internal):
- [ ] Undisturbed numbers inserted from outputs/session28/undisturbed_stats.json
- [ ] PRF 2025 PDF attached (FukamiGustRe5000.pdf)
- [ ] Sent (date -> HANDOFF D180)
- [ ] Replies integrated into tab:dns_pending / Table 1 (Phase D; \pending{} rows
      stand untouched until then; NEVER fabricate these rows)
